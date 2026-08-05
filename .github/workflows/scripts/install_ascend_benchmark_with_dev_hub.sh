#!/bin/bash
set -euo pipefail

WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}
VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO:-$WORKSPACE_ROOT}
VLLM_HUST_REPO=${VLLM_HUST_REPO:-$WORKSPACE_ROOT/vllm-hust}
VLLM_HUST_BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-$WORKSPACE_ROOT/vllm-hust-benchmark}
VLLM_HUST_CONDA_ENV=${VLLM_HUST_CONDA_ENV:-vllm-hust-dev}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
ASCEND_BENCHMARK_STACK_MARKER_VERSION=${ASCEND_BENCHMARK_STACK_MARKER_VERSION:-2026-07-17-install-only-v1}
ASCEND_BENCHMARK_TRITON_ASCEND_INDEX_URL=${ASCEND_BENCHMARK_TRITON_ASCEND_INDEX_URL:-https://mirrors.huaweicloud.com/ascend/repos/pypi}
CURRENT_USER_NAME="$(id -un 2>/dev/null || printf '%s' "${USER:-runner}")"
CURRENT_USER_HOME="$(getent passwd "$CURRENT_USER_NAME" 2>/dev/null | cut -d: -f6 || true)"
CONDA_BIN=""

log() {
  printf '[prepare-bootstrap] %s\n' "$*"
}

write_github_env() {
  local name="$1"
  local value="$2"

  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s=%s\n' "$name" "$value" >> "$GITHUB_ENV"
  fi
}

ensure_home_dirs() {
  if [[ -z "${HOME:-}" || ! -d "${HOME:-/nonexistent}" || ! -w "${HOME:-/nonexistent}" ]]; then
    if [[ -n "${CI_HOME:-}" ]]; then
      export HOME="$CI_HOME"
    elif [[ -n "$CURRENT_USER_HOME" ]]; then
      export HOME="$CURRENT_USER_HOME"
    else
      export HOME="${RUNNER_TEMP:-/tmp}/${CURRENT_USER_NAME:-runner}-home"
    fi
  fi

  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
  export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$XDG_CACHE_HOME/pip}"

  mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$PIP_CACHE_DIR"
  chmod 700 "$HOME"
}

resolve_conda_bin() {
  local candidate

  for candidate in \
    "${CONDA_EXE:-}" \
    "$(command -v conda 2>/dev/null || true)" \
    "${CI_HOME:-}/miniconda3/bin/conda" \
    "${HOME:-}/miniconda3/bin/conda" \
    "${CURRENT_USER_HOME:-}/miniconda3/bin/conda" \
    "/opt/conda/bin/conda" \
    "/usr/local/miniconda3/bin/conda" \
    "/usr/local/anaconda3/bin/conda"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "conda executable not found for benchmark bootstrap" >&2
  return 1
}

resolve_conda_env_prefix() {
  local candidate_prefix
  local resolved_prefix

  if [[ -n "${VLLM_HUST_CONDA_PREFIX:-}" && -x "${VLLM_HUST_CONDA_PREFIX}/bin/python" ]]; then
    printf '%s\n' "${VLLM_HUST_CONDA_PREFIX}"
    return 0
  fi

  resolved_prefix="$("$CONDA_BIN" env list 2>/dev/null | awk -v env_name="$VLLM_HUST_CONDA_ENV" '$1 == env_name {print $NF; exit}')"
  if [[ -n "$resolved_prefix" && -x "$resolved_prefix/bin/python" ]]; then
    printf '%s\n' "$resolved_prefix"
    return 0
  fi

  for candidate_prefix in \
    "${CI_HOME:-}/miniconda3/envs/${VLLM_HUST_CONDA_ENV}" \
    "${HOME:-}/miniconda3/envs/${VLLM_HUST_CONDA_ENV}" \
    "${CURRENT_USER_HOME:-}/miniconda3/envs/${VLLM_HUST_CONDA_ENV}" \
    "/opt/conda/envs/${VLLM_HUST_CONDA_ENV}" \
    "/usr/local/miniconda3/envs/${VLLM_HUST_CONDA_ENV}" \
    "/usr/local/anaconda3/envs/${VLLM_HUST_CONDA_ENV}"; do
    if [[ -n "$candidate_prefix" && -x "$candidate_prefix/bin/python" ]]; then
      printf '%s\n' "$candidate_prefix"
      return 0
    fi
  done

  return 1
}

ensure_conda_env() {
  local env_prefix
  local python_bin

  CONDA_BIN="$(resolve_conda_bin 2>/dev/null || true)"
  if [[ -z "$CONDA_BIN" ]]; then
    python_bin="${VLLM_HUST_PYTHON_BIN:-$(command -v python3 2>/dev/null || true)}"
    if [[ -z "$python_bin" || ! -x "$python_bin" ]]; then
      echo "Neither conda nor a usable python3 executable is available for benchmark bootstrap" >&2
      return 1
    fi
    VLLM_HUST_PYTHON_BIN="$python_bin"
    VLLM_HUST_CONDA_PREFIX="$($python_bin -c 'import sys; print(sys.prefix)' 2>/dev/null)"
    if [[ -z "$VLLM_HUST_CONDA_PREFIX" ]]; then
      echo "Could not resolve the container Python prefix: $python_bin" >&2
      return 1
    fi
    export VLLM_HUST_PYTHON_BIN VLLM_HUST_CONDA_PREFIX
    write_github_env "VLLM_HUST_CONDA_PREFIX" "$VLLM_HUST_CONDA_PREFIX"
    write_github_env "VLLM_HUST_PYTHON_BIN" "$VLLM_HUST_PYTHON_BIN"
    log "Conda is unavailable; reusing container Python: $VLLM_HUST_PYTHON_BIN"
    return 0
  fi

  env_prefix="$(resolve_conda_env_prefix || true)"

  if [[ -z "$env_prefix" ]]; then
    log "Creating benchmark conda env: $VLLM_HUST_CONDA_ENV (python=$PYTHON_VERSION)"
    "$CONDA_BIN" create -y -n "$VLLM_HUST_CONDA_ENV" "python=$PYTHON_VERSION" pip
    env_prefix="$(resolve_conda_env_prefix)"
  else
    log "Reusing benchmark conda env: $VLLM_HUST_CONDA_ENV ($env_prefix)"
  fi

  export CONDA_EXE="$CONDA_BIN"
  export VLLM_HUST_CONDA_PREFIX="$env_prefix"
  export VLLM_HUST_PYTHON_BIN="$env_prefix/bin/python"
  write_github_env "VLLM_HUST_CONDA_PREFIX" "$VLLM_HUST_CONDA_PREFIX"
  write_github_env "VLLM_HUST_PYTHON_BIN" "$VLLM_HUST_PYTHON_BIN"
}

run_env_cmd() {
  env \
    HOME="$HOME" \
    XDG_CACHE_HOME="$XDG_CACHE_HOME" \
    XDG_CONFIG_HOME="$XDG_CONFIG_HOME" \
    PIP_CACHE_DIR="$PIP_CACHE_DIR" \
    "$@"
}

run_env_pip() {
  run_env_cmd "$VLLM_HUST_PYTHON_BIN" -m pip "$@"
}

collect_unsatisfied_requirements() {
  run_env_cmd "$VLLM_HUST_PYTHON_BIN" - "$@" <<'PY'
import importlib.metadata as metadata
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

installed = {}
for dist in metadata.distributions():
    name = dist.metadata.get("Name")
    if not name:
        continue
    installed[canonicalize_name(name)] = dist.version

unsatisfied = []
for raw in sys.argv[1:]:
    requirement = Requirement(raw)
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    if requirement.extras:
        unsatisfied.append(raw)
        continue
    version = installed.get(canonicalize_name(requirement.name))
    if version is None:
        unsatisfied.append(raw)
        continue
    if requirement.specifier and version not in requirement.specifier:
        unsatisfied.append(raw)

if unsatisfied:
    print("\n".join(unsatisfied))
PY
}

ensure_python_requirements() {
  local description="$1"
  shift
  local requirement_specs=("$@")
  local missing_specs=()

  mapfile -t missing_specs < <(collect_unsatisfied_requirements "${requirement_specs[@]}" || true)
  if (( ${#missing_specs[@]} == 0 )); then
    log "Reusing installed $description"
    return 0
  fi

  log "Installing missing $description"
  run_env_pip install "${missing_specs[@]}"
}

ensure_triton_ascend() {
  local triton_ascend_spec="triton-ascend==3.2.1"
  local missing_specs=()

  mapfile -t missing_specs < <(collect_unsatisfied_requirements "$triton_ascend_spec" || true)
  if (( ${#missing_specs[@]} == 0 )); then
    log "Reusing installed $triton_ascend_spec"
    return 0
  fi

  log "Installing $triton_ascend_spec from Ascend PyPI mirror: $ASCEND_BENCHMARK_TRITON_ASCEND_INDEX_URL"
  run_env_pip install --no-deps --index-url "$ASCEND_BENCHMARK_TRITON_ASCEND_INDEX_URL" "$triton_ascend_spec"
}

read_requirement_specs_from_file() {
  local requirements_file="$1"

  awk '
    /^[[:space:]]*#/ || /^[[:space:]]*$/ { next }
    {
      line = $0
      sub(/[[:space:]]+#.*$/, "", line)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      if (line == "" || line ~ /^-/) {
        next
      }
      print line
    }
  ' "$requirements_file"
}

ensure_bootstrap_python_tools() {
  log "Ensuring bootstrap packaging tools are available in $VLLM_HUST_CONDA_ENV"
  run_env_pip install --upgrade pip "packaging>=24.2" "setuptools>=77,<81" wheel
}

ensure_conda_runtime_libs() {
  if [[ -z "$CONDA_BIN" ]]; then
    log "Skipping conda runtime library installation in container-native Python mode"
    return 0
  fi
  if [[ -f "$VLLM_HUST_CONDA_PREFIX/lib/libstdc++.so.6" && -f "$VLLM_HUST_CONDA_PREFIX/lib/libgcc_s.so.1" ]]; then
    log "Reusing conda runtime libs from $VLLM_HUST_CONDA_PREFIX/lib"
    return 0
  fi

  log "Installing conda runtime libs into $VLLM_HUST_CONDA_ENV"
  "$CONDA_BIN" install -y -n "$VLLM_HUST_CONDA_ENV" libgcc-ng libstdcxx-ng
}

detect_cann_major_version() {
  local candidate
  local version

  for candidate in \
    "${ASCEND_HOME_PATH:-}/version.info" \
    "${ASCEND_HOME_PATH:-}/runtime/version.info" \
    "/usr/local/Ascend/ascend-toolkit/latest/version.info" \
    "/usr/local/Ascend/ascend-toolkit/latest/runtime/version.info" \
    "${VLLM_HUST_CONDA_PREFIX:-}/Ascend/cann/version.info" \
    "${VLLM_HUST_CONDA_PREFIX:-}/Ascend/cann/runtime/version.info"; do
    if [[ -f "$candidate" ]]; then
      version="$(grep -oE '[0-9]+[.][0-9]+([.][0-9]+)?' "$candidate" | head -n 1 || true)"
      if [[ -n "$version" ]]; then
        printf '%s\n' "${version%%.*}"
        return 0
      fi
    fi
  done

  return 1
}

ascend_custom_kernel_build_prereqs_present() {
  local tool

  for tool in git cmake make; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      return 1
    fi
  done

  if ! command -v g++ >/dev/null 2>&1 && ! command -v c++ >/dev/null 2>&1; then
    return 1
  fi

  return 0
}

resolve_compile_custom_kernels() {
  local requested="${COMPILE_CUSTOM_KERNELS:-auto}"
  local cann_major

  if [[ "$requested" == "auto" ]]; then
    if [[ "${HUST_ASCEND_TBE_AVAILABLE:-1}" != "1" ]]; then
      printf '0\n'
      return 0
    fi

    cann_major="$(detect_cann_major_version || true)"
    if [[ "$cann_major" == "9" ]] && ascend_custom_kernel_build_prereqs_present; then
      printf '1\n'
    else
      printf '0\n'
    fi
    return 0
  fi

  if [[ ! "$requested" =~ ^[0-9]+$ ]]; then
    echo "Invalid COMPILE_CUSTOM_KERNELS: $requested" >&2
    return 2
  fi

  printf '%s\n' "$requested"
}

ensure_ascend_catlass_submodule_ready() {
  local submodule_relative_path="csrc/third_party/catlass"
  local submodule_path="$VLLM_ASCEND_HUST_REPO/$submodule_relative_path"

  if [[ -e "$submodule_path/CMakeLists.txt" || -e "$submodule_path/README.md" ]]; then
    return 0
  fi

  if [[ ! -d "$VLLM_ASCEND_HUST_REPO/.git" && ! -f "$VLLM_ASCEND_HUST_REPO/.git" ]]; then
    log "Skipping $submodule_relative_path initialization because $VLLM_ASCEND_HUST_REPO has no git metadata"
    return 0
  fi

  log "Initializing Ascend submodule: $submodule_relative_path"
  git -C "$VLLM_ASCEND_HUST_REPO" submodule update --init --recursive "$submodule_relative_path"
}

patch_triton_ascend_for_cann9() {
  local python_dir
  local triton_backends
  local npu_utils
  local cann_major

  cann_major="$(detect_cann_major_version || true)"
  if [[ "$cann_major" != "9" ]]; then
    return 0
  fi

  python_dir="$(find "$VLLM_HUST_CONDA_PREFIX/lib" -maxdepth 1 -type d -name 'python*' | sort | head -n 1 || true)"
  if [[ -z "$python_dir" ]]; then
    return 0
  fi

  triton_backends="$python_dir/site-packages/triton/backends/ascend"
  npu_utils="$triton_backends/npu_utils.cpp"
  if [[ ! -f "$npu_utils" ]]; then
    return 0
  fi

  if grep -q 'RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE' "$npu_utils" 2>/dev/null; then
    log "Patching triton-ascend npu_utils.cpp for CANN 9 compatibility"
    sed -i '/RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE/d' "$npu_utils"
    find "$triton_backends" -path '*npu_utils*.so' -delete 2>/dev/null || true
  fi
}

install_benchmark_baseline_stack() {
  local marker_root="$VLLM_HUST_CONDA_PREFIX"
  if [[ ! -w "$marker_root" ]]; then
    marker_root="${CI_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}}/ascend-benchmark"
    mkdir -p "$marker_root"
  fi
  local marker_file="$marker_root/.ascend-benchmark-install-only-stack"
  local vllm_hust_requirements_file="$VLLM_HUST_REPO/requirements/common.txt"
  local requirements_hash
  local expected_marker
  local validation_specs=(
    "torch==2.10.0"
    "torch-npu==2.10.0"
    "torchvision==0.25.0"
    "torchaudio==2.10.0"
    "triton-ascend==3.2.1"
    "huggingface_hub>=0.20"
    "jsonschema>=4"
  )
  local core_stack_specs=(
    "numpy<2.0.0"
    "torch==2.10.0"
    "torch-npu==2.10.0"
    "torchvision==0.25.0"
    "torchaudio==2.10.0"
  )
  local extra_runtime_specs=(
    "attrs"
    "cmake>=3.26"
    "decorator"
    "googleapis-common-protos"
    "msgpack"
    "numba"
    "packaging>=24.2"
    "pandas"
    "pandas-stubs"
    "pybind11"
    "quart"
    "scipy"
    "setuptools-scm>=8"
    "setuptools-rust>=1.9.0"
    "xgrammar>=0.1.30"
    "compressed_tensors>=0.11.0"
    "arctic-inference==0.1.1"
    "transformers==5.5.4"
    "jsonschema>=4"
    "huggingface_hub>=0.20"
  )
  local vllm_hust_runtime_specs=()
  local validation_missing=()

  requirements_hash="$(sha256sum "$vllm_hust_requirements_file" | awk '{print $1}')"
  expected_marker="${ASCEND_BENCHMARK_STACK_MARKER_VERSION}:${requirements_hash}"

  if [[ -f "$marker_file" ]] && grep -qx "$expected_marker" "$marker_file"; then
    mapfile -t validation_missing < <(collect_unsatisfied_requirements "${validation_specs[@]}" || true)
    if (( ${#validation_missing[@]} == 0 )); then
      log "Reusing benchmark baseline stack marker: $marker_file"
      return 0
    fi
    log "Benchmark baseline stack marker is stale; reinstalling missing runtime pieces"
  fi

  log "Installing benchmark baseline runtime stack into $VLLM_HUST_CONDA_ENV"
  ensure_python_requirements "pinned torch stack" "${core_stack_specs[@]}"
  ensure_triton_ascend
  mapfile -t vllm_hust_runtime_specs < <(read_requirement_specs_from_file "$vllm_hust_requirements_file")
  ensure_python_requirements "vllm-hust runtime requirements" "${vllm_hust_runtime_specs[@]}"
  ensure_python_requirements "Ascend benchmark extra runtime deps" "${extra_runtime_specs[@]}"
  printf '%s\n' "$expected_marker" > "$marker_file"
}

install_editable_repo_no_deps() {
  local repo_path="$1"
  shift

  log "Installing editable package from: $repo_path"
  run_env_cmd env "$@" \
    "$VLLM_HUST_PYTHON_BIN" -m pip install -e "$repo_path" --no-build-isolation --no-deps
}

if [[ ! -f "$VLLM_HUST_REPO/pyproject.toml" ]]; then
  echo "vllm-hust checkout not found: $VLLM_HUST_REPO" >&2
  exit 2
fi

if [[ ! -f "$VLLM_HUST_BENCHMARK_REPO/pyproject.toml" ]]; then
  echo "vllm-hust-benchmark checkout not found: $VLLM_HUST_BENCHMARK_REPO" >&2
  exit 2
fi

if [[ ! -f "$VLLM_ASCEND_HUST_REPO/pyproject.toml" ]]; then
  echo "vllm-ascend-hust checkout not found: $VLLM_ASCEND_HUST_REPO" >&2
  exit 2
fi

if [[ ! -f "$WORKSPACE_ROOT/ascend-runtime-manager/pyproject.toml" ]]; then
  echo "ascend-runtime-manager checkout not found under workspace: $WORKSPACE_ROOT/ascend-runtime-manager" >&2
  exit 2
fi

ensure_home_dirs
ensure_conda_env
ensure_conda_runtime_libs
ensure_bootstrap_python_tools

if [[ -f "$VLLM_ASCEND_HUST_REPO/scripts/use_single_ascend_env.sh" ]]; then
  export HUST_REQUIRE_CANN_TBE="${HUST_REQUIRE_CANN_TBE:-0}"
  # shellcheck source=/dev/null
  source "$VLLM_ASCEND_HUST_REPO/scripts/use_single_ascend_env.sh"
fi

resolved_compile_custom_kernels="$(resolve_compile_custom_kernels)"
export COMPILE_CUSTOM_KERNELS="$resolved_compile_custom_kernels"
if [[ "$resolved_compile_custom_kernels" == "0" ]]; then
  log "Using install-only repo bootstrap (no quickstart; editable --no-deps installs)"
  log "COMPILE_CUSTOM_KERNELS=auto resolved to lightweight mode"
else
  ensure_ascend_catlass_submodule_ready
  log "Using install-only repo bootstrap (no quickstart; editable --no-deps installs)"
  log "Using configured COMPILE_CUSTOM_KERNELS=$resolved_compile_custom_kernels"
fi

install_benchmark_baseline_stack
patch_triton_ascend_for_cann9

install_editable_repo_no_deps "$VLLM_HUST_REPO" \
  VLLM_TARGET_DEVICE=empty \
  VLLM_USE_PRECOMPILED=0 \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0
install_editable_repo_no_deps "$VLLM_HUST_BENCHMARK_REPO"
bash "$VLLM_ASCEND_HUST_REPO/scripts/install_local_ascend_plugin.sh" "$VLLM_ASCEND_HUST_REPO"

write_github_env "COMPILE_CUSTOM_KERNELS" "${COMPILE_CUSTOM_KERNELS:-auto}"
