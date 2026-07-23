#!/usr/bin/env bash

_HUST_MANAGER_HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_HUST_MANAGER_REPO_ROOT="$(cd "${_HUST_MANAGER_HELPER_DIR}/.." && pwd)"
_HUST_MANAGER_WORKSPACE_ROOT="$(cd "${_HUST_MANAGER_REPO_ROOT}/.." && pwd)"
_HUST_DEFAULT_HF_ENDPOINT="${HUST_DEFAULT_HF_ENDPOINT:-https://hf-mirror.com}"
_HUST_DEFAULT_GET_PIP_URL="${HUST_GET_PIP_URL:-https://bootstrap.pypa.io/get-pip.py}"

_resolve_hust_ascend_manager_conda_python() {
  local env_prefix="${VLLM_HUST_CONDA_PREFIX:-}"
  local env_name="${VLLM_HUST_CONDA_ENV:-vllm-hust-dev}"
  local current_user_name
  local current_user_home
  local conda_bin
  local conda_root
  local candidate_prefix
  local ci_home
  local resolved_prefix

  if [[ -n "${env_prefix}" && -x "${env_prefix}/bin/python" ]]; then
    printf '%s\n' "${env_prefix}/bin/python"
    return 0
  fi

  ci_home="${CI_HOME:-}"
  if [[ -n "${ci_home}" ]]; then
    for candidate_prefix in \
      "${ci_home}/miniconda3/envs/${env_name}" \
      "${ci_home}/anaconda3/envs/${env_name}" \
      "${ci_home}/mambaforge/envs/${env_name}" \
      "${ci_home}/miniforge3/envs/${env_name}"; do
      if [[ -x "${candidate_prefix}/bin/python" ]]; then
        printf '%s\n' "${candidate_prefix}/bin/python"
        return 0
      fi
    done
  fi

  current_user_name="$(id -un 2>/dev/null || printf '%s' "${USER:-}")"
  current_user_home="$(getent passwd "${current_user_name}" 2>/dev/null | cut -d: -f6 || true)"

  if [[ -n "${current_user_home}" ]]; then
    for candidate_prefix in \
      "${current_user_home}/miniconda3/envs/${env_name}" \
      "${current_user_home}/anaconda3/envs/${env_name}" \
      "${current_user_home}/mambaforge/envs/${env_name}" \
      "${current_user_home}/miniforge3/envs/${env_name}"; do
      if [[ -x "${candidate_prefix}/bin/python" ]]; then
        printf '%s\n' "${candidate_prefix}/bin/python"
        return 0
      fi
    done
  fi

  if command -v conda >/dev/null 2>&1; then
    resolved_prefix="$(conda env list 2>/dev/null | awk -v env_name="${env_name}" '$1 == env_name {print $NF; exit}')"
    if [[ -n "${resolved_prefix}" && -x "${resolved_prefix}/bin/python" ]]; then
      printf '%s\n' "${resolved_prefix}/bin/python"
      return 0
    fi

    conda_bin="$(command -v conda)"
    conda_root="$(cd "$(dirname "${conda_bin}")/.." && pwd)"
    for candidate_prefix in \
      "${conda_root}/envs/${env_name}" \
      "${conda_root}/../envs/${env_name}"; do
      if [[ -x "${candidate_prefix}/bin/python" ]]; then
        printf '%s\n' "${candidate_prefix}/bin/python"
        return 0
      fi
    done
  fi

  for candidate_prefix in \
    "/opt/conda/envs/${env_name}" \
    "/usr/local/miniconda3/envs/${env_name}" \
    "/usr/local/anaconda3/envs/${env_name}"; do
    if [[ -x "${candidate_prefix}/bin/python" ]]; then
      printf '%s\n' "${candidate_prefix}/bin/python"
      return 0
    fi
  done

  return 1
}

_resolve_hust_ascend_manager_python() {
  if [[ -n "${VLLM_HUST_PYTHON_BIN:-}" ]]; then
    if [[ -x "${VLLM_HUST_PYTHON_BIN}" ]]; then
      printf '%s\n' "${VLLM_HUST_PYTHON_BIN}"
      return 0
    fi
    echo "[WARN] VLLM_HUST_PYTHON_BIN is set but not executable: ${VLLM_HUST_PYTHON_BIN}" >&2
  fi

  if _resolve_hust_ascend_manager_conda_python >/dev/null 2>&1; then
    _resolve_hust_ascend_manager_conda_python
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi

  return 1
}

hust_resolve_python_bin() {
  _resolve_hust_ascend_manager_python "$@"
}

hust_ensure_python_pip() {
  local python_bin="$1"
  local get_pip_script

  if "${python_bin}" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  if "${python_bin}" -m ensurepip --upgrade >/dev/null 2>&1; then
    "${python_bin}" -m pip --version >/dev/null 2>&1
    return $?
  fi

  get_pip_script="$(mktemp "${TMPDIR:-/tmp}/hust-get-pip.XXXXXX.py")" || {
    echo "[ERROR] Could not create a temporary get-pip.py path" >&2
    return 1
  }

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${_HUST_DEFAULT_GET_PIP_URL}" -o "${get_pip_script}" || {
      rm -f "${get_pip_script}"
      echo "[ERROR] Could not download get-pip.py from ${_HUST_DEFAULT_GET_PIP_URL}" >&2
      return 1
    }
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "${get_pip_script}" "${_HUST_DEFAULT_GET_PIP_URL}" || {
      rm -f "${get_pip_script}"
      echo "[ERROR] Could not download get-pip.py from ${_HUST_DEFAULT_GET_PIP_URL}" >&2
      return 1
    }
  else
    rm -f "${get_pip_script}"
    echo "[ERROR] Could not locate curl or wget to download get-pip.py" >&2
    return 1
  fi

  "${python_bin}" "${get_pip_script}" --user >/dev/null 2>&1 || {
    rm -f "${get_pip_script}"
    echo "[ERROR] get-pip.py failed for Python interpreter: ${python_bin}" >&2
    return 1
  }
  rm -f "${get_pip_script}"

  if "${python_bin}" -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  echo "[ERROR] Python interpreter cannot run pip and ensurepip is unavailable: ${python_bin}" >&2
  echo "[ERROR] Install pip for this interpreter or point VLLM_HUST_PYTHON_BIN to a Python with pip." >&2
  return 1
}

_hust_ascend_manager_command_needs_pip() {
  local arg
  for arg in "$@"; do
    case "${arg}" in
      --install-python-stack|--install-plugin)
        return 0
        ;;
    esac
  done
  return 1
}

hust_run_pip() {
  local python_bin
  python_bin="$(_resolve_hust_ascend_manager_python 2>/dev/null)" || python_bin=""

  if [[ -n "${python_bin}" ]]; then
    if ! hust_ensure_python_pip "${python_bin}" >/dev/null 2>&1; then
      python_bin=""
    fi
  fi

  if [[ -n "${python_bin}" ]]; then
    "${python_bin}" -m pip "$@"
    return $?
  fi

  if command -v pip >/dev/null 2>&1; then
    pip "$@"
    return $?
  fi

  if command -v pip3 >/dev/null 2>&1; then
    pip3 "$@"
    return $?
  fi

  python_bin="$(_resolve_hust_ascend_manager_python 2>/dev/null)" || {
    echo "[ERROR] Could not locate python3/python for pip operations" >&2
    return 1
  }

  hust_ensure_python_pip "${python_bin}" || return 1

  "${python_bin}" -m pip "$@"
}

_resolve_hust_ascend_manager_src() {
  local manager_repo="${HUST_ASCEND_MANAGER_REPO:-${_HUST_MANAGER_WORKSPACE_ROOT}/ascend-runtime-manager}"
  local manager_src="${manager_repo}/src"
  if [[ -d "${manager_src}" ]]; then
    printf '%s\n' "${manager_src}"
    return 0
  fi
  return 1
}

hust_apply_default_hf_mirror() {
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    return 0
  fi

  if [[ "${HUST_DISABLE_DEFAULT_HF_MIRROR:-0}" == "1" ]]; then
    return 0
  fi

  export HF_ENDPOINT="${_HUST_DEFAULT_HF_ENDPOINT}"
}

hust_prioritize_conda_runtime_libs() {
  local conda_prefix="${1:-}"
  local conda_lib_dir
  local conda_libstdcpp
  local entry
  local rebuilt_ld_library_path=""
  local -a ld_library_path_entries

  if [[ -z "${conda_prefix}" && -n "${VLLM_HUST_CONDA_PREFIX:-}" ]]; then
    conda_prefix="${VLLM_HUST_CONDA_PREFIX}"
  fi
  if [[ -z "${conda_prefix}" && -n "${PYTHON_BIN:-}" ]]; then
    conda_prefix="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd -P 2>/dev/null || true)"
  fi
  if [[ -z "${conda_prefix}" && -n "${VLLM_HUST_PYTHON_BIN:-}" ]]; then
    conda_prefix="$(cd "$(dirname "${VLLM_HUST_PYTHON_BIN}")/.." && pwd -P 2>/dev/null || true)"
  fi
  if [[ -z "${conda_prefix}" && -n "${CONDA_PREFIX:-}" ]]; then
    conda_prefix="${CONDA_PREFIX}"
  fi

  conda_lib_dir="${conda_prefix:+${conda_prefix}/lib}"
  if [[ -z "${conda_lib_dir}" || ! -d "${conda_lib_dir}" ]]; then
    return 0
  fi

  IFS=':' read -r -a ld_library_path_entries <<< "${LD_LIBRARY_PATH:-}"
  for entry in "${ld_library_path_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == "${conda_lib_dir}" ]]; then
      continue
    fi
    rebuilt_ld_library_path="${rebuilt_ld_library_path:+${rebuilt_ld_library_path}:}${entry}"
  done

  export LD_LIBRARY_PATH="${conda_lib_dir}${rebuilt_ld_library_path:+:${rebuilt_ld_library_path}}"
  conda_libstdcpp="${conda_lib_dir}/libstdc++.so.6"
  if [[ -f "${conda_libstdcpp}" ]]; then
    case ":${LD_PRELOAD:-}:" in
      *:"${conda_libstdcpp}":*)
        ;;
      *)
        export LD_PRELOAD="${conda_libstdcpp}${LD_PRELOAD:+:${LD_PRELOAD}}"
        ;;
    esac
  fi
  echo "[INFO] LD_LIBRARY_PATH prioritized for conda runtime libs: ${conda_lib_dir}"
}

hust_ascend_manager_available() {
  if command -v hust-ascend-manager >/dev/null 2>&1; then
    return 0
  fi

  local manager_src
  local python_bin
  manager_src="$(_resolve_hust_ascend_manager_src 2>/dev/null)" || return 1
  python_bin="$(_resolve_hust_ascend_manager_python 2>/dev/null)" || return 1
  [[ -n "${manager_src}" && -n "${python_bin}" ]]
}

hust_ascend_manager_run() {
  if command -v hust-ascend-manager >/dev/null 2>&1; then
    hust-ascend-manager "$@"
    return $?
  fi

  local manager_src
  local python_bin
  manager_src="$(_resolve_hust_ascend_manager_src 2>/dev/null)" || {
    echo "[ERROR] hust-ascend-manager is required but not found in PATH" >&2
    echo "[ERROR] Could not locate a local ascend-runtime-manager workspace checkout." >&2
    return 1
  }
  python_bin="$(_resolve_hust_ascend_manager_python 2>/dev/null)" || {
    echo "[ERROR] hust-ascend-manager is required but not found in PATH" >&2
    echo "[ERROR] Could not locate a Python interpreter for the local manager fallback." >&2
    return 1
  }

  if [[ "${_HUST_ASCEND_MANAGER_FALLBACK_WARNED:-0}" != "1" ]]; then
    echo "[WARN] hust-ascend-manager not found in PATH; using local workspace fallback at ${manager_src}" >&2
    _HUST_ASCEND_MANAGER_FALLBACK_WARNED=1
  fi

  if _hust_ascend_manager_command_needs_pip "$@"; then
    hust_ensure_python_pip "${python_bin}" || return 1
  fi

  PYTHONPATH="${manager_src}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" -m hust_ascend_manager.cli "$@"
}
