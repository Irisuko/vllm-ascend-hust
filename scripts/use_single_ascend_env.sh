#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source scripts/use_single_ascend_env.sh [ASCEND_TOOLKIT_ROOT]
#
# Runtime resolution and exports are centralized in hust-ascend-manager.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/hust_ascend_manager_helper.sh"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[ERROR] This script must be sourced, not executed."
  echo "[ERROR] Use: source scripts/use_single_ascend_env.sh [ASCEND_TOOLKIT_ROOT]"
  exit 1
fi

if ! hust_ascend_manager_available; then
  echo "[ERROR] hust-ascend-manager is required but not found in PATH"
  echo "[ERROR] No local ascend-runtime-manager fallback was found either."
  return 1
fi

ASCEND_ROOT_ARG="${1:-}"
manager_env=""
manager_env_status=0
if [[ -n "${ASCEND_ROOT_ARG}" ]]; then
  manager_env="$(hust_ascend_manager_run env --shell --ascend-root "${ASCEND_ROOT_ARG}")" || manager_env_status=$?
else
  manager_env="$(hust_ascend_manager_run env --shell)" || manager_env_status=$?
fi

if [[ "${manager_env_status}" -eq 0 ]]; then
  eval "${manager_env}"
else
  echo "[WARN] hust-ascend-manager env failed; falling back to local CANN set_env.sh discovery." >&2
fi

if [[ -n "${HUST_ATB_SET_ENV:-}" && -f "${HUST_ATB_SET_ENV}" ]]; then
  set +u
  source "${HUST_ATB_SET_ENV}" --cxx_abi=1
  set -u
fi

cann_tbe_python_bin() {
  local python_bin="${PYTHON_BIN:-}"

  if [[ -n "${python_bin}" && -x "${python_bin}" ]]; then
    printf '%s\n' "${python_bin}"
    return 0
  fi

  hust_resolve_python_bin
}

python_can_import_tbe() {
  local python_bin
  python_bin="$(cann_tbe_python_bin 2>/dev/null)" || return 1

  "${python_bin}" - <<'PY' >/dev/null 2>&1
import tbe  # noqa: F401
PY
}

append_unique_path_var() {
  local var_name="$1"
  local candidate="$2"
  local current_value

  if [[ -z "${candidate}" || ! -d "${candidate}" ]]; then
    return 1
  fi

  current_value="${!var_name:-}"
  case ":${current_value}:" in
    *:"${candidate}":*)
      return 0
      ;;
  esac

  if [[ -n "${current_value}" ]]; then
    printf -v "${var_name}" '%s:%s' "${candidate}" "${current_value}"
  else
    printf -v "${var_name}" '%s' "${candidate}"
  fi
  export "${var_name}"
}

enrich_cann_python_env() {
  local candidate
  local python_bin
  local python_prefix
  local -a python_candidates=(
    "${ASCEND_HOME_PATH:-}/python/site-packages"
    "${ASCEND_TOOLKIT_HOME:-}/python/site-packages"
    "${ASCEND_TOOLKIT_LATEST_HOME:-}/python/site-packages"
    "${ASCEND_OPP_PATH:-}/built-in/op_impl/ai_core/tbe"
    "${ASCEND_OPP_PATH:-}/built-in/op_impl/ai_core/tbe/op_tiling"
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages"
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/te"
    "/usr/local/Ascend/ascend-toolkit/python/site-packages"
  )
  local -a library_candidates=(
    "${ASCEND_HOME_PATH:-}/python/site-packages"
    "${ASCEND_TOOLKIT_HOME:-}/python/site-packages"
    "${ASCEND_TOOLKIT_LATEST_HOME:-}/python/site-packages"
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages"
  )

  for candidate in "${python_candidates[@]}"; do
    append_unique_path_var PYTHONPATH "${candidate}" || true
  done

  for candidate in "${library_candidates[@]}"; do
    append_unique_path_var LD_LIBRARY_PATH "${candidate}" || true
  done

  python_bin="$(cann_tbe_python_bin 2>/dev/null || true)"
  if [[ -n "${python_bin}" ]]; then
    python_prefix="$(cd "$(dirname "${python_bin}")/.." && pwd -P)"
    append_unique_path_var LD_LIBRARY_PATH "${python_prefix}/lib" || true
  fi
}

source_cann_set_env_if_present() {
  local set_env_file="$1"
  local source_status

  if [[ -z "${set_env_file}" || ! -f "${set_env_file}" ]]; then
    return 1
  fi

  echo "[INFO] Sourcing CANN environment: ${set_env_file}"
  source_status=0
  set +u
  # shellcheck source=/dev/null
  source "${set_env_file}" || source_status=$?
  set -u
  return "${source_status}"
}

ensure_cann_tbe_env() {
  local candidate
  local require_cann_tbe="${HUST_REQUIRE_CANN_TBE:-1}"
  local candidates=(
    "${ASCEND_HOME_PATH:-}/set_env.sh"
    "${ASCEND_TOOLKIT_HOME:-}/set_env.sh"
    "${ASCEND_TOOLKIT_LATEST_HOME:-}/set_env.sh"
    "${CONDA_PREFIX:-}/Ascend/cann/set_env.sh"
    /usr/local/Ascend/cann-*/set_env.sh
    "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
    "/usr/local/Ascend/ascend-toolkit/set_env.sh"
  )

  enrich_cann_python_env
  if [[ -n "${ASCEND_HOME_PATH:-}" && -n "${ASCEND_OPP_PATH:-}" ]] && python_can_import_tbe; then
    export HUST_ASCEND_TBE_AVAILABLE=1
    return 0
  fi

  for candidate in "${candidates[@]}"; do
    if source_cann_set_env_if_present "${candidate}"; then
      enrich_cann_python_env
      if [[ -n "${ASCEND_HOME_PATH:-}" && -n "${ASCEND_OPP_PATH:-}" ]] && python_can_import_tbe; then
        export HUST_ASCEND_TBE_AVAILABLE=1
        return 0
      fi
    fi
  done

  export HUST_ASCEND_TBE_AVAILABLE=0
  if [[ "${require_cann_tbe}" != "1" ]]; then
    echo "[WARN] CANN TBE Python module is unavailable to the current Python, but HUST_REQUIRE_CANN_TBE=${require_cann_tbe}; continuing without strict TBE enforcement." >&2
    echo "[WARN] ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}" >&2
    echo "[WARN] ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-<unset>}" >&2
    echo "[WARN] PYTHON_BIN=$(cann_tbe_python_bin 2>/dev/null || printf '<unresolved>')" >&2
    echo "[WARN] PYTHONPATH=${PYTHONPATH:-<unset>}" >&2
    return 0
  fi

  echo "[ERROR] CANN TBE Python module is unavailable to the benchmark Python." >&2
  echo "[ERROR] ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}" >&2
  echo "[ERROR] ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-<unset>}" >&2
  echo "[ERROR] PYTHON_BIN=$(cann_tbe_python_bin 2>/dev/null || printf '<unresolved>')" >&2
  echo "[ERROR] PYTHONPATH=${PYTHONPATH:-<unset>}" >&2
  echo "[ERROR] Source the correct CANN set_env.sh or install the CANN TBE component before running Ascend benchmarks." >&2
  return 1
}

ensure_cann_tbe_env || return 1

# CANN setup may prepend its system runtime paths after the Python environment
# was discovered. Move the selected Python/Conda runtime back to the front so
# extension modules do not load an older system libstdc++ ABI.
hust_prioritize_conda_runtime_libs

normalize_visible_devices() {
  local raw_value="${1:-}"
  local device
  local -a devices=()

  IFS=',' read -r -a raw_devices <<< "${raw_value}"
  for device in "${raw_devices[@]}"; do
    device="${device//[[:space:]]/}"
    if [[ -n "${device}" ]]; then
      devices+=("${device}")
    fi
  done

  if [[ "${#devices[@]}" -eq 0 ]]; then
    return 1
  fi

  local normalized_devices
  normalized_devices="$(IFS=','; echo "${devices[*]}")"
  printf '%s\n' "${normalized_devices}"
}

resolved_visible_devices="$(normalize_visible_devices "${ASCEND_VISIBLE_DEVICES:-}" 2>/dev/null || true)"
resolved_rt_visible_devices="$(normalize_visible_devices "${ASCEND_RT_VISIBLE_DEVICES:-}" 2>/dev/null || true)"

if [[ -z "${resolved_rt_visible_devices}" && -n "${resolved_visible_devices}" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${resolved_visible_devices}"
  echo "[INFO] Derived ASCEND_RT_VISIBLE_DEVICES from ASCEND_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES}"
elif [[ -n "${resolved_rt_visible_devices}" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${resolved_rt_visible_devices}"
elif [[ -n "${ASCEND_RT_VISIBLE_DEVICES+x}" ]]; then
  unset ASCEND_RT_VISIBLE_DEVICES
  echo "[WARN] Ignoring empty ASCEND_RT_VISIBLE_DEVICES from parent environment"
fi

if [[ "${HUST_ASCEND_HAS_STREAM_ATTR:-0}" != "1" ]]; then
  echo "[WARN] Current Ascend runtime does not export aclrtSetStreamAttribute"
  echo "[WARN] npugraph_ex requires a newer CANN runtime. vllm-ascend currently recommends CANN 8.5.1."
fi

if [[ -z "${HUST_ASCEND_RUNTIME_VERSION:-}" ]]; then
  echo "[WARN] Could not detect the exact CANN runtime version from hust-ascend-manager output."
  echo "[WARN] Current ASCEND_HOME_PATH resolves to ${ASCEND_HOME_PATH:-<unset>}."
fi

if [[ "${HUST_REQUIRE_NPUGRAPH:-0}" == "1" && "${HUST_ASCEND_HAS_STREAM_ATTR:-0}" != "1" ]]; then
  echo "[ERROR] HUST_REQUIRE_NPUGRAPH=1 but current runtime cannot support npugraph_ex"
  return 1
fi

if [[ -n "${VLLM_ASCEND_HUST_REPO:-}" && -d "${VLLM_ASCEND_HUST_REPO}" ]]; then
  expected_repo="$(cd "${VLLM_ASCEND_HUST_REPO}" && pwd -P)"
  sanitized_pythonpath=""

  IFS=':' read -r -a pythonpath_entries <<< "${PYTHONPATH:-}"
  for entry in "${pythonpath_entries[@]}"; do
    if [[ -z "${entry}" ]]; then
      continue
    fi

    resolved_entry="$entry"
    if [[ -d "${entry}" ]]; then
      resolved_entry="$(cd "${entry}" && pwd -P)"
    fi

    if [[ "${resolved_entry}" != "${expected_repo}" && (
      "${resolved_entry}" == */vllm-ascend-hust ||
      -d "${resolved_entry}/vllm_ascend"
    ) ]]; then
      continue
    fi

    if [[ -n "${sanitized_pythonpath}" ]]; then
      sanitized_pythonpath+=":${resolved_entry}"
    else
      sanitized_pythonpath="${resolved_entry}"
    fi
  done

  if [[ -n "${sanitized_pythonpath}" ]]; then
    export PYTHONPATH="${expected_repo}:${sanitized_pythonpath}"
  else
    export PYTHONPATH="${expected_repo}"
  fi

  echo "[INFO] PYTHONPATH prioritized for vllm-ascend-hust: ${expected_repo}"
fi

echo "[OK] Single Ascend runtime is configured"
echo "  ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
echo "  CANN_VERSION=${HUST_ASCEND_RUNTIME_VERSION:-<unknown>}"
echo "  HAS_aclrtSetStreamAttribute=${HUST_ASCEND_HAS_STREAM_ATTR:-0}"
