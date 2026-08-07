#!/bin/bash
set -euo pipefail

subcommand=${1:-}
if [[ -z "$subcommand" ]]; then
  echo "Usage: $0 <runtime-ready|same-spec|serve|cleanup-paths|cleanup-processes> [args...]" >&2
  exit 2
fi
shift || true

# Some Ascend env helper scripts assume these variables exist and crash under
# `set -u` when sudo strips them from the environment.
export PATH="${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

restore_user_workspace_ownership() {
  local target_uid=${SUDO_UID:-}
  local target_gid=${SUDO_GID:-}
  local target

  if [[ -z "$target_uid" || -z "$target_gid" ]]; then
    return 0
  fi

  for target in "$@"; do
    [[ -z "$target" ]] && continue
    [[ ! -e "$target" ]] && continue
    chown -R "$target_uid:$target_gid" "$target" 2>/dev/null || true
  done
}

case "$subcommand" in
  cleanup-processes)
    cleanup_script=${1:?cleanup process script path is required}
    shift
    expected_script=${VLLM_ASCEND_HUST_REPO:?VLLM_ASCEND_HUST_REPO must be set}/.github/workflows/scripts/cleanup_ascend_benchmark_processes.py
    if [[ "$cleanup_script" != "$expected_script" || ! -f "$cleanup_script" ]]; then
      echo "Invalid Ascend benchmark cleanup script: $cleanup_script" >&2
      exit 2
    fi
    exec "${PYTHON_BIN:-python3}" "$cleanup_script" "$@"
    ;;
  cleanup-paths)
    if [[ "$#" -eq 0 ]]; then
      echo "cleanup-paths requires at least one path" >&2
      exit 2
    fi

    for target in "$@"; do
      [[ -z "$target" ]] && continue
      rm -rf -- "$target"
    done
    ;;
  runtime-ready)
    "${PYTHON_BIN:?PYTHON_BIN must be set}" - <<'PY'
import sys

try:
    import torch_npu
    torch_npu.npu.get_soc_version()
except Exception as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
PY
    ;;
  same-spec)
    same_spec_runner=${1:?same-spec runner path is required}
    same_spec_spec_file=${2:?same-spec spec file path is required}
    bash "$same_spec_runner" "$same_spec_spec_file"
    restore_user_workspace_ownership \
      "${WORKSPACE_ROOT:-}" \
      "${RESULT_DIR:-}" \
      "${RESULT_ROOT:-}" \
      "${CI_RUNTIME_ROOT:-}" \
      "${CURRENT_VLLM_CACHE_ROOT:-}" \
      "${CI_HOME:-}" \
      "${XDG_CACHE_HOME:-}" \
      "${XDG_CONFIG_HOME:-}"
    ;;
  serve)
    max_model_len_args=()
    serve_extra_args=()
    if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
      max_model_len_args=(--max-model-len "$MAX_MODEL_LEN")
    fi
    if [[ "${ASCEND_BENCHMARK_ENFORCE_EAGER:-0}" == "1" ]]; then
      serve_extra_args+=(--enforce-eager)
    fi

    exec env VLLM_ASCEND_TORCH_PREFLIGHT=0 \
      "${PYTHON_BIN:?PYTHON_BIN must be set}" -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_NAME:?MODEL_NAME must be set}" \
      --host "${HOST:?HOST must be set}" \
      --port "${PORT:?PORT must be set}" \
      --dtype "${DTYPE:?DTYPE must be set}" \
      "${max_model_len_args[@]}" \
      --max-num-seqs "${MAX_NUM_SEQS:?MAX_NUM_SEQS must be set}" \
      "${serve_extra_args[@]}"
    ;;
  *)
    echo "Unsupported subcommand: $subcommand" >&2
    exit 2
    ;;
esac
