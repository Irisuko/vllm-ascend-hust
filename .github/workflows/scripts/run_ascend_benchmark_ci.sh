#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=${WORKSPACE_ROOT:-${GITHUB_WORKSPACE:-$PWD}}
VLLM_HUST_REPO=${VLLM_HUST_REPO:-$WORKSPACE_ROOT/vllm-hust}
VLLM_ASCEND_HUST_REPO=${VLLM_ASCEND_HUST_REPO:-$WORKSPACE_ROOT}
VLLM_HUST_BENCHMARK_REPO=${VLLM_HUST_BENCHMARK_REPO:-$WORKSPACE_ROOT/vllm-hust-benchmark}
CI_STATE_ROOT=${CI_STATE_ROOT:-$WORKSPACE_ROOT/../vllm-ascend-hust-ci-state}
BENCHMARK_RESULTS_ROOT=${BENCHMARK_RESULTS_ROOT:-$CI_STATE_ROOT/benchmarks/ci}
ASCEND_HUST_TARGET_REPOSITORY=${ASCEND_HUST_TARGET_REPOSITORY:-${GITHUB_REPOSITORY:-unknown}}
ASCEND_HUST_TARGET_REF=${ASCEND_HUST_TARGET_REF:-${GITHUB_REF_NAME:-detached}}
ASCEND_HUST_TARGET_SHA=${ASCEND_HUST_TARGET_SHA:-${GITHUB_SHA:-local}}
ASCEND_HUST_TARGET_COMMIT_URL=${ASCEND_HUST_TARGET_COMMIT_URL:-${GITHUB_SERVER_URL:-https://github.com}/${ASCEND_HUST_TARGET_REPOSITORY}/commit/${ASCEND_HUST_TARGET_SHA}}
ASCEND_HUST_TARGET_SHA_SHORT=$(printf '%s' "$ASCEND_HUST_TARGET_SHA" | cut -c1-8)

RUN_ID=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${ASCEND_HUST_TARGET_SHA_SHORT}}
RESULT_ROOT=${RESULT_ROOT:-$BENCHMARK_RESULTS_ROOT/$RUN_ID}
RAW_RESULT_FILE=${RAW_RESULT_FILE:-$RESULT_ROOT/raw_benchmark.json}
SUBMISSIONS_ROOT=${SUBMISSIONS_ROOT:-$RESULT_ROOT/submissions}
SUBMISSION_DIR=${SUBMISSION_DIR:-$SUBMISSIONS_ROOT/$RUN_ID}
AGGREGATE_OUTPUT_DIR=${AGGREGATE_OUTPUT_DIR:-$RESULT_ROOT/leaderboard-data}
SERVER_LOG=${SERVER_LOG:-$RESULT_ROOT/server.log}
RUNTIME_READY_LOG=${RUNTIME_READY_LOG:-$RESULT_ROOT/runtime-ready.log}
CI_RUNTIME_ROOT=${CI_RUNTIME_ROOT:-$CI_STATE_ROOT/runtime}
PROCESS_MARKER_DIR=${PROCESS_MARKER_DIR:-$CI_RUNTIME_ROOT/process-markers}
SERVER_PID_MARKER=${SERVER_PID_MARKER:-$PROCESS_MARKER_DIR/ascend-benchmark-server.pid}
SERVER_PGID_MARKER=${SERVER_PGID_MARKER:-$PROCESS_MARKER_DIR/ascend-benchmark-server.pgid}
BENCH_SCENARIO=${BENCH_SCENARIO:-random-online}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-}
BENCH_CONSTRAINTS_FILE=${BENCH_CONSTRAINTS_FILE:-}
ALLOW_RANDOM_HF_PUBLISH=${ALLOW_RANDOM_HF_PUBLISH:-0}
SAME_SPEC_BENCHMARK_ENABLED=${SAME_SPEC_BENCHMARK_ENABLED:-1}
SAME_SPEC_SPEC_FILE=${SAME_SPEC_SPEC_FILE:-}
SAME_SPEC_CONSTRAINTS_FILE=${SAME_SPEC_CONSTRAINTS_FILE:-$VLLM_HUST_BENCHMARK_REPO/docs/official-baselines/official-ascend-constraints.stub.json}
SAME_SPEC_PR_PREVIEW_COMPAT=${SAME_SPEC_PR_PREVIEW_COMPAT:-1}

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}
MODEL_PARAMETERS=${MODEL_PARAMETERS:-14B}
MODEL_PRECISION=${MODEL_PRECISION:-FP16}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-}
DTYPE=${DTYPE:-float16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
BENCH_NUM_PROMPTS=${BENCH_NUM_PROMPTS:-200}
BENCH_RANDOM_INPUT_LEN=${BENCH_RANDOM_INPUT_LEN:-1024}
BENCH_RANDOM_OUTPUT_LEN=${BENCH_RANDOM_OUTPUT_LEN:-256}
BENCH_RANDOM_BATCH_SIZE=${BENCH_RANDOM_BATCH_SIZE:-1}
BENCH_REQUEST_RATE=${BENCH_REQUEST_RATE:-1}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-4}
BENCH_INPUT_LEN=${BENCH_INPUT_LEN:-}
BENCH_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-}
HARDWARE_VENDOR=${HARDWARE_VENDOR:-Huawei}
HARDWARE_CHIP_MODEL=${HARDWARE_CHIP_MODEL:-910B2}
CHIP_COUNT=${CHIP_COUNT:-1}
NODE_COUNT=${NODE_COUNT:-1}
PUBLISH_TO_HF=${PUBLISH_TO_HF:-0}
PUBLISH_TO_BENCHMARK_REPO=${PUBLISH_TO_BENCHMARK_REPO:-0}
HF_REPO_ID=${HF_REPO_ID:-}
VLLM_HUST_BENCHMARK_REF=${VLLM_HUST_BENCHMARK_REF:-}

validate_benchmark_inputs() {
  if [[ -n "$VLLM_HUST_BENCHMARK_REF" && "$MODEL_NAME" == "$VLLM_HUST_BENCHMARK_REF" ]]; then
    echo "Invalid benchmark workflow inputs: model_name equals benchmark_ref ($MODEL_NAME)." >&2
    echo "Use model_name for the model repository/path (for example: aly16/Qwen2.5-14B-W8A8)." >&2
    echo "Use benchmark_ref for the vllm-hust-benchmark branch (for example: ws/quantized-model-leaderboard)." >&2
    exit 2
  fi

  if [[ "$MODEL_NAME" == ws/quantized-* || "$MODEL_NAME" == ws/*leaderboard* ]]; then
    echo "Invalid benchmark workflow inputs: model_name looks like a git branch/ref: $MODEL_NAME" >&2
    echo "Expected model_name to be a Hugging Face model id or local model path." >&2
    echo "Did you mean to put '$MODEL_NAME' in benchmark_ref instead?" >&2
    exit 2
  fi
}

validate_benchmark_inputs

# shellcheck source=/dev/null
source "${VLLM_ASCEND_HUST_REPO}/scripts/hust_ascend_manager_helper.sh"

PYTHON_BIN="$(hust_resolve_python_bin 2>/dev/null)" || {
  echo "Could not locate python3/python for benchmark workflow" >&2
  exit 1
}

CI_HOME=${CI_HOME:-$CI_STATE_ROOT/home}
HOME=$CI_HOME
XDG_CACHE_HOME=$CI_HOME/.cache
XDG_CONFIG_HOME=$CI_HOME/.config
export CI_STATE_ROOT BENCHMARK_RESULTS_ROOT CI_HOME HOME XDG_CACHE_HOME XDG_CONFIG_HOME

export PYTHONPATH="${VLLM_HUST_REPO}:${VLLM_HUST_BENCHMARK_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
VLLM_CLI=("${PYTHON_BIN}" -m vllm.entrypoints.cli.main)
VLLM_SERVE=("${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server)
ASCEND_BENCHMARK_ENFORCE_EAGER=${ASCEND_BENCHMARK_ENFORCE_EAGER:-0}
SERVER_READY_TIMEOUT_SECONDS=${SERVER_READY_TIMEOUT_SECONDS:-600}
SERVER_READY_POLL_SECONDS=${SERVER_READY_POLL_SECONDS:-2}
CHAT_SMOKE_TIMEOUT_SECONDS=${CHAT_SMOKE_TIMEOUT_SECONDS:-120}
CHAT_SMOKE_POLL_SECONDS=${CHAT_SMOKE_POLL_SECONDS:-5}
CHAT_SMOKE_REQUEST_TIMEOUT_SECONDS=${CHAT_SMOKE_REQUEST_TIMEOUT_SECONDS:-15}
SERVER_START_RETRIES=${SERVER_START_RETRIES:-8}
SERVER_START_RETRY_DELAY_SECONDS=${SERVER_START_RETRY_DELAY_SECONDS:-10}
ASCEND_RUNTIME_READY_TIMEOUT_SECONDS=${ASCEND_RUNTIME_READY_TIMEOUT_SECONDS:-30}
ASCEND_RUNTIME_READY_POLL_SECONDS=${ASCEND_RUNTIME_READY_POLL_SECONDS:-10}
RESOURCE_BUSY_EXIT_CODE=${RESOURCE_BUSY_EXIT_CODE:-75}
SUDO_AUTH_EXIT_CODE=${SUDO_AUTH_EXIT_CODE:-76}
INVALID_BENCHMARK_RESULT_EXIT_CODE=${INVALID_BENCHMARK_RESULT_EXIT_CODE:-77}
NODE_ENV_FAILURE_EXIT_CODE=${NODE_ENV_FAILURE_EXIT_CODE:-86}
ASCEND_BENCHMARK_USE_SUDO=${ASCEND_BENCHMARK_USE_SUDO:-0}
SAME_SPEC_READY_TIMEOUT_SECONDS=${SAME_SPEC_READY_TIMEOUT_SECONDS:-$SERVER_READY_TIMEOUT_SECONDS}
SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS=${SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS:-300}
CURRENT_VLLM_CACHE_ROOT=${CURRENT_VLLM_CACHE_ROOT:-$CI_RUNTIME_ROOT/current-ascend-same-spec-cache}

normalize_benchmark_spec_path() {
  local spec_file=${1:-}
  if [[ -z "$spec_file" ]]; then
    return 1
  fi
  if [[ "$spec_file" = /* ]]; then
    printf '%s\n' "$spec_file"
  else
    printf '%s\n' "$VLLM_HUST_BENCHMARK_REPO/$spec_file"
  fi
}

resolve_same_spec_spec_file() {
  local resolved_spec_file
  if [[ -n "${SAME_SPEC_SPEC_FILE:-}" ]]; then
    SAME_SPEC_SPEC_FILE=$(normalize_benchmark_spec_path "$SAME_SPEC_SPEC_FILE")
    export SAME_SPEC_SPEC_FILE
    return 0
  fi

  if ! resolved_spec_file=$("$PYTHON_BIN" -m vllm_hust_benchmark.perfgate_specs resolve \
    --scenario "$BENCH_SCENARIO" \
    --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
    --repo-root "$VLLM_HUST_BENCHMARK_REPO"); then
    echo "Could not resolve same-spec benchmark spec for ${BENCH_SCENARIO}/${HARDWARE_CHIP_MODEL}" >&2
    return 2
  fi

  SAME_SPEC_SPEC_FILE=$(normalize_benchmark_spec_path "$resolved_spec_file")
  export SAME_SPEC_SPEC_FILE
  echo "Resolved same-spec benchmark spec file: $SAME_SPEC_SPEC_FILE"
}

DEFAULT_SYSTEM_ASCEND_BENCHMARK_ROOT_HELPER=${DEFAULT_SYSTEM_ASCEND_BENCHMARK_ROOT_HELPER:-/usr/local/bin/run_ascend_benchmark_root_helper.sh}
REPO_ASCEND_BENCHMARK_ROOT_HELPER=$VLLM_ASCEND_HUST_REPO/.github/workflows/scripts/run_ascend_benchmark_root_helper.sh
REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT=$VLLM_ASCEND_HUST_REPO/scripts/install_ascend_benchmark_root_helper.sh
BENCHMARK_DIAGNOSTICS_FILE=${BENCHMARK_DIAGNOSTICS_FILE:-$RESULT_ROOT/benchmark_diagnostics.md}
if [[ -n "${ASCEND_BENCHMARK_ROOT_HELPER:-}" ]]; then
  ASCEND_BENCHMARK_ROOT_HELPER=$ASCEND_BENCHMARK_ROOT_HELPER
elif [[ -x "$DEFAULT_SYSTEM_ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
  ASCEND_BENCHMARK_ROOT_HELPER=$DEFAULT_SYSTEM_ASCEND_BENCHMARK_ROOT_HELPER
else
  ASCEND_BENCHMARK_ROOT_HELPER=$REPO_ASCEND_BENCHMARK_ROOT_HELPER
fi

if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "auto" ]]; then
  if [[ "$(id -u)" == "0" ]]; then
    ASCEND_BENCHMARK_USE_SUDO=0
    echo "Ascend benchmark sudo mode: disabled via auto detection; current user is root"
  elif command -v sudo >/dev/null 2>&1 && [[ -x "$ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
    ASCEND_BENCHMARK_USE_SUDO=1
    echo "Ascend benchmark sudo mode: enabled via auto detection ($ASCEND_BENCHMARK_ROOT_HELPER)"
  else
    ASCEND_BENCHMARK_USE_SUDO=0
    echo "Ascend benchmark sudo mode: disabled via auto detection; sudo or root helper is unavailable"
  fi
fi

server_pid=""
server_group_pid=""
cleanup_ran=0

find_orphaned_engine_pids() {
  "$PYTHON_BIN" - <<'PY'
import os


def read_proc_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as proc_file:
            return proc_file.read()
    except OSError:
        return b""


def read_environ(pid: str) -> dict[str, str]:
    env = {}
    for item in read_proc_bytes(f"/proc/{pid}/environ").split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env[key.decode(errors="ignore")] = value.decode(errors="ignore")
    return env


run_id = os.environ.get("GITHUB_RUN_ID", "")
job_name = os.environ.get("GITHUB_JOB", "")
repository = os.environ.get("GITHUB_REPOSITORY", "")
workspace = os.environ.get("RUNNER_WORKSPACE", "")
current_pid = str(os.getpid())
matches = []

if not run_id or not job_name or not repository:
    print("")
    raise SystemExit(0)

for entry in os.listdir("/proc"):
    if not entry.isdigit() or entry == current_pid:
        continue

    status_text = read_proc_bytes(f"/proc/{entry}/status").decode(errors="ignore")
    status_name = ""
    for line in status_text.splitlines():
        if line.startswith("Name:\t"):
            status_name = line.split("\t", 1)[1].strip()
            break

    if status_name != "VLLM::EngineCor":
        cmdline_text = read_proc_bytes(f"/proc/{entry}/cmdline").replace(b"\0", b" ").decode(errors="ignore")
        if "VLLM::EngineCore" not in cmdline_text:
            continue

    proc_env = read_environ(entry)
    if proc_env.get("GITHUB_RUN_ID") != run_id:
        continue
    if proc_env.get("GITHUB_JOB") != job_name:
        continue
    if proc_env.get("GITHUB_REPOSITORY") != repository:
        continue
    if workspace and proc_env.get("RUNNER_WORKSPACE") != workspace:
        continue

    matches.append(entry)

print(" ".join(matches))
PY
}

kill_matching_pids() {
  local signal="$1"
  shift

  if [[ "$#" -eq 0 ]]; then
    return
  fi

  kill "-$signal" "$@" 2>/dev/null || true
}

SUDO_PRESERVE_ENV_VARS=(
  ASCEND_AICPU_PATH
  ASCEND_BENCHMARK_USE_SUDO
  ASCEND_BENCHMARK_ENFORCE_EAGER
  ASCEND_HOME_PATH
  ASCEND_OPP_PATH
  ASCEND_RT_VISIBLE_DEVICES
  ASCEND_TOOLKIT_HOME
  ASCEND_TOOLKIT_LATEST_HOME
  ASCEND_VISIBLE_DEVICES
  ATB_COMPARE_TILING_EVERY_KERNEL
  ATB_HOME_PATH
  ATB_MATMUL_SHUFFLE_K_ENABLE
  ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT
  ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT
  ATB_SHARE_MEMORY_NAME_SUFFIX
  ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE
  ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE
  ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE
  ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE
  BENCH_CONSTRAINTS_FILE
  BENCH_DATASET_PATH
  BENCH_INPUT_LEN
  BENCH_MAX_CONCURRENCY
  BENCH_NUM_PROMPTS
  BENCH_OUTPUT_LEN
  BENCHMARK_REPO_GH_TOKEN
  BENCHMARK_REPO_SLUG
  BENCHMARK_REPO_SSH_KEY
  BENCH_RANDOM_BATCH_SIZE
  BENCH_RANDOM_INPUT_LEN
  BENCH_RANDOM_OUTPUT_LEN
  BENCH_REQUEST_RATE
  BENCH_SCENARIO
  CHIP_COUNT
  CI_HOME
  CLIENT_READY_CHECK_TIMEOUT_SECONDS
  CONSTRAINTS_FILE
  CMAKE_PREFIX_PATH
  CURRENT_CLIENT_PORT
  CURRENT_DATA_SOURCE
  CURRENT_DTYPE
  CURRENT_GIT_COMMIT
  CURRENT_GITHUB_REF
  CURRENT_GITHUB_REPOSITORY
  CURRENT_HARDWARE_CHIP_MODEL
  CURRENT_MODEL_NAME
  CURRENT_MODEL_PARAMETERS
  CURRENT_MODEL_PATH
  CURRENT_MODEL_PRECISION
  CURRENT_MODEL_QUANTIZATION
  CURRENT_PLUGIN_ENGINE
  CURRENT_PLUGIN_GIT_COMMIT
  CURRENT_PLUGIN_GITHUB_REF
  CURRENT_PLUGIN_GITHUB_REPOSITORY
  CURRENT_RUNTIME_CWD
  CURRENT_RUNTIME_PYTHON
  CURRENT_SERVER_PORT
  CURRENT_SUBMITTER
  CURRENT_VLLM_ASCEND_HUST_REPO
  CURRENT_VLLM_CACHE_ROOT
  CURRENT_VLLM_HUST_REPO
  DTYPE
  GITHUB_ACTOR
  GITHUB_EVENT_NAME
  HCCL_CONNECT_TIMEOUT
  HCCL_EXEC_TIMEOUT
  HF_HOME
  HOME
  HOST
  LD_LIBRARY_PATH
  MAX_MODEL_LEN
  MAX_NUM_SEQS
  MODEL_NAME
  MODEL_PARAMETERS
  MODEL_PRECISION
  MODEL_QUANTIZATION
  NODE_COUNT
  PATH
  PORT
  PUBLISH_TO_BENCHMARK_REPO
  PYTHON_BIN
  PYTHONPATH
  RESULT_DIR
  RESULT_ROOT
  RUN_ID
  SAME_SPEC_CONSTRAINTS_FILE
  SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS
  SAME_SPEC_PR_PREVIEW_COMPAT
  SAME_SPEC_READY_TIMEOUT_SECONDS
  SAME_SPEC_SPEC_FILE
  SERVER_LOG
  TMPDIR
  VLLM_ASCEND_TORCH_PREFLIGHT
  VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE
  VLLM_ASCEND_HUST_REPO
  VLLM_HUST_WORKSPACE_ROOT
  VLLM_HUST_BENCHMARK_REPO
  VLLM_HUST_REPO
  WORKSPACE_ROOT
  XDG_CACHE_HOME
  XDG_CONFIG_HOME
)

build_sudo_env_preserve_list() {
  local preserved=()
  local var_name

  for var_name in "${SUDO_PRESERVE_ENV_VARS[@]}"; do
    if [[ -n "${!var_name+x}" ]]; then
      preserved+=("$var_name")
    fi
  done

  local joined=""
  if [[ "${#preserved[@]}" -gt 0 ]]; then
    joined=$(IFS=,; printf '%s' "${preserved[*]}")
  fi
  printf '%s\n' "$joined"
}

append_benchmark_diagnostic() {
  local line=${1:-}

  [[ -z "$line" ]] && return 0
  printf '%s\n' "$line" >> "$BENCHMARK_DIAGNOSTICS_FILE"
}

benchmark_root_helper_fix_command() {
  printf 'sudo RUNNER_USER=grunner bash %s\n' "$REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT"
}

report_runner_checkout_fix() {
  local missing_path=${1:-}

  [[ -n "$missing_path" ]] && echo "Runner-local benchmark script missing from checkout: $missing_path" >&2
  echo "Runner checkout fix: restore or sync the vllm-ascend-hust checkout on the runner host, then rerun the benchmark workflow." >&2

  if [[ -n "$missing_path" ]]; then
    append_benchmark_diagnostic "- Runner checkout script: missing at \`$missing_path\`"
  fi
  append_benchmark_diagnostic "- Runner checkout fix: \`restore or sync the vllm-ascend-hust checkout on the runner host, then rerun the benchmark workflow\`"
}

report_runner_host_fix() {
  local fix_command

  if [[ -f "$REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT" ]]; then
    fix_command=$(benchmark_root_helper_fix_command)
    echo "Runner host fix: $fix_command" >&2
    append_benchmark_diagnostic "- Runner host fix: \`$fix_command\`"
    return 0
  fi

  echo "Runner-local install script missing from checkout: $REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT" >&2
  append_benchmark_diagnostic "- Runner install script: missing at \`$REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT\`"
  report_runner_checkout_fix "$REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT"
}

export_sudo_preserved_env_vars() {
  local var_name

  for var_name in "${SUDO_PRESERVE_ENV_VARS[@]}"; do
    if [[ -n "${!var_name+x}" ]]; then
      export "$var_name"
    fi
  done
}

run_ascend_root_helper() {
  if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
    local preserve_list
    export_sudo_preserved_env_vars
    preserve_list=$(build_sudo_env_preserve_list)
    if [[ ! -x "$ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
      echo "Ascend benchmark root helper is not executable: $ASCEND_BENCHMARK_ROOT_HELPER" >&2
      append_benchmark_diagnostic "- Benchmark execution: \`benchmark root helper is not executable at $ASCEND_BENCHMARK_ROOT_HELPER\`"
      if [[ "$ASCEND_BENCHMARK_ROOT_HELPER" == "$DEFAULT_SYSTEM_ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
        report_runner_host_fix
      fi
      return 1
    fi
    if [[ -n "$preserve_list" ]]; then
      sudo --preserve-env="$preserve_list" -E -n "$ASCEND_BENCHMARK_ROOT_HELPER" "$@"
    else
      sudo -E -n "$ASCEND_BENCHMARK_ROOT_HELPER" "$@"
    fi
  else
    echo "run_ascend_root_helper requires ASCEND_BENCHMARK_USE_SUDO=1" >&2
    return 2
  fi
}

verify_root_helper_matches_repo() {
  if [[ "$ASCEND_BENCHMARK_USE_SUDO" != "1" ]]; then
    return 0
  fi
  if [[ ! -f "$REPO_ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
    echo "Runner-local benchmark root helper source missing from checkout: $REPO_ASCEND_BENCHMARK_ROOT_HELPER" >&2
    append_benchmark_diagnostic "- Benchmark execution: \`runner-local benchmark root helper source is missing\`"
    report_runner_checkout_fix "$REPO_ASCEND_BENCHMARK_ROOT_HELPER"
    return 2
  fi
  if [[ ! -f "$REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT" ]]; then
    echo "Runner-local benchmark helper install script missing from checkout: $REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT" >&2
    append_benchmark_diagnostic "- Benchmark execution: \`runner-local benchmark helper install script is missing\`"
    report_runner_checkout_fix "$REPO_ASCEND_BENCHMARK_ROOT_HELPER_INSTALL_SCRIPT"
    return 2
  fi
  if [[ "$ASCEND_BENCHMARK_ROOT_HELPER" != "$DEFAULT_SYSTEM_ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
    if [[ ! -x "$ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
      echo "Configured Ascend benchmark root helper is not executable: $ASCEND_BENCHMARK_ROOT_HELPER" >&2
      append_benchmark_diagnostic "- Benchmark execution: \`configured benchmark root helper is not executable at $ASCEND_BENCHMARK_ROOT_HELPER\`"
      return 2
    fi
    return 0
  fi
  if [[ ! -x "$ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
    echo "Installed Ascend benchmark root helper is missing or not executable: $ASCEND_BENCHMARK_ROOT_HELPER" >&2
    append_benchmark_diagnostic "- Benchmark execution: \`installed benchmark root helper is missing or not executable at $ASCEND_BENCHMARK_ROOT_HELPER\`"
    report_runner_host_fix
    return 2
  fi
  if [[ ! -r "$ASCEND_BENCHMARK_ROOT_HELPER" || ! -r "$REPO_ASCEND_BENCHMARK_ROOT_HELPER" ]]; then
    return 0
  fi
  if cmp -s "$ASCEND_BENCHMARK_ROOT_HELPER" "$REPO_ASCEND_BENCHMARK_ROOT_HELPER"; then
    return 0
  fi

  echo "Installed Ascend benchmark root helper is stale: $ASCEND_BENCHMARK_ROOT_HELPER" >&2
  echo "Expected it to match: $REPO_ASCEND_BENCHMARK_ROOT_HELPER" >&2
  append_benchmark_diagnostic "- Benchmark execution: \`installed benchmark root helper is stale at $ASCEND_BENCHMARK_ROOT_HELPER\`"
  append_benchmark_diagnostic "- Expected helper source: \`$REPO_ASCEND_BENCHMARK_ROOT_HELPER\`"
  report_runner_host_fix
  return 2
}

cleanup() {
  if [[ "$cleanup_ran" == "1" ]]; then
    return
  fi
  cleanup_ran=1

  if [[ -n "$server_group_pid" ]]; then
    kill -TERM -- "-$server_group_pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 -- "-$server_group_pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -KILL -- "-$server_group_pid" 2>/dev/null || true
  elif [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
  fi

  if [[ -n "$server_pid" ]]; then
    wait "$server_pid" || true
  fi

  # GitHub Actions cancellation can outlive the vLLM launcher and leave
  # EngineCore workers behind, so sweep matching leftovers by run metadata.
  local orphaned_engine_pids
  orphaned_engine_pids="$(find_orphaned_engine_pids)"
  if [[ -n "$orphaned_engine_pids" ]]; then
    kill_matching_pids TERM $orphaned_engine_pids
    for _ in $(seq 1 10); do
      local remaining_engine_pids=()
      local orphaned_pid
      for orphaned_pid in $orphaned_engine_pids; do
        if kill -0 "$orphaned_pid" 2>/dev/null; then
          remaining_engine_pids+=("$orphaned_pid")
        fi
      done
      if [[ "${#remaining_engine_pids[@]}" -eq 0 ]]; then
        orphaned_engine_pids=""
        break
      fi
      orphaned_engine_pids="${remaining_engine_pids[*]}"
      sleep 1
    done

    if [[ -n "$orphaned_engine_pids" ]]; then
      kill_matching_pids KILL $orphaned_engine_pids
    fi
  fi

  server_pid=""
  server_group_pid=""
  rm -f "$SERVER_PID_MARKER" "$SERVER_PGID_MARKER"
}

same_spec_server_log_indicates_resource_busy() {
  local same_spec_server_log=$RESULT_ROOT/server.stdout.log
  [[ -f "$same_spec_server_log" ]] && grep -qE 'Resource_Busy\(EL0005\)|aclInit, error code is 507899|The resources are busy' "$same_spec_server_log"
}

print_same_spec_server_log_tail() {
  local same_spec_server_log=$RESULT_ROOT/server.stdout.log

  if [[ ! -f "$same_spec_server_log" ]]; then
    echo "same-spec server log not found: $same_spec_server_log" >&2
    return 0
  fi

  echo "---- same-spec server log tail: $same_spec_server_log ----" >&2
  tail -n 120 "$same_spec_server_log" >&2 || true
  echo "---- end same-spec server log tail ----" >&2
}

server_log_indicates_resource_busy() {
  [[ -f "$SERVER_LOG" ]] && grep -qE 'Resource_Busy\(EL0005\)|aclInit, error code is 507899|The resources are busy' "$SERVER_LOG"
}

runtime_ready_log_indicates_resource_busy() {
  [[ -f "$RUNTIME_READY_LOG" ]] && grep -qE 'Resource_Busy\(EL0005\)|aclInit, error code is 507899|The resources are busy' "$RUNTIME_READY_LOG"
}

runtime_ready_log_indicates_sudo_auth_failure() {
  [[ -f "$RUNTIME_READY_LOG" ]] && grep -qE 'sudo: (a password is required|a terminal is required|sorry, you must have a tty|is not allowed to execute|command not found)' "$RUNTIME_READY_LOG"
}

runtime_ready_log_indicates_node_env_failure() {
  [[ -f "$RUNTIME_READY_LOG" ]] && grep -qE "ASCEND_HOME_PATH environment variable is not set|ModuleNotFoundError: No module named 'tbe'|Environment_Error_Import_Python_Module_Failed|Failed to init tbe|GEInitialize failed|OpCompileProcessor init failed" "$RUNTIME_READY_LOG"
}

cleanup_previous_ci_processes() {
  local marker_pgid marker_pid remaining_matches remaining_pids

  if [[ -f "$SERVER_PGID_MARKER" ]]; then
    marker_pgid=$(tr -d '[:space:]' <"$SERVER_PGID_MARKER")
    if [[ -n "$marker_pgid" ]] && kill -0 "$marker_pgid" 2>/dev/null; then
      echo "Cleaning leftover Ascend benchmark process group: $marker_pgid"
      kill -TERM -- "-$marker_pgid" 2>/dev/null || true
      for _ in $(seq 1 10); do
        if ! kill -0 "$marker_pgid" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      kill -KILL -- "-$marker_pgid" 2>/dev/null || true
    fi
  fi

  if [[ -f "$SERVER_PID_MARKER" ]]; then
    marker_pid=$(tr -d '[:space:]' <"$SERVER_PID_MARKER")
    if [[ -n "$marker_pid" ]] && kill -0 "$marker_pid" 2>/dev/null; then
      echo "Cleaning leftover Ascend benchmark process: $marker_pid"
      kill "$marker_pid" 2>/dev/null || true
    fi
  fi

  remaining_matches=$(ps -eo pid,ppid,pgid,sid,etimes,args \
    | grep -F "$WORKSPACE_ROOT" \
    | grep -E 'vllm|python|pytest' \
    | grep -v grep || true)
  if [[ -n "$remaining_matches" ]]; then
    echo "Remaining workspace-scoped vLLM/Python processes before benchmark:"
    echo "$remaining_matches"
    remaining_pids=$(printf '%s\n' "$remaining_matches" | awk '{print $1}')
    if [[ -n "$remaining_pids" ]]; then
      echo "Cleaning workspace-scoped leftover process(es): $remaining_pids"
      # shellcheck disable=SC2086
      kill -TERM $remaining_pids 2>/dev/null || true
      for _ in $(seq 1 10); do
        if ! ps -p "$(printf '%s' "$remaining_pids" | paste -sd, -)" >/dev/null 2>&1; then
          break
        fi
        sleep 1
      done
      # shellcheck disable=SC2086
      kill -KILL $remaining_pids 2>/dev/null || true
    fi
  else
    echo "No leftover workspace-scoped vLLM/Python processes detected before benchmark."
  fi

  rm -f "$SERVER_PID_MARKER" "$SERVER_PGID_MARKER"
}

wait_for_ascend_runtime_ready() {
  local max_attempts
  max_attempts=$(((ASCEND_RUNTIME_READY_TIMEOUT_SECONDS + ASCEND_RUNTIME_READY_POLL_SECONDS - 1) / ASCEND_RUNTIME_READY_POLL_SECONDS))
  if (( max_attempts < 1 )); then
    max_attempts=1
  fi

  for runtime_attempt in $(seq 1 "$max_attempts"); do
    if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
      if run_ascend_root_helper runtime-ready >"$RUNTIME_READY_LOG" 2>&1; then
        return 0
      fi
    elif "${PYTHON_BIN}" - <<'PY' >"$RUNTIME_READY_LOG" 2>&1
import sys

try:
    import torch_npu
    torch_npu.npu.get_soc_version()
except Exception as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(1)
PY
    then
      return 0
    fi

    cat "$RUNTIME_READY_LOG" >&2

    if runtime_ready_log_indicates_sudo_auth_failure; then
      echo "Ascend runtime sudo fallback is not authorized for helper: $ASCEND_BENCHMARK_ROOT_HELPER" >&2
      echo "Grant passwordless sudo for this helper script with SETENV support, or disable ASCEND_BENCHMARK_USE_SUDO." >&2
      return "$SUDO_AUTH_EXIT_CODE"
    fi

    if runtime_ready_log_indicates_node_env_failure; then
      echo "Detected Ascend node environment failure during runtime readiness check." >&2
      echo "CANN/TBE runtime is not available in the benchmark process environment; check set_env.sh, ASCEND_HOME_PATH, and the Python path containing the tbe module." >&2
      return "$NODE_ENV_FAILURE_EXIT_CODE"
    fi

    if [[ "$runtime_attempt" -eq "$max_attempts" ]]; then
      if runtime_ready_log_indicates_resource_busy; then
        return "$RESOURCE_BUSY_EXIT_CODE"
      fi
      return 1
    fi

    echo "Ascend runtime not ready yet; waiting ${ASCEND_RUNTIME_READY_POLL_SECONDS}s before retrying device initialization (${runtime_attempt}/${max_attempts})"
    sleep "$ASCEND_RUNTIME_READY_POLL_SECONDS"
  done
}

resolve_npu_smi_bin() {
  local candidate
  if candidate="$(command -v npu-smi 2>/dev/null)" && [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  for candidate in /usr/local/bin/npu-smi /usr/local/sbin/npu-smi /usr/sbin/npu-smi /usr/bin/npu-smi; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

start_server() {
  local max_model_len_args=()
  local serve_extra_args=()
  if [[ -n "$MAX_MODEL_LEN" ]]; then
    max_model_len_args=(--max-model-len "$MAX_MODEL_LEN")
  fi
  if [[ "$ASCEND_BENCHMARK_ENFORCE_EAGER" == "1" ]]; then
    serve_extra_args+=(--enforce-eager)
  fi

  if command -v setsid >/dev/null 2>&1; then
    if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
      local preserve_list
      export_sudo_preserved_env_vars
      preserve_list=$(build_sudo_env_preserve_list)
      if [[ -n "$preserve_list" ]]; then
        setsid sudo --preserve-env="$preserve_list" -E -n "$ASCEND_BENCHMARK_ROOT_HELPER" serve >"$SERVER_LOG" 2>&1 &
      else
        setsid sudo -E -n "$ASCEND_BENCHMARK_ROOT_HELPER" serve >"$SERVER_LOG" 2>&1 &
      fi
    else
      setsid env VLLM_ASCEND_TORCH_PREFLIGHT=0 "${VLLM_SERVE[@]}" \
        --model "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        "${max_model_len_args[@]}" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        "${serve_extra_args[@]}" >"$SERVER_LOG" 2>&1 &
    fi
    server_pid=$!
    server_group_pid=$server_pid
    printf '%s\n' "$server_pid" >"$SERVER_PID_MARKER"
    printf '%s\n' "$server_group_pid" >"$SERVER_PGID_MARKER"
  else
    if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
      run_ascend_root_helper serve >"$SERVER_LOG" 2>&1 &
    else
      env VLLM_ASCEND_TORCH_PREFLIGHT=0 "${VLLM_SERVE[@]}" \
        --model "$MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        --dtype "$DTYPE" \
        "${max_model_len_args[@]}" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        "${serve_extra_args[@]}" >"$SERVER_LOG" 2>&1 &
    fi
    server_pid=$!
    printf '%s\n' "$server_pid" >"$SERVER_PID_MARKER"
  fi
}

run_completions_smoke() {
  local request_timeout="${1:-$CHAT_SMOKE_REQUEST_TIMEOUT_SECONDS}"

  "$PYTHON_BIN" - "$HOST" "$PORT" "$MODEL_NAME" "$RESULT_ROOT/completions_smoke.json" "$request_timeout" <<'PY'
import json
import sys
import urllib.error
import urllib.request

host, port, model, output_path, request_timeout = sys.argv[1:6]
url = f"http://{host}:{port}/v1/completions"
payload = {
    "model": model,
    "prompt": "Reply with ok.",
    "max_tokens": 2,
    "temperature": 0,
}
request = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=float(request_timeout)) as response:
        body = response.read().decode("utf-8", errors="replace")
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    print(
        f"completions smoke returned HTTP {exc.code}; "
        f"response prefix: {body[:500]}",
        file=sys.stderr,
    )
    raise SystemExit(1)
except Exception as exc:
    print(f"completions smoke request failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

try:
    data = json.loads(body)
except json.JSONDecodeError as exc:
    print(f"completions smoke returned invalid JSON: {exc}", file=sys.stderr)
    raise SystemExit(1)

choices = data.get("choices")
if not isinstance(choices, list) or not choices:
    print("completions smoke returned no choices", file=sys.stderr)
    raise SystemExit(1)

text = choices[0].get("text")

usage = data.get("usage")
completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
has_text = isinstance(text, str) and bool(text.strip())
has_completion_tokens = isinstance(completion_tokens, int) and completion_tokens > 0
if not has_text and not has_completion_tokens:
    print(
        "completions smoke did not observe generated text or completion tokens",
        file=sys.stderr,
    )
    raise SystemExit(1)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "id": data.get("id"),
            "model": data.get("model"),
            "finish_reason": choices[0].get("finish_reason"),
            "text_length": len(str(text)),
            "completion_tokens": completion_tokens,
        },
        f,
        ensure_ascii=True,
        indent=2,
    )

print("completions smoke succeeded")
PY
}

wait_for_completions_smoke() {
  local deadline
  local remaining
  local request_timeout
  local sleep_seconds

  deadline=$(($(date +%s) + CHAT_SMOKE_TIMEOUT_SECONDS))

  while true; do
    remaining=$((deadline - $(date +%s)))
    if (( remaining <= 0 )); then
      break
    fi

    request_timeout="$CHAT_SMOKE_REQUEST_TIMEOUT_SECONDS"
    if (( request_timeout > remaining )); then
      request_timeout="$remaining"
    fi
    if (( request_timeout < 1 )); then
      request_timeout=1
    fi

    if run_completions_smoke "$request_timeout"; then
      return 0
    fi

    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "vLLM server exited while waiting for completions smoke"
      return 1
    fi

    remaining=$((deadline - $(date +%s)))
    if (( remaining <= 0 )); then
      break
    fi
    sleep_seconds="$CHAT_SMOKE_POLL_SECONDS"
    if (( sleep_seconds > remaining )); then
      sleep_seconds="$remaining"
    fi
    sleep "$sleep_seconds"
  done

  echo "Timed out waiting for completions smoke after ${CHAT_SMOKE_TIMEOUT_SECONDS}s"
  return 1
}

run_same_spec_current_benchmark() {
  local same_spec_runner=$VLLM_HUST_BENCHMARK_REPO/scripts/run-current-ascend-same-spec.sh
  local same_spec_raw_result=$RESULT_ROOT/raw_benchmark_result.json
  local same_spec_submission_dir=$RESULT_ROOT/submission
  local effective_same_spec_file=$SAME_SPEC_SPEC_FILE
  local current_vllm_hust_commit
  local same_spec_exit_code=0

  if [[ ! -f "$same_spec_runner" ]]; then
    echo "same-spec benchmark runner not found: $same_spec_runner" >&2
    return 2
  fi
  if [[ ! -f "$SAME_SPEC_SPEC_FILE" ]]; then
    echo "same-spec benchmark spec file not found: $SAME_SPEC_SPEC_FILE" >&2
    return 2
  fi
  if [[ ! -f "$SAME_SPEC_CONSTRAINTS_FILE" ]]; then
    echo "same-spec benchmark constraints file not found: $SAME_SPEC_CONSTRAINTS_FILE" >&2
    return 2
  fi

  current_vllm_hust_commit=$(git -C "$VLLM_HUST_REPO" rev-parse HEAD 2>/dev/null || true)
  rm -f "$same_spec_raw_result" "$RAW_RESULT_FILE"
  rm -rf "$same_spec_submission_dir" "$SUBMISSION_DIR"

  prepare_same_spec_pr_preview_compat_file() {
    local output_file=$RESULT_ROOT/pr-preview-same-spec.compat.json

    "$PYTHON_BIN" - "$SAME_SPEC_SPEC_FILE" "$output_file" <<'PY'
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
payload = json.loads(source.read_text(encoding="utf-8"))

server_parameters = dict(payload.get("server_parameters") or {})
client_parameters = dict(payload.get("client_parameters") or {})

# PR preview runs on self-hosted Ascend runners where the shared same-spec
# defaults can exercise unstable plugin paths before the CI signal is useful.
server_parameters["no_enable_chunked_prefill"] = True
server_parameters["no_enable_prefix_caching"] = True
server_parameters["gpu_memory_utilization"] = 0.6
client_parameters.setdefault("temperature", 0)
client_parameters["max_concurrency"] = 1
client_parameters["request_rate"] = 1
client_parameters["num_prompts"] = int(os.environ.get("BENCH_NUM_PROMPTS", "8"))
client_parameters["input_len"] = int(os.environ.get("BENCH_RANDOM_INPUT_LEN", "64"))
client_parameters["output_len"] = int(os.environ.get("BENCH_RANDOM_OUTPUT_LEN", "16"))

payload["server_parameters"] = server_parameters
payload["client_parameters"] = client_parameters

target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(target)
PY
  }

  if [[ "$SAME_SPEC_PR_PREVIEW_COMPAT" == "1" && \
    ( "${GITHUB_EVENT_NAME:-}" == "pull_request" || \
      "${GITHUB_EVENT_NAME:-}" == "issue_comment" || \
      ( "${GITHUB_EVENT_NAME:-}" == "workflow_dispatch" && "${MODEL_PARAMETERS:-}" == "3B" ) ) ]]; then
    effective_same_spec_file=$(prepare_same_spec_pr_preview_compat_file)
    echo "Using PR preview same-spec compatibility overlay: $effective_same_spec_file"
  fi

  if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "1" ]]; then
    VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
      CURRENT_RUNTIME_CWD=/tmp \
      CURRENT_RUNTIME_PYTHON="$PYTHON_BIN" \
      CURRENT_MODEL_NAME="$MODEL_NAME" \
      CURRENT_MODEL_PATH="$MODEL_NAME" \
      CURRENT_MODEL_PARAMETERS="$MODEL_PARAMETERS" \
      CURRENT_MODEL_PRECISION="$MODEL_PRECISION" \
      CURRENT_MODEL_QUANTIZATION="$MODEL_QUANTIZATION" \
      CURRENT_HARDWARE_CHIP_MODEL="$HARDWARE_CHIP_MODEL" \
      CURRENT_DTYPE="$DTYPE" \
      CURRENT_VLLM_HUST_REPO="$VLLM_HUST_REPO" \
      CURRENT_VLLM_ASCEND_HUST_REPO="$VLLM_ASCEND_HUST_REPO" \
      CURRENT_VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT" \
      CURRENT_GITHUB_REPOSITORY="vLLM-HUST/vllm-hust" \
      CURRENT_GITHUB_REF="$VLLM_HUST_REF" \
      CURRENT_GIT_COMMIT="$current_vllm_hust_commit" \
      CURRENT_PLUGIN_ENGINE="vllm-ascend-hust" \
      CURRENT_PLUGIN_GITHUB_REPOSITORY="$ASCEND_HUST_TARGET_REPOSITORY" \
      CURRENT_PLUGIN_GITHUB_REF="$ASCEND_HUST_TARGET_REF" \
      CURRENT_PLUGIN_GIT_COMMIT="$ASCEND_HUST_TARGET_SHA" \
      CURRENT_SUBMITTER="${GITHUB_ACTOR:-ci}" \
      CURRENT_DATA_SOURCE="vllm-ascend-hust-ci-same-spec" \
      RESULT_DIR="$RESULT_ROOT" \
      RUN_ID="$RUN_ID" \
      CURRENT_SERVER_PORT="$PORT" \
      CURRENT_CLIENT_PORT="$PORT" \
      READY_TIMEOUT_SECONDS="$SAME_SPEC_READY_TIMEOUT_SECONDS" \
      CLIENT_READY_CHECK_TIMEOUT_SECONDS="$SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS" \
      CONSTRAINTS_FILE="$SAME_SPEC_CONSTRAINTS_FILE" \
      run_ascend_root_helper same-spec "$same_spec_runner" "$effective_same_spec_file" || same_spec_exit_code=$?
  else
    env \
      VLLM_HUST_WORKSPACE_ROOT="$WORKSPACE_ROOT" \
      CURRENT_RUNTIME_CWD=/tmp \
      CURRENT_RUNTIME_PYTHON="$PYTHON_BIN" \
      CURRENT_MODEL_NAME="$MODEL_NAME" \
      CURRENT_MODEL_PATH="$MODEL_NAME" \
      CURRENT_MODEL_PARAMETERS="$MODEL_PARAMETERS" \
      CURRENT_MODEL_PRECISION="$MODEL_PRECISION" \
      CURRENT_MODEL_QUANTIZATION="$MODEL_QUANTIZATION" \
      CURRENT_HARDWARE_CHIP_MODEL="$HARDWARE_CHIP_MODEL" \
      CURRENT_DTYPE="$DTYPE" \
      CURRENT_VLLM_HUST_REPO="$VLLM_HUST_REPO" \
      CURRENT_VLLM_ASCEND_HUST_REPO="$VLLM_ASCEND_HUST_REPO" \
      CURRENT_VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT" \
      CURRENT_GITHUB_REPOSITORY="vLLM-HUST/vllm-hust" \
      CURRENT_GITHUB_REF="$VLLM_HUST_REF" \
      CURRENT_GIT_COMMIT="$current_vllm_hust_commit" \
      CURRENT_PLUGIN_ENGINE="vllm-ascend-hust" \
      CURRENT_PLUGIN_GITHUB_REPOSITORY="$ASCEND_HUST_TARGET_REPOSITORY" \
      CURRENT_PLUGIN_GITHUB_REF="$ASCEND_HUST_TARGET_REF" \
      CURRENT_PLUGIN_GIT_COMMIT="$ASCEND_HUST_TARGET_SHA" \
      CURRENT_SUBMITTER="${GITHUB_ACTOR:-ci}" \
      CURRENT_DATA_SOURCE="vllm-ascend-hust-ci-same-spec" \
      RESULT_DIR="$RESULT_ROOT" \
      RUN_ID="$RUN_ID" \
      CURRENT_SERVER_PORT="$PORT" \
      CURRENT_CLIENT_PORT="$PORT" \
      READY_TIMEOUT_SECONDS="$SAME_SPEC_READY_TIMEOUT_SECONDS" \
      CLIENT_READY_CHECK_TIMEOUT_SECONDS="$SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS" \
      CONSTRAINTS_FILE="$SAME_SPEC_CONSTRAINTS_FILE" \
      bash "$same_spec_runner" "$effective_same_spec_file" || same_spec_exit_code=$?
  fi

  if [[ "$same_spec_exit_code" -ne 0 ]]; then
    print_same_spec_server_log_tail
    return "$same_spec_exit_code"
  fi

  if [[ ! -f "$same_spec_raw_result" ]]; then
    echo "same-spec benchmark did not produce raw result: $same_spec_raw_result" >&2
    return 2
  fi
  if [[ ! -f "$same_spec_submission_dir/leaderboard_manifest.json" || ! -f "$same_spec_submission_dir/run_leaderboard.json" ]]; then
    echo "same-spec benchmark did not produce submission artifacts under: $same_spec_submission_dir" >&2
    return 2
  fi
  local validation_status=0
  validate_benchmark_result_file "$same_spec_raw_result" || validation_status=$?
  if [[ "$validation_status" -ne 0 ]]; then
    print_same_spec_server_log_tail
    return "$validation_status"
  fi

  mkdir -p "$SUBMISSION_DIR"
  cp "$same_spec_raw_result" "$RAW_RESULT_FILE"
  cp "$same_spec_submission_dir/leaderboard_manifest.json" "$SUBMISSION_DIR/leaderboard_manifest.json"
  cp "$same_spec_submission_dir/run_leaderboard.json" "$SUBMISSION_DIR/run_leaderboard.json"
}

validate_benchmark_result_file() {
  local result_file=${1:-}

  if [[ -z "$result_file" ]]; then
    echo "validate_benchmark_result_file requires a result file path" >&2
    return 2
  fi
  if [[ ! -f "$result_file" ]]; then
    echo "benchmark result file not found: $result_file" >&2
    return 2
  fi

  RESULT_FILE="$result_file" INVALID_EXIT_CODE="$INVALID_BENCHMARK_RESULT_EXIT_CODE" "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys

result_file = os.environ["RESULT_FILE"]
invalid_exit_code = int(os.environ["INVALID_EXIT_CODE"])

with open(result_file, "r", encoding="utf-8") as handle:
    payload = json.load(handle)

completed = int(payload.get("completed", 0) or 0)
failed = int(payload.get("failed", 0) or 0)
total = int(payload.get("total_input", completed + failed) or (completed + failed))

if completed == 0 and failed > 0:
    print(
        f"invalid-all-failed completed={completed} failed={failed} total_input={total}",
        file=sys.stderr,
    )
    sys.exit(invalid_exit_code)

print(f"validated benchmark result: completed={completed}, failed={failed}, total_input={total}")
PY
}

sync_benchmark_publication_to_github() {
  local publisher_script=${BENCHMARK_PUBLICATION_SYNC_SCRIPT:-$VLLM_ASCEND_HUST_REPO/.github/workflows/scripts/sync_benchmark_snapshots_to_github.sh}

  if [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then
    return 0
  fi

  if [[ ! -x "$publisher_script" ]]; then
    echo "benchmark publication sync script is missing or not executable: $publisher_script" >&2
    return 2
  fi

  BENCHMARK_REPO_DIR="$VLLM_HUST_BENCHMARK_REPO" \
  WEBSITE_REPO_DIR="$VLLM_HUST_WEBSITE_REPO" \
  CURRENT_SUBMISSION_DIR="$SUBMISSION_DIR" \
  VLLM_HUST_REPO_DIR="$VLLM_HUST_REPO" \
  LOCAL_SNAPSHOT_OUTPUT_DIR="$AGGREGATE_OUTPUT_DIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  BENCHMARK_REPO_SLUG="${BENCHMARK_REPO_SLUG:-vLLM-HUST/vllm-hust-benchmark}" \
  BENCHMARK_REPO_GH_TOKEN="${BENCHMARK_REPO_GH_TOKEN:-}" \
  BENCHMARK_REPO_SSH_KEY="${BENCHMARK_REPO_SSH_KEY:-}" \
  SNAPSHOT_COMMIT_MESSAGE="chore(data): sync vllm-ascend benchmark publication $RUN_ID" \
  RUN_ID="$RUN_ID" \
  "$publisher_script"
}

allocate_local_port() {
  "${PYTHON_BIN}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -z "$PORT" ]]; then
  PORT=$(allocate_local_port)
fi

mkdir -p "$RESULT_ROOT" "$SUBMISSIONS_ROOT" "$AGGREGATE_OUTPUT_DIR" "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$PROCESS_MARKER_DIR"
: > "$BENCHMARK_DIAGNOSTICS_FILE"
cleanup_previous_ci_processes

NPU_SMI_BIN="$(resolve_npu_smi_bin 2>/dev/null || true)"
if [[ -n "$NPU_SMI_BIN" ]]; then
  echo "Using npu-smi binary: $NPU_SMI_BIN"
else
  echo "Could not resolve npu-smi binary; single-card device selection may be unavailable"
fi

USER_PROVIDED_ASCEND_VISIBLE_DEVICES=0
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  USER_PROVIDED_ASCEND_VISIBLE_DEVICES=1
fi

select_ascend_device() {
  ASCEND_DEVICE_SELECTION_ATTEMPT="${1:-1}" NPU_SMI_BIN="${2:-}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def parse_logical_map(mapping_output: str) -> dict[tuple[str, str], int]:
  logical_map = {}
  for line in mapping_output.splitlines():
    parts = line.split()
    if len(parts) < 3:
      continue
    npu_id, chip_id, logical_id = parts[:3]
    if npu_id.isdigit() and chip_id.isdigit() and logical_id.isdigit():
      logical_map[(npu_id, chip_id)] = int(logical_id)
  return logical_map


def list_logical_devices(mapping_output: str) -> list[int]:
  logical_devices = set(parse_logical_map(mapping_output).values())
  return sorted(logical_devices)


def list_status_devices(info_output: str) -> list[int]:
  status_devices = set()
  for raw_line in info_output.splitlines():
    line = raw_line.strip()
    if not line.startswith("|"):
      continue

    parts = [part.strip() for part in line.strip("|").split("|")]
    if len(parts) < 2:
      continue

    left_column = parts[0].split()
    if len(left_column) >= 2 and left_column[0].isdigit() and parts[1] and ":" not in parts[1]:
      status_devices.add(int(left_column[0]))

  return sorted(status_devices)


def list_devnode_devices() -> list[int]:
  devnode_devices = set()
  for device_path in Path("/dev").glob("davinci[0-9]*"):
    suffix = device_path.name.removeprefix("davinci")
    if suffix.isdigit():
      devnode_devices.add(int(suffix))
  return sorted(devnode_devices)


def run_npu_smi(*args: str) -> subprocess.CompletedProcess[str] | None:
  npu_smi_bin = os.environ.get("NPU_SMI_BIN")
  if not npu_smi_bin:
    return None

  # Probe npu-smi with a minimal environment so Ascend/Python job variables do
  # not interfere with the management CLI.
  clean_env = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", ""),
    "LANG": os.environ.get("LANG", "C.UTF-8"),
    "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
  }

  try:
    result = subprocess.run(
      [npu_smi_bin, *args],
      check=False,
      capture_output=True,
      text=True,
      timeout=5,
      env=clean_env,
    )
    if result.returncode == 0 or os.geteuid() == 0:
      return result

    if os.environ.get("NPU_SMI_USE_SUDO_FALLBACK", "1") == "0":
      print(f"sudo fallback disabled for npu-smi {' '.join(args)}", file=sys.stderr)
      return result

    sudo_bin = shutil.which("sudo", path=clean_env["PATH"])
    if not sudo_bin:
      print(f"sudo fallback unavailable for npu-smi {' '.join(args)}: sudo not found in PATH", file=sys.stderr)
      return result

    try:
      sudo_result = subprocess.run(
        [sudo_bin, "-n", npu_smi_bin, *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
        env=clean_env,
      )
    except subprocess.TimeoutExpired:
      print(f"sudo npu-smi {' '.join(args)} timed out after 5s", file=sys.stderr)
      return result
    except Exception as exc:
      print(f"sudo npu-smi {' '.join(args)} failed: {exc}", file=sys.stderr)
      return result

    if sudo_result.returncode == 0:
      print(f"sudo fallback succeeded for npu-smi {' '.join(args)}", file=sys.stderr)
      return sudo_result

    print(
      f"sudo fallback for npu-smi {' '.join(args)} returned {sudo_result.returncode}: "
      f"{format_npu_smi_failure(sudo_result)}",
      file=sys.stderr,
    )
    return result
  except subprocess.TimeoutExpired:
    print(f"npu-smi {' '.join(args)} timed out after 5s", file=sys.stderr)
    return None
  except Exception as exc:
    print(f"npu-smi {' '.join(args)} failed: {exc}", file=sys.stderr)
    return None


def select_best_idle_device(info_output: str, logical_map: dict[tuple[str, str], int]) -> tuple[int, str] | None:
  hbm_usage_pattern = re.compile(r"(\d+)\s*/\s*(\d+)\s*$")
  device_stats = []
  current_npu_id = None
  current_health = None

  for raw_line in info_output.splitlines():
    line = raw_line.strip()
    if not line.startswith("|"):
      continue

    parts = [part.strip() for part in line.strip("|").split("|")]
    if len(parts) < 3:
      continue

    left_column = parts[0].split()
    if len(left_column) >= 2 and left_column[0].isdigit() and parts[1] and ":" not in parts[1]:
      current_npu_id = left_column[0]
      current_health = parts[1]
      continue

    if current_npu_id is None or current_health != "OK":
      continue

    if len(left_column) != 1 or not left_column[0].isdigit() or ":" not in parts[1]:
      continue

    chip_id = left_column[0]
    logical_id = logical_map.get((current_npu_id, chip_id))
    device_source = "idle"
    if logical_id is None:
      if chip_id != "0":
        continue
      logical_id = int(current_npu_id)
      device_source = "status-fallback"

    hbm_match = hbm_usage_pattern.search(parts[2])
    if hbm_match is None:
      continue

    used_memory_mb = int(hbm_match.group(1))
    total_memory_mb = int(hbm_match.group(2))
    free_memory_mb = max(0, total_memory_mb - used_memory_mb)
    device_stats.append((logical_id, free_memory_mb, total_memory_mb, device_source))

  if not device_stats:
    return None

  device_stats.sort(key=lambda item: (-item[1], item[0], item[3]))
  selected_device, _, _, selected_source = device_stats[0]
  return selected_device, selected_source

mapping_result = run_npu_smi("info", "-m")
logical_map = {}
logical_devices = []
if mapping_result is not None and mapping_result.returncode == 0:
  logical_map = parse_logical_map(mapping_result.stdout)
  logical_devices = list_status_devices(mapping_result.stdout)
elif mapping_result is not None:
  print(f"npu-smi info -m returned {mapping_result.returncode}: {mapping_result.stderr.strip()}", file=sys.stderr)

selection_attempt = max(1, int(os.environ.get("ASCEND_DEVICE_SELECTION_ATTEMPT", "1")))

info_result = run_npu_smi("info")
if info_result is not None and info_result.returncode == 0:
  selected_device = select_best_idle_device(info_result.stdout, logical_map)
  if selected_device is not None:
    device_id, device_source = selected_device
    print(f"{device_id}\t{device_source}")
    sys.exit(0)
  status_devices = list_status_devices(info_result.stdout)
  if status_devices:
    fallback_device = status_devices[(selection_attempt - 1) % len(status_devices)]
    print(f"{fallback_device}\tstatus-round-robin")
    sys.exit(0)
elif info_result is not None:
  print(f"npu-smi info returned {info_result.returncode}: {info_result.stderr.strip()}", file=sys.stderr)

if logical_devices:
  fallback_device = logical_devices[(selection_attempt - 1) % len(logical_devices)]
  print(f"{fallback_device}\tfallback-round-robin")
  sys.exit(0)

devnode_devices = list_devnode_devices()
if devnode_devices:
  fallback_device = devnode_devices[(selection_attempt - 1) % len(devnode_devices)]
  print(f"{fallback_device}\tdevnode-round-robin")
PY
}

configure_single_card_ascend_device() {
  local start_attempt="${1:-1}"
  local selected_device_info=""

  if [[ "$USER_PROVIDED_ASCEND_VISIBLE_DEVICES" == "1" ]]; then
    export VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE="${VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE:-npu:0}"
    echo "using explicit Ascend visible devices from environment: $ASCEND_RT_VISIBLE_DEVICES"
    echo "skipping automatic single-card Ascend device selection"
    return 0
  fi

  selected_device_info="$(select_ascend_device "$start_attempt" "$NPU_SMI_BIN")"
  if [[ -n "$selected_device_info" ]]; then
    IFS=$'\t' read -r SELECTED_ASCEND_DEVICE SELECTED_ASCEND_DEVICE_SOURCE <<<"$selected_device_info"
    export ASCEND_RT_VISIBLE_DEVICES="$SELECTED_ASCEND_DEVICE"
    export VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE="npu:0"
    echo "selected single-card Ascend device: $ASCEND_RT_VISIBLE_DEVICES (${SELECTED_ASCEND_DEVICE_SOURCE})"
  else
    unset ASCEND_RT_VISIBLE_DEVICES
    unset VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE
    echo "Could not resolve a single-card Ascend device; probing runtime without device scoping"
  fi
}

echo "== Ascend benchmark CI =="
echo "workspace root: $WORKSPACE_ROOT"
echo "vllm-hust repo: $VLLM_HUST_REPO"
echo "vllm-ascend-hust repo: $VLLM_ASCEND_HUST_REPO"
echo "benchmark repo: $VLLM_HUST_BENCHMARK_REPO"
echo "benchmark target repository: $ASCEND_HUST_TARGET_REPOSITORY"
echo "benchmark target ref: $ASCEND_HUST_TARGET_REF"
echo "benchmark target sha: $ASCEND_HUST_TARGET_SHA"
echo "run id: $RUN_ID"
echo "result root: $RESULT_ROOT"
echo "benchmark port: $PORT"
echo "benchmark scenario: $BENCH_SCENARIO"
echo "publish to benchmark repo: $PUBLISH_TO_BENCHMARK_REPO"
echo "publish to hf: $PUBLISH_TO_HF"
echo "same-spec benchmark enabled: $SAME_SPEC_BENCHMARK_ENABLED"
echo "ascend benchmark use sudo: $ASCEND_BENCHMARK_USE_SUDO"

verify_root_helper_matches_repo

if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then
  EFFECTIVE_INPUT_LEN=${BENCH_INPUT_LEN:-}
  EFFECTIVE_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-}
  EFFECTIVE_CONSTRAINTS_FILE=$SAME_SPEC_CONSTRAINTS_FILE
  bench_args=()
else
  case "$BENCH_SCENARIO" in
    random-online)
      EFFECTIVE_INPUT_LEN=${BENCH_INPUT_LEN:-$BENCH_RANDOM_INPUT_LEN}
      EFFECTIVE_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-$BENCH_RANDOM_OUTPUT_LEN}
      EFFECTIVE_CONSTRAINTS_FILE=${BENCH_CONSTRAINTS_FILE:-$VLLM_ASCEND_HUST_REPO/.github/workflows/data/random-online-ci-constraints.json}
      bench_args=(
        --backend vllm
        --endpoint /v1/completions
        --dataset-name random
        --random-input-len "$BENCH_RANDOM_INPUT_LEN"
        --random-output-len "$BENCH_RANDOM_OUTPUT_LEN"
        --random-batch-size "$BENCH_RANDOM_BATCH_SIZE"
        --num-prompts "$BENCH_NUM_PROMPTS"
        --request-rate "$BENCH_REQUEST_RATE"
        --max-concurrency "$BENCH_MAX_CONCURRENCY"
      )
      ;;
    sharegpt-online)
      EFFECTIVE_INPUT_LEN=${BENCH_INPUT_LEN:-1024}
      EFFECTIVE_OUTPUT_LEN=${BENCH_OUTPUT_LEN:-256}
      if [[ -z "$BENCH_DATASET_PATH" ]]; then
        echo "BENCH_DATASET_PATH is required for sharegpt-online" >&2
        exit 2
      fi
      if [[ -z "$BENCH_CONSTRAINTS_FILE" ]]; then
        echo "BENCH_CONSTRAINTS_FILE is required for sharegpt-online" >&2
        exit 2
      fi
      EFFECTIVE_CONSTRAINTS_FILE="$BENCH_CONSTRAINTS_FILE"
      bench_args=(
        --backend vllm
        --endpoint /v1/completions
        --dataset-name sharegpt
        --dataset-path "$BENCH_DATASET_PATH"
        --num-prompts "$BENCH_NUM_PROMPTS"
        --request-rate "$BENCH_REQUEST_RATE"
        --max-concurrency "$BENCH_MAX_CONCURRENCY"
      )
      ;;
    *)
      echo "Unsupported BENCH_SCENARIO without same-spec mode: $BENCH_SCENARIO" >&2
      exit 2
      ;;
  esac
fi

if [[ "$SAME_SPEC_BENCHMARK_ENABLED" != "1" && "$PUBLISH_TO_HF" == "1" && "$BENCH_SCENARIO" == "random-online" && "$ALLOW_RANDOM_HF_PUBLISH" != "1" ]]; then
  echo "Refusing to publish random-online CI preview to HF without ALLOW_RANDOM_HF_PUBLISH=1" >&2
  exit 2
fi

if [[ ! -f "$EFFECTIVE_CONSTRAINTS_FILE" ]]; then
  echo "constraints file not found: $EFFECTIVE_CONSTRAINTS_FILE" >&2
  exit 2
fi

if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then
  resolve_same_spec_spec_file
  for start_attempt in $(seq 1 "$SERVER_START_RETRIES"); do
    if [[ "$CHIP_COUNT" == "1" ]]; then
      configure_single_card_ascend_device "$start_attempt"
    fi

    if wait_for_ascend_runtime_ready; then
      runtime_ready_status=0
    else
      runtime_ready_status=$?
    fi

    if [[ "$runtime_ready_status" -ne 0 ]]; then
      echo "Ascend runtime did not become ready after ${ASCEND_RUNTIME_READY_TIMEOUT_SECONDS}s before same-spec benchmark startup"
      if [[ "$runtime_ready_status" -eq "$SUDO_AUTH_EXIT_CODE" ]]; then
        exit "$runtime_ready_status"
      fi
      if [[ "$runtime_ready_status" -eq "$NODE_ENV_FAILURE_EXIT_CODE" ]]; then
        echo "Detected Ascend node environment failure during same-spec runtime readiness; not retrying across devices."
        exit "$runtime_ready_status"
      fi
      if [[ "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
        echo "Retrying same-spec benchmark after runtime readiness failure in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})"
        sleep "$SERVER_START_RETRY_DELAY_SECONDS"
        continue
      fi
      if [[ "$runtime_ready_status" -eq "$RESOURCE_BUSY_EXIT_CODE" ]]; then
        echo "Detected transient Ascend resource busy state during same-spec runtime readiness after exhausting ${SERVER_START_RETRIES} attempt(s)"
        exit "$RESOURCE_BUSY_EXIT_CODE"
      fi
      exit "$runtime_ready_status"
    fi

    if run_same_spec_current_benchmark; then
      break
    else
      same_spec_exit_code=$?
    fi
    if same_spec_server_log_indicates_resource_busy; then
      if [[ "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
        echo "Detected transient Ascend resource busy state during same-spec benchmark; retrying in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})"
        sleep "$SERVER_START_RETRY_DELAY_SECONDS"
        continue
      fi
      echo "Detected transient Ascend resource busy state after exhausting ${SERVER_START_RETRIES} same-spec benchmark attempt(s)"
      exit "$RESOURCE_BUSY_EXIT_CODE"
    fi
    exit "$same_spec_exit_code"
  done
else
  server_ready_max_attempts=$(((SERVER_READY_TIMEOUT_SECONDS + SERVER_READY_POLL_SECONDS - 1) / SERVER_READY_POLL_SECONDS))
  if (( server_ready_max_attempts < 1 )); then
    server_ready_max_attempts=1
  fi

  server_ready=0

  for start_attempt in $(seq 1 "$SERVER_START_RETRIES"); do
    if [[ "$CHIP_COUNT" == "1" ]]; then
      configure_single_card_ascend_device "$start_attempt"
    fi

    if wait_for_ascend_runtime_ready; then
      runtime_ready_status=0
    else
      runtime_ready_status=$?
    fi

    if [[ "$runtime_ready_status" -ne 0 ]]; then
      echo "Ascend runtime did not become ready after ${ASCEND_RUNTIME_READY_TIMEOUT_SECONDS}s"
      if [[ "$runtime_ready_status" -eq "$SUDO_AUTH_EXIT_CODE" ]]; then
        exit "$runtime_ready_status"
      fi
      if [[ "$runtime_ready_status" -eq "$NODE_ENV_FAILURE_EXIT_CODE" ]]; then
        echo "Detected Ascend node environment failure during runtime readiness; not retrying server startup across devices."
        exit "$runtime_ready_status"
      fi
      if [[ "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
        echo "Retrying server start after runtime readiness failure in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})"
        sleep "$SERVER_START_RETRY_DELAY_SECONDS"
        continue
      fi
      if [[ "$runtime_ready_status" -eq "$RESOURCE_BUSY_EXIT_CODE" ]]; then
        echo "Detected transient Ascend resource busy state during runtime readiness after exhausting ${SERVER_START_RETRIES} start attempt(s)"
        exit "$RESOURCE_BUSY_EXIT_CODE"
      fi
      exit "$runtime_ready_status"
    fi

    start_server

    for attempt in $(seq 1 "$server_ready_max_attempts"); do
      if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null; then
        if wait_for_completions_smoke; then
          server_ready=1
          break 2
        fi
        echo "vLLM server models endpoint is ready, but completions smoke failed"
        cat "$SERVER_LOG"
        if server_log_indicates_resource_busy; then
          if [[ "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
            echo "Detected transient Ascend resource busy state after completions smoke failure; retrying server start in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})"
            cleanup
            sleep "$SERVER_START_RETRY_DELAY_SECONDS"
            break
          fi

          echo "Detected transient Ascend resource busy state after exhausting ${SERVER_START_RETRIES} start attempt(s)"
          exit "$RESOURCE_BUSY_EXIT_CODE"
        fi
        exit 1
      fi

      if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "vLLM server exited before becoming ready"
        cat "$SERVER_LOG"
        if server_log_indicates_resource_busy; then
          if [[ "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
            echo "Detected transient Ascend resource busy state; retrying server start in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})"
            cleanup
            sleep "$SERVER_START_RETRY_DELAY_SECONDS"
            break
          fi

          echo "Detected transient Ascend resource busy state after exhausting ${SERVER_START_RETRIES} start attempt(s)"
          exit "$RESOURCE_BUSY_EXIT_CODE"
        fi
        exit 1
      fi

      if [[ "$attempt" -eq "$server_ready_max_attempts" ]]; then
        echo "Timed out waiting for vLLM server to become ready after ${SERVER_READY_TIMEOUT_SECONDS}s"
        cat "$SERVER_LOG"
        if server_log_indicates_resource_busy; then
          if [[ "$start_attempt" -lt "$SERVER_START_RETRIES" ]]; then
            echo "Detected transient Ascend resource busy state after timeout; retrying server start in ${SERVER_START_RETRY_DELAY_SECONDS}s (attempt ${start_attempt}/${SERVER_START_RETRIES})"
            cleanup
            sleep "$SERVER_START_RETRY_DELAY_SECONDS"
            break
          fi

          echo "Detected transient Ascend resource busy state after exhausting ${SERVER_START_RETRIES} start attempt(s)"
          exit "$RESOURCE_BUSY_EXIT_CODE"
        fi
        exit 1
      fi

      sleep "$SERVER_READY_POLL_SECONDS"
    done
  done

  if [[ "$server_ready" != "1" ]]; then
    echo "vLLM server did not become ready after ${SERVER_START_RETRIES} start attempt(s)"
    cat "$SERVER_LOG"
    if server_log_indicates_resource_busy; then
      echo "Detected transient Ascend resource busy state after exhausting ${SERVER_START_RETRIES} start attempt(s)"
      exit "$RESOURCE_BUSY_EXIT_CODE"
    fi
    exit 1
  fi

  "${VLLM_CLI[@]}" bench serve \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    "${bench_args[@]}" \
    --save-result \
    --result-dir "$RESULT_ROOT" \
    --result-filename "$(basename "$RAW_RESULT_FILE")"

  validate_benchmark_result_file "$RAW_RESULT_FILE"

  CORE_VERSION=$("${PYTHON_BIN}" - <<'PY'
import vllm
print(vllm.__version__)
PY
)

  BACKEND_VERSION=$("${PYTHON_BIN}" - <<'PY'
from vllm_ascend._version import __version__
print(__version__)
PY
)

  ENGINE_VERSION="$ASCEND_HUST_TARGET_SHA_SHORT"

  "${PYTHON_BIN}" -m vllm_hust_benchmark.cli submit \
    "$BENCH_SCENARIO" \
    --benchmark-result-file "$RAW_RESULT_FILE" \
    --constraints-file "$EFFECTIVE_CONSTRAINTS_FILE" \
    --run-id "$RUN_ID" \
    --engine vllm-hust \
    --engine-version "$ENGINE_VERSION" \
    --model-name "$MODEL_NAME" \
    --model-parameters "$MODEL_PARAMETERS" \
    --model-precision "$MODEL_PRECISION" \
    --hardware-vendor "$HARDWARE_VENDOR" \
    --hardware-chip-model "$HARDWARE_CHIP_MODEL" \
    --chip-count "$CHIP_COUNT" \
    --node-count "$NODE_COUNT" \
    --submitter "${GITHUB_ACTOR:-ci}" \
    --data-source "vllm-ascend-hust-ci-$BENCH_SCENARIO" \
    --input-length "$EFFECTIVE_INPUT_LEN" \
    --output-length "$EFFECTIVE_OUTPUT_LEN" \
    --concurrent-requests "$BENCH_MAX_CONCURRENCY" \
    --backend-version "$BACKEND_VERSION" \
    --core-version "$CORE_VERSION" \
    --git-commit "$ASCEND_HUST_TARGET_SHA" \
    --github-commit-url "$ASCEND_HUST_TARGET_COMMIT_URL" \
    --github-repository "$ASCEND_HUST_TARGET_REPOSITORY" \
    --github-ref "$ASCEND_HUST_TARGET_REF" \
    --github-event-name "${GITHUB_EVENT_NAME:-manual}" \
    --submissions-dir "$SUBMISSIONS_ROOT"
fi

if [[ "$PUBLISH_TO_BENCHMARK_REPO" == "1" ]]; then
  sync_benchmark_publication_to_github
fi

if [[ "$PUBLISH_TO_HF" == "1" ]]; then
  if [[ -z "$HF_REPO_ID" ]]; then
    echo "HF_REPO_ID must be set when PUBLISH_TO_HF=1" >&2
    exit 2
  fi

  "${PYTHON_BIN}" -m vllm_hust_benchmark.cli sync-submission-to-hf \
    --submission-dir "$SUBMISSION_DIR" \
    --aggregate-output-dir "$AGGREGATE_OUTPUT_DIR" \
    --repo-id "$HF_REPO_ID" \
    --submissions-prefix submissions-auto \
    --commit-message "chore: sync vllm-hust benchmark from vllm-ascend-hust $RUN_ID (${ASCEND_HUST_TARGET_REPOSITORY}@${ASCEND_HUST_TARGET_REF}:${ASCEND_HUST_TARGET_SHA_SHORT})" \
    --execute
elif [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then
  "${PYTHON_BIN}" -m vllm_hust_benchmark.cli publish-website \
    --source-dir "$SUBMISSIONS_ROOT" \
    --output-dir "$AGGREGATE_OUTPUT_DIR" \
    --execute
fi

echo "RUN_ID=$RUN_ID"
echo "RAW_RESULT_FILE=$RAW_RESULT_FILE"
echo "SUBMISSION_DIR=$SUBMISSION_DIR"
echo "AGGREGATE_OUTPUT_DIR=$AGGREGATE_OUTPUT_DIR"
echo "SERVER_LOG=$SERVER_LOG"
echo "BENCH_SCENARIO=$BENCH_SCENARIO"
echo "EFFECTIVE_CONSTRAINTS_FILE=$EFFECTIVE_CONSTRAINTS_FILE"
