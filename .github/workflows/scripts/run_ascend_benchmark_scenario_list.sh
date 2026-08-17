#!/bin/bash
set -euo pipefail

RUN_ID_BASE=${RUN_ID:-ci-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${ASCEND_HUST_TARGET_SHA_SHORT:-local}}
RESULT_ROOT_BASE=${RESULT_ROOT:-${BENCHMARK_RESULTS_ROOT:-${GITHUB_WORKSPACE:-$PWD}/.benchmarks/ci}/$RUN_ID_BASE}
SCENARIOS_RAW=${BENCH_SCENARIOS:-${BENCH_SCENARIO:-random-online}}
GITHUB_ENV=${GITHUB_ENV:-/dev/null}

trim() {
  local value=$1
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

scenario_slug() {
  printf '%s' "$1" | tr -c '[:alnum:]._-' '-'
}

resolve_schedule_spec() {
  local scenario=$1
  local spec_name
  case "$scenario" in
    random-online) spec_name="official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json" ;;
    sharegpt-online) spec_name="official-ascend-jan-2026-v0180-sharegpt-online-qwen25-14b-910b2.json" ;;
    prefix-repetition-online) spec_name="official-ascend-jan-2026-v0180-prefix-repetition-online-qwen25-14b-910b2.json" ;;
    random-latency) spec_name="official-ascend-jan-2026-v0180-random-latency-qwen25-14b-910b2.json" ;;
    sharegpt-throughput) spec_name="official-ascend-jan-2026-v0180-sharegpt-throughput-qwen25-14b-910b2.json" ;;
    sonnet-throughput) spec_name="official-ascend-jan-2026-v0180-sonnet-throughput-qwen25-14b-910b2.json" ;;
    instructcoder-online) spec_name="official-ascend-jan-2026-v0180-instructcoder-online-qwen25-coder-14b-910b2.json" ;;
    agent-research-online) spec_name="official-ascend-jan-2026-v0180-agent-research-online-qwen25-14b-910b2.json" ;;
    visionarena-online) spec_name="official-ascend-jan-2026-v0180-visionarena-online-qwen25-vl-7b-910b2.json" ;;
    *)
      echo "No official schedule spec is registered for scenario: $scenario" >&2
      return 2
      ;;
  esac

  local spec_path="${VLLM_HUST_BENCHMARK_REPO:?}/docs/official-baselines/$spec_name"
  if [[ ! -f "$spec_path" ]]; then
    echo "Official schedule spec not found: $spec_path" >&2
    return 2
  fi
  printf '%s\n' "$spec_path"
}

write_env() {
  local name=$1
  local value=$2
  local delimiter="EOF_${name}_$$_${RANDOM}"
  {
    echo "${name}<<${delimiter}"
    printf '%s\n' "$value"
    echo "$delimiter"
  } >> "$GITHUB_ENV"
}

scenarios=()
SCENARIOS_RAW=${SCENARIOS_RAW//$'\n'/,}
IFS=',' read -r -a raw_scenarios <<< "$SCENARIOS_RAW"
for raw_scenario in "${raw_scenarios[@]}"; do
  scenario=$(trim "$raw_scenario")
  if [[ -n "$scenario" ]]; then
    scenarios+=("$scenario")
  fi
done

if [[ "${#scenarios[@]}" -le 1 ]]; then
  bash .github/workflows/scripts/run_ascend_benchmark_ci.sh
  exit $?
fi

if [[ -n "${SAME_SPEC_SPEC_FILE:-}" ]]; then
  echo "SAME_SPEC_SPEC_FILE cannot be combined with BENCH_SCENARIOS because each scenario must resolve its own perfgate spec." >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT_BASE"
summary_file="$RESULT_ROOT_BASE/multi_scenario_results.tsv"
printf 'scenario\trun_id\tresult_root\traw_result\tsubmission_dir\texit_code\n' > "$summary_file"

overall_exit_code=0
for scenario in "${scenarios[@]}"; do
  slug=$(scenario_slug "$scenario")
  scenario_run_id="${RUN_ID_BASE}-${slug}"
  scenario_result_root="${RESULT_ROOT_BASE}/${slug}"
  scenario_submission_dir="${scenario_result_root}/submissions/${scenario_run_id}"
  scenario_raw_result="${scenario_result_root}/raw_benchmark.json"

  scenario_spec_file=""
  scenario_model_name="${MODEL_NAME:-}"
  scenario_model_parameters="${MODEL_PARAMETERS:-}"
  scenario_model_precision="${MODEL_PRECISION:-}"
  scenario_dtype="${DTYPE:-}"
  if [[ "${GITHUB_EVENT_NAME:-}" != "pull_request" \
    && "${GITHUB_EVENT_NAME:-}" != "issue_comment" \
    && "${MODEL_PARAMETERS:-}" == "14B" ]]; then
    scenario_spec_file=$(resolve_schedule_spec "$scenario")
    scenario_model_name=$(jq -r '.model // empty' "$scenario_spec_file")
    scenario_model_parameters=$(jq -r '.model_parameters // empty' "$scenario_spec_file")
    scenario_model_precision=$(jq -r '.model_precision // empty' "$scenario_spec_file")
    case "${scenario_model_precision,,}" in
      bf16|bfloat16) scenario_dtype="bfloat16" ;;
      fp16|float16) scenario_dtype="float16" ;;
    esac
    echo "Using official schedule spec: $scenario_spec_file"
    echo "Schedule scenario model: $scenario_model_name ($scenario_model_parameters, $scenario_model_precision)"
  fi

  echo "::group::Ascend benchmark scenario: ${scenario}"
  set +e
  BENCH_SCENARIO="$scenario" \
    BENCH_SCENARIOS="$scenario" \
    BENCH_SCENARIO_COUNT=1 \
    RUN_ID="$scenario_run_id" \
    RESULT_ROOT="$scenario_result_root" \
    RAW_RESULT_FILE="$scenario_raw_result" \
    SUBMISSIONS_ROOT="${scenario_result_root}/submissions" \
    SUBMISSION_DIR="$scenario_submission_dir" \
    AGGREGATE_OUTPUT_DIR="${scenario_result_root}/leaderboard-data" \
    SERVER_LOG="${scenario_result_root}/server.log" \
    RUNTIME_READY_LOG="${scenario_result_root}/runtime-ready.log" \
    BENCHMARK_DIAGNOSTICS_FILE="${scenario_result_root}/benchmark_diagnostics.md" \
    SAME_SPEC_SPEC_FILE="$scenario_spec_file" \
    MODEL_NAME="$scenario_model_name" \
    MODEL_PARAMETERS="$scenario_model_parameters" \
    MODEL_PRECISION="$scenario_model_precision" \
    DTYPE="$scenario_dtype" \
    bash .github/workflows/scripts/run_ascend_benchmark_ci.sh
  scenario_exit_code=$?
  set -e
  echo "::endgroup::"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$scenario" \
    "$scenario_run_id" \
    "$scenario_result_root" \
    "$scenario_raw_result" \
    "$scenario_submission_dir" \
    "$scenario_exit_code" >> "$summary_file"

  if [[ "$scenario_exit_code" -ne 0 && "$overall_exit_code" -eq 0 ]]; then
    overall_exit_code=$scenario_exit_code
  fi
done

write_env BENCHMARK_MULTI_SCENARIO_SUMMARY_FILE "$summary_file"
echo "Multi-scenario benchmark summary: $summary_file"
exit "$overall_exit_code"
