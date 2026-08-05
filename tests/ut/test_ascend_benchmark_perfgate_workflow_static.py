#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/ascend-benchmark-leaderboard.yml"
SCRIPT_DIR = REPO_ROOT / ".github/workflows/scripts"
MANAGER_HELPER = REPO_ROOT / "scripts/hust_ascend_manager_helper.sh"
INSTALL_PLUGIN_SCRIPT = REPO_ROOT / "scripts/install_local_ascend_plugin.sh"
INSTALL_DEV_HUB_SCRIPT = SCRIPT_DIR / "install_ascend_benchmark_with_dev_hub.sh"
USE_SINGLE_ASCEND_ENV_SCRIPT = REPO_ROOT / "scripts/use_single_ascend_env.sh"
PERFGATE_VALIDATE_REQUIRED_SCRIPT = SCRIPT_DIR / "perfgate_validate_required.sh"
PROCESS_CLEANUP_SCRIPT = SCRIPT_DIR / "cleanup_ascend_benchmark_processes.sh"


def test_perfgate_scripts_are_present() -> None:
    for script_name in (
        "perfgate_fetch_baseline.sh",
        "perfgate_stage1_compare.sh",
        "perfgate_stage2_rebase_and_benchmark.sh",
        "perfgate_compare.sh",
        "perfgate_store_baseline.sh",
        "perfgate_validate_required.sh",
        "install_ascend_benchmark_with_dev_hub.sh",
        "parse_ascend_comment_command.py",
        "resolve_ascend_benchmark_scenario.py",
        "resolve_perfgate_spec_file.py",
        "cleanup_ascend_benchmark_processes.py",
        "cleanup_ascend_benchmark_processes.sh",
        "capture_ascend_benchmark_diagnostics.sh",
        "prepare_plugin_perfgate_artifact.py",
    ):
        assert (SCRIPT_DIR / script_name).is_file()


def test_engine_core_cleanup_is_scoped_and_runs_before_hardware_unlock() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")
    root_helper = (SCRIPT_DIR / "run_ascend_benchmark_root_helper.sh").read_text(encoding="utf-8")
    cleanup_wrapper = PROCESS_CLEANUP_SCRIPT.read_text(encoding="utf-8")

    acquire_index = workflow.index("- name: Acquire Ascend hardware lock")
    stale_index = workflow.index("- name: Cleanup stale Ascend benchmark processes")
    current_index = workflow.index("- name: Cleanup current Ascend benchmark processes")
    release_index = workflow.index("- name: Release Ascend hardware lock")
    assert acquire_index < stale_index < current_index < release_index

    current_step = workflow[current_index:release_index]
    assert "if: always()" in current_step
    assert "cleanup_ascend_benchmark_processes.sh current" in current_step
    assert "ASCEND_BENCHMARK_CLEANUP_MARKER_FILE:" in current_step
    assert "runtime/process-markers/ascend-benchmark-server.pid" in current_step
    assert "cleanup_ascend_benchmark_processes.sh stale" in workflow[stale_index:current_index]

    preserve_block = runner_script[runner_script.index("SUDO_PRESERVE_ENV_VARS=(") :]
    for variable in (
        "GITHUB_REPOSITORY",
        "GITHUB_WORKFLOW",
        "GITHUB_JOB",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "RUNNER_NAME",
        "RUNNER_WORKSPACE",
    ):
        assert variable in preserve_block
        assert variable in cleanup_wrapper

    assert "cleanup-processes)" in root_helper
    assert "cleanup_ascend_benchmark_processes.py" in root_helper
    assert 'elif [[ "$mode" == "stale" ]]' in cleanup_wrapper
    assert "ascend-benchmark-server.pid" in cleanup_wrapper
    assert "pkill" not in cleanup_wrapper
    assert "grep -E 'vllm|python|pytest'" not in runner_script
    assert 'ASCEND_BENCHMARK_CLEANUP_MARKER_FILE="$SERVER_PID_MARKER"' in runner_script
    assert "record_server_marker_members()" in runner_script
    assert '--refresh-marker-members "$SERVER_PID_MARKER"' in runner_script
    assert "record_server_marker\n  # Start the verified-member journal immediately" in runner_script
    assert (
        'for attempt in $(seq 1 "$server_ready_max_attempts"); do\n'
        "      # Keep an append-only snapshot while the launcher identity and ancestry\n"
        "      # are still available. Cleanup revalidates every PID by start time.\n"
        "      record_server_marker_members"
    ) in runner_script
    assert '  record_server_marker_members\n\n  case "$ASCEND_BENCHMARK_CLEANUP_VALIDATION_MODE"' in runner_script
    assert 'kill -TERM -- "-$server_group_pid"' not in runner_script
    assert "if ! command -v setsid >/dev/null 2>&1; then" in runner_script
    assert "setsid is required to launch the benchmark with an isolated process group" in runner_script
    assert "VLLM_ASCEND_HUST_BENCHMARK_RUNNER_LABEL" in workflow
    assert "linux-aarch64-a2b3-npu0" in workflow
    assert "vllm-ascend-0-21-0rc1" in workflow
    assert "      - ascend\n      - 910b\n      - docker" in workflow
    assert 'ASCEND_RT_VISIBLE_DEVICES: "0"' in workflow
    assert "cleanup-ascend-benchmark:" in workflow
    assert "needs: ascend-benchmark" in workflow
    assert "runner_name: ${{ steps.runner-identity.outputs.runner_name }}" in workflow
    assert "Capture benchmark runner identity" in workflow
    assert "Verify cleanup runner identity" in workflow
    assert "Cleanup runner mismatch: expected" in workflow
    assert "ASCEND_BENCHMARK_CLEANUP_EXPECTED_RUNNER_NAME:" in workflow
    assert "cleanup_validation_mode:" in workflow
    assert "failure-after-server-ready" in workflow
    assert "wait-for-manual-cancel" in workflow
    assert "wait-for-timeout" in workflow
    assert "cleanup_validation_timeout_minutes:" in workflow
    assert "inputs.cleanup_validation_mode != 'normal'" in workflow
    assert "format('{0}@{1}', github.repository, github.ref_name)" in workflow
    assert "always() && !cancelled() &&" in workflow
    assert "Capture idle Ascend NPU diagnostics" in workflow
    assert workflow.count("Capture post-cleanup Ascend NPU diagnostics") == 2
    assert "if: ${{ always() && needs.ascend-benchmark.result != 'skipped' }}" in workflow
    assert "ASCEND_BENCHMARK_CLEANUP_TARGET_JOB: ascend-benchmark" in workflow
    assert "ASCEND_BENCHMARK_CLEANUP_TARGET_RUN_ID: ${{ github.run_id }}" in workflow
    assert "ASCEND_BENCHMARK_CLEANUP_TARGET_RUN_ATTEMPT: ${{ github.run_attempt }}" in workflow
    assert "ASCEND_BENCHMARK_CLEANUP_MARKER_FILE:" in workflow
    cleanup_job = workflow[workflow.index("  cleanup-ascend-benchmark:") :]
    assert "BENCHMARK_CHECKOUT_USE_SSH_443:" in cleanup_job
    assert "Configure GitHub SSH over 443" in cleanup_job
    assert "ssh-key: ${{ secrets.VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY }}" in cleanup_job
    assert "ssh-strict: ${{ env.BENCHMARK_CHECKOUT_USE_SSH_443 != '1' }}" in cleanup_job
    assert "timeout-minutes: 10" in workflow

    diagnostics = (SCRIPT_DIR / "capture_ascend_benchmark_diagnostics.sh").read_text(encoding="utf-8")
    assert '"$npu_smi_bin" info' in diagnostics
    assert "HBM diagnostics could not be captured" in diagnostics

    assert "validate_cleanup_validation_mode()" in runner_script
    assert "Ascend cleanup validation modes are restricted to workflow_dispatch" in runner_script
    assert "Ascend cleanup validation modes cannot publish benchmark results" in runner_script
    assert "CLEANUP_VALIDATION_READY mode=" in runner_script
    assert "failure-after-server-ready" in runner_script
    assert "wait-for-manual-cancel|wait-for-timeout" in runner_script
    assert 'capture_ascend_benchmark_diagnostics.sh" job-exit' in runner_script


def test_cleanup_wrapper_fails_closed_on_runner_mismatch() -> None:
    cleanup_wrapper = (SCRIPT_DIR / "cleanup_ascend_benchmark_processes.sh").read_text(encoding="utf-8")

    assert "ASCEND_BENCHMARK_CLEANUP_EXPECTED_RUNNER_NAME" in cleanup_wrapper
    assert '"$RUNNER_NAME" != "$expected_runner_name"' in cleanup_wrapper
    assert "Cleanup runner mismatch: expected" in cleanup_wrapper


def test_ascend_benchmark_workflow_wires_two_stage_perfgate() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "PERFGATE_MODE" in workflow
    assert "PERFGATE_SPEC_FILE" in workflow
    assert "SOC_VERSION: ascend910b2" in workflow
    assert (
        "runs-on:\n"
        "      - self-hosted\n"
        "      - Linux\n"
        "      - ARM64\n"
        "      - ascend\n"
        "      - 910b\n"
        "      - docker\n"
        "      - vllm-ascend-0-21-0rc1\n"
        "      - linux-aarch64-a2b3-pool\n"
        "      - ascend-benchmark\n"
        "      - ${{ vars.VLLM_ASCEND_HUST_BENCHMARK_RUNNER_LABEL || 'linux-aarch64-a2b3-npu0' }}"
    ) in workflow
    assert (
        "HARDWARE_CHIP_MODEL: ${{ github.event_name == 'workflow_dispatch' && inputs.hardware_chip_model || '910B2' }}"
        in workflow
    )
    assert "Resolve perfgate same-spec file" in workflow
    assert "Resolve main same-spec file" in workflow
    main_spec_start = workflow.index("- name: Resolve main same-spec file")
    main_spec_end = workflow.index("- name: Checkout ascend-runtime-manager repo")
    main_spec_block = workflow[main_spec_start:main_spec_end]
    assert "github.event_name != 'push'" not in main_spec_block
    assert "resolve_perfgate_spec_file.py" in workflow
    assert "MAIN_SAME_SPEC_SPEC_FILE:" in workflow
    assert "github.event_name != 'pull_request' && github.event_name != 'issue_comment'" in workflow
    assert '--explicit-chip-model "${HARDWARE_CHIP_MODEL}"' in workflow
    assert '--benchmark-repo "${VLLM_HUST_BENCHMARK_REPO}"' in workflow
    assert '--explicit-same-spec-file ""' in workflow
    assert 'spec_file="${SAME_SPEC_SPEC_FILE:-$MAIN_SAME_SPEC_SPEC_FILE}"' in workflow
    assert "MAIN_BENCH_SCENARIO" in workflow
    assert '--scenario "${MAIN_BENCH_SCENARIO}"' in workflow
    assert '--repo-root "${VLLM_HUST_BENCHMARK_REPO}"' in workflow
    assert "docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json" not in workflow
    assert "docs/official-baselines/perfgate-ascend-qwen25-3b-910b3.json" not in workflow
    assert "perfgate-ascend-qwen25-3b-910b3.json" not in workflow
    assert "VLLM_HUST_BENCHMARK_REF" in workflow
    assert "model_parameters:" in workflow
    assert "inputs.model_parameters" in workflow
    assert "ref: ${{ env.VLLM_HUST_BENCHMARK_REF }}" in workflow
    assert 'hust_run_pip install -e "${VLLM_HUST_BENCHMARK_REPO}[publish]"' not in workflow
    assert "Detect PR fork point" in workflow
    assert "Performance gate - fetch Stage 1 baseline" in workflow
    assert "Performance gate - Stage 1 comparison" in workflow
    assert "Performance gate - Stage 2 trial rebase and benchmark" in workflow
    assert "Performance gate - two-stage comparison" in workflow
    assert "store-main-perfgate-baseline:" not in workflow
    assert "Publish central Plugin perfgate baseline" in workflow
    assert "perfgate_report.md" in workflow
    assert "issue_comment:" in workflow
    assert "Parse Ascend comment command" in workflow
    assert "resolve_ascend_benchmark_scenario.py" in workflow
    assert "github.event_name == 'issue_comment'" in workflow
    assert "benchmark_scenarios:" in workflow
    assert "BENCH_SCENARIOS:" in workflow
    assert "inputs.benchmark_scenarios" in workflow
    assert "vars.VLLM_ASCEND_HUST_PR_BENCHMARK_SCENARIOS" in workflow
    assert "vars.VLLM_ASCEND_HUST_MAIN_BENCHMARK_SCENARIOS" in workflow
    assert "run_ascend_benchmark_scenario_list.sh" in workflow
    assert "steps.resolve-scenario.outputs.BENCH_SCENARIO_COUNT == '1'" in workflow
    assert (
        "(github.event_name == 'pull_request' || github.event_name == 'issue_comment' || "
        "(github.event_name == 'workflow_dispatch' && "
        "inputs.benchmark_scenarios == 'perfgate-bootstrap')) "
        "&& steps.resolve-scenario.outputs.BENCH_SCENARIO_COUNT == '1'"
    ) in workflow
    assert (
        "github.event_name != 'pull_request' && github.event_name != 'issue_comment' "
        "&& !(github.event_name == 'workflow_dispatch' && "
        "inputs.benchmark_scenarios == 'perfgate-bootstrap') "
        "&& steps.resolve-scenario.outputs.BENCH_SCENARIO_COUNT == '1'"
    ) in workflow
    assert "multi_scenario_results.tsv" in workflow
    assert "Perfgate comparison: `skipped for multi-scenario run" in workflow
    assert "os.environ.get('BENCH_SCENARIO_COUNT', '1') == '1'" in workflow
    assert (
        "timeout-minutes: ${{ fromJSON(github.event_name == 'push' && "
        "github.ref == 'refs/heads/main' && '150' || "
        "(github.event_name == 'workflow_dispatch' && "
        "inputs.cleanup_validation_timeout_minutes || '60')) }}"
    ) in workflow
    assert "VLLM_ASCEND_HUST_PUBLISH_BENCHMARK_ON_PR" not in workflow
    assert "github.event_name == 'pull_request' || github.event_name == 'issue_comment'" in workflow
    assert "Checkout dev-hub repo" not in workflow
    assert "VLLM_HUST_DEV_HUB_REF" not in workflow
    assert "HUST_ASCEND_MANAGER_REF" in workflow
    assert "ref: ${{ env.HUST_ASCEND_MANAGER_REF }}" in workflow
    assert "install_ascend_benchmark_with_dev_hub.sh" in workflow
    assert 'hust_run_pip install "torch==2.9.0"' not in workflow
    assert "scripts/install_local_ascend_plugin.sh" not in workflow
    assert "resolve_cann_major_version()" not in workflow
    assert "vars.VLLM_ASCEND_HUST_BENCHMARK_USE_SUDO || 'auto'" in workflow
    assert "CURRENT_VLLM_CACHE_ROOT: ${{ github.workspace }}/../.hf-cache/vllm" in workflow
    assert "VLLM_ASCEND_HUST_SAME_SPEC_READY_TIMEOUT_SECONDS || '1800'" in workflow
    assert "VLLM_ASCEND_HUST_SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS || '300'" in workflow
    assert "vars.VLLM_ASCEND_HUST_COMPILE_CUSTOM_KERNELS || 'auto'" in workflow
    assert "VLLM_ASCEND_HUST_STAGE2_DEV_HUB_QUICKSTART_CONDA" not in workflow
    assert "github.event_name == 'pull_request' && 'enforce'" in workflow
    assert "Validate required PR perfgate scenario" in workflow
    assert "Validate required performance gate completion" in workflow
    assert 'PERFGATE_REQUIRED: "1"' in workflow
    assert "PERFGATE_BASELINE_UNAVAILABLE_REASON" in workflow
    assert "PERFGATE_STAGE2_NOT_RUN_REASON" in workflow
    assert "always() && (github.event_name == 'pull_request' || github.event_name == 'issue_comment')" in workflow
    assert "PERFGATE_STAGE2_REBASE_CONFLICT_FILE" in workflow


def test_required_perfgate_scripts_fail_fast() -> None:
    stage1_script = (SCRIPT_DIR / "perfgate_stage1_compare.sh").read_text(encoding="utf-8")
    stage2_script = (SCRIPT_DIR / "perfgate_stage2_rebase_and_benchmark.sh").read_text(encoding="utf-8")

    assert "write_env PERFGATE_STAGE1_COMPLETED 1" in stage1_script
    assert '"$MODE" == "enforce"' in stage1_script
    assert '"$MODE" != "enforce"' in stage2_script
    assert "write_env PERFGATE_STAGE2_EXECUTED 1" in stage2_script
    assert 'write_env PERFGATE_STAGE2_BASELINE_AVAILABLE "$stage2_baseline_available"' in stage2_script
    assert stage2_script.count('if [[ "$MODE" == "enforce" ]]') >= 2


def test_stage1_comparison_fails_only_in_enforce_mode(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
report_file=""
while (( $# > 0 )); do
  if [[ "$1" == "--report-file" ]]; then
    report_file=$2
    break
  fi
  shift
done
printf '**Overall: FAIL**\n' > "$report_file"
exit 2
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    current.write_text("{}\n", encoding="utf-8")
    baseline.write_text("{}\n", encoding="utf-8")

    common_env = {
        **os.environ,
        "PYTHON_BIN": str(fake_python),
        "PERFGATE_BASELINE_AVAILABLE": "1",
        "PERFGATE_BASELINE_FILE": str(baseline),
        "PERFGATE_STAGE1_CURRENT_FILE": str(current),
        "PERFGATE_REPORT_FILE": str(tmp_path / "report.md"),
        "GITHUB_ENV": str(tmp_path / "github-env"),
    }
    enforce_result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "perfgate_stage1_compare.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={**common_env, "PERFGATE_MODE": "enforce"},
    )
    report_result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "perfgate_stage1_compare.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={**common_env, "PERFGATE_MODE": "report"},
    )

    assert enforce_result.returncode == 2
    assert report_result.returncode == 0


def test_stage1_missing_baseline_fails_in_enforce_mode(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "perfgate_stage1_compare.sh")],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PERFGATE_MODE": "enforce",
            "PERFGATE_BASELINE_AVAILABLE": "0",
            "PERFGATE_REPORT_FILE": str(tmp_path / "report.md"),
            "GITHUB_ENV": str(tmp_path / "github-env"),
        },
    )

    assert result.returncode == 2
    assert "Stage 1 performance gate skipped" in result.stdout


def test_required_perfgate_validator_rejects_incomplete_gate() -> None:
    result = subprocess.run(
        ["bash", str(PERFGATE_VALIDATE_REQUIRED_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PERFGATE_REQUIRED": "1"},
    )

    assert result.returncode == 2
    assert "incomplete or failed" in result.stderr


def test_required_perfgate_validator_accepts_complete_gate(tmp_path: Path) -> None:
    stage1_baseline = tmp_path / "stage1-baseline.json"
    stage2_current = tmp_path / "stage2-current.json"
    stage2_baseline = tmp_path / "stage2-baseline.json"
    report = tmp_path / "perfgate-report.md"
    for path in (stage1_baseline, stage2_current, stage2_baseline, report):
        path.write_text("{}\n", encoding="utf-8")

    env = {
        **os.environ,
        "PERFGATE_REQUIRED": "1",
        "PERFGATE_MODE": "enforce",
        "BENCH_SCENARIO_COUNT": "1",
        "BENCH_SCENARIO": "random-online",
        "PERFGATE_BASELINE_AVAILABLE": "1",
        "PERFGATE_STAGE1_COMPLETED": "1",
        "PERFGATE_STAGE1_RESULT": "pass",
        "PERFGATE_STAGE2_EXECUTED": "1",
        "PERFGATE_STAGE2_BASELINE_AVAILABLE": "1",
        "PERFGATE_STAGE2_COMPLETED": "1",
        "PERFGATE_STAGE2_RESULT": "pass",
        "PERFGATE_STAGE2_SKIPPED": "0",
        "PERFGATE_STAGE2_REBASE_CONFLICT": "0",
        "PERFGATE_RESULT": "pass",
        "PERFGATE_BASELINE_FILE": str(stage1_baseline),
        "PERFGATE_STAGE2_B1PRIME_FILE": str(stage2_current),
        "PERFGATE_STAGE2_M2_BASELINE_FILE": str(stage2_baseline),
        "PERFGATE_REPORT_FILE": str(report),
    }
    result = subprocess.run(
        ["bash", str(PERFGATE_VALIDATE_REQUIRED_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert "completed successfully" in result.stdout


def test_schedule_runs_registered_multi_scenario_benchmark_publish() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'cron: "0 19 * * *"' in workflow
    assert "github.event_name == 'schedule'" in workflow
    assert "VLLM_ASCEND_HUST_SCHEDULE_BENCHMARK_SCENARIOS" in workflow
    assert "VLLM_ASCEND_HUST_SCHEDULE_PUBLISH_BENCHMARK != '0'" in workflow
    for scenario in (
        "random-online",
        "sharegpt-online",
        "prefix-repetition-online",
        "random-latency",
        "sharegpt-throughput",
        "sonnet-throughput",
        "instructcoder-online",
        "agent-research-online",
        "visionarena-online",
    ):
        assert scenario in workflow


def test_benchmark_runner_resolves_same_spec_without_random_online_default() -> None:
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")

    assert "SAME_SPEC_SPEC_FILE=${SAME_SPEC_SPEC_FILE:-}" in runner_script
    assert "SAME_SPEC_PR_PREVIEW_COMPAT=${SAME_SPEC_PR_PREVIEW_COMPAT:-1}" in runner_script
    assert "SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS=${SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS:-300}" in runner_script
    assert "vllm_hust_benchmark.perfgate_specs resolve" in runner_script
    assert '--scenario "$BENCH_SCENARIO"' in runner_script
    assert '--hardware-chip-model "$HARDWARE_CHIP_MODEL"' in runner_script
    assert '--repo-root "$VLLM_HUST_BENCHMARK_REPO"' in runner_script
    assert "official-ascend-jan-2026-v0180-random-online-qwen25-14b-910b2.json" not in runner_script
    assert 'if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then' in runner_script
    same_spec_block = runner_script[
        runner_script.index('if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then') : runner_script.index(
            "else", runner_script.index('if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then')
        )
    ]
    assert "EFFECTIVE_CONSTRAINTS_FILE=$SAME_SPEC_CONSTRAINTS_FILE" in same_spec_block
    assert "bench_args=()" in same_spec_block
    assert (
        'if [[ "$BENCH_SCENARIO" == "random-online" && "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then'
    ) not in runner_script
    sharegpt_block = runner_script[runner_script.index("    sharegpt-online)") :]
    sharegpt_block = sharegpt_block[: sharegpt_block.index("    *)")]
    assert "BENCH_DATASET_PATH is required for sharegpt-online" in sharegpt_block
    assert 'CLIENT_READY_CHECK_TIMEOUT_SECONDS="$SAME_SPEC_CLIENT_READY_TIMEOUT_SECONDS"' in runner_script
    assert "print_same_spec_server_log_tail" in runner_script
    assert "prepare_same_spec_pr_preview_compat_file()" in runner_script
    assert 'server_parameters["no_enable_chunked_prefill"] = True' in runner_script
    assert 'server_parameters["no_enable_prefix_caching"] = True' in runner_script
    assert 'client_parameters.setdefault("temperature", 0)' in runner_script
    assert 'client_parameters["max_concurrency"] = 1' in runner_script
    assert 'client_parameters["request_rate"] = 1' in runner_script
    assert '"$SAME_SPEC_PR_PREVIEW_COMPAT" == "1"' in runner_script
    assert '"$effective_same_spec_file"' in runner_script
    validation_failure_block = runner_script[runner_script.index('if [[ "$validation_status" -ne 0 ]]; then') :]
    validation_failure_block = validation_failure_block[: validation_failure_block.index("  fi")]
    assert "print_same_spec_server_log_tail" in validation_failure_block


def test_formal_main_and_perfgate_producer_keep_separate_workload_sizes() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    preview_spec_events = (
        "github.event_name == 'pull_request' || github.event_name == 'issue_comment' || "
        "(github.event_name == 'workflow_dispatch' && "
        "inputs.benchmark_scenarios == 'perfgate-bootstrap')"
    )
    for setting in (
        "MODEL_NAME:",
        "MODEL_PARAMETERS:",
        "MODEL_PRECISION:",
        "DTYPE:",
        "BENCH_RANDOM_INPUT_LEN:",
        "BENCH_RANDOM_OUTPUT_LEN:",
    ):
        line = next(line for line in workflow.splitlines() if line.strip().startswith(setting))
        assert preview_spec_events in line
        assert "github.event_name == 'push'" not in line

    assert "'Qwen/Qwen2.5-3B-Instruct'" in workflow
    assert "&& '3B' || (github.event_name == 'workflow_dispatch' && inputs.model_parameters || '14B')" in workflow
    assert "&& 'BF16' ||" in workflow
    assert "&& 'bfloat16' ||" in workflow
    assert "&& '64' || '1024'" in workflow
    assert "&& '16' || '256'" in workflow

    producer_start = workflow.index("- name: Run Plugin perfgate baseline producer")
    producer_end = workflow.index("- name: Upload Plugin perfgate producer artifact")
    producer = workflow[producer_start:producer_end]
    assert "MODEL_NAME: Qwen/Qwen2.5-3B-Instruct" in producer
    assert "MODEL_PARAMETERS: 3B" in producer
    assert "MODEL_PRECISION: BF16" in producer
    assert "DTYPE: bfloat16" in producer
    assert 'BENCH_RANDOM_INPUT_LEN: "64"' in producer
    assert 'BENCH_RANDOM_OUTPUT_LEN: "16"' in producer


def test_main_perfgate_producer_is_reachable_and_pins_dependencies() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "push:" in workflow
    assert "branches:\n      - main" in workflow
    assert "(github.event_name == 'push' && github.ref == 'refs/heads/main')" in workflow
    assert "Resolve trusted main dependency SHAs" in workflow
    for variable in (
        "VLLM_ASCEND_HUST_PERFGATE_BENCHMARK_RUNNER_SHA",
        "VLLM_ASCEND_HUST_PERFGATE_VLLM_HUST_SHA",
        "VLLM_ASCEND_HUST_PERFGATE_RUNTIME_MANAGER_SHA",
    ):
        assert variable in workflow
    assert '[[ ! "$value" =~ ^[0-9a-f]{40}$ ]]' in workflow
    assert "Verify trusted main dependency SHAs" in workflow
    assert "Run Plugin perfgate baseline producer" in workflow
    producer_start = workflow.index("- name: Run Plugin perfgate baseline producer")
    formal_start = workflow.index("- name: Run benchmark CI and optional formal publish")
    producer = workflow[producer_start:formal_start]
    assert 'PERFGATE_WARMUP_RUNS: "1"' in producer
    assert 'PERFGATE_MEASURED_RUNS: "3"' in producer
    assert 'PERFGATE_AGGREGATION: "primary-median-run"' in producer
    assert 'SAME_SPEC_PR_PREVIEW_COMPAT: "0"' in producer
    assert "fetch_with_retry()" in producer
    assert "fetch_with_retry --unshallow origin main" in producer
    assert "fetch_with_retry origin main" in producer
    assert "prepare_plugin_perfgate_artifact.py" in producer
    assert producer.index("Upload Plugin perfgate producer artifact") < producer.index(
        "Publish central Plugin perfgate baseline"
    )
    assert producer_start < workflow.index("Upload Plugin perfgate producer artifact")
    assert workflow.index("Publish central Plugin perfgate baseline") < formal_start
    publication = producer[producer.index("Publish central Plugin perfgate baseline") :]
    assert "VLLM_ASCEND_HUST_CENTRAL_BASELINE_WRITER_TOKEN" in publication
    assert "plugin-run_leaderboard.json" in publication
    assert "--runtime-manager-sha" in workflow
    assert "store-main-perfgate-baseline:" not in workflow
    formal = workflow[formal_start : workflow.index("- name: Performance gate - Stage 1 comparison")]
    assert "!(github.event_name == 'push' && github.ref == 'refs/heads/main')" not in formal
    snapshot_step = workflow[workflow.index("- name: Sync GitHub leaderboard snapshots") :]
    snapshot_step = snapshot_step[: snapshot_step.index("- name: Release Ascend hardware lock")]
    assert "!(github.event_name == 'push' && github.ref == 'refs/heads/main')" not in snapshot_step


def test_pull_request_trigger_preserves_ready_labels_on_updates() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "types: [labeled, synchronize, reopened]" in workflow
    assert "contains(github.event.pull_request.labels.*.name, 'ready')" in workflow
    assert "contains(github.event.pull_request.labels.*.name, 'verified')" in workflow
    assert "github.event.label.name" not in workflow


def test_plugin_producer_preserves_measurement_and_provenance_evidence() -> None:
    runner = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")
    store = (SCRIPT_DIR / "perfgate_store_baseline.sh").read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    for variable in (
        "PERFGATE_WARMUP_RUNS",
        "PERFGATE_MEASURED_RUNS",
        "PERFGATE_AGGREGATION",
    ):
        assert variable in runner[runner.index("SUDO_PRESERVE_ENV_VARS=(") :]
    assert runner.count('PERFGATE_WARMUP_RUNS="$PERFGATE_WARMUP_RUNS"') == 2
    assert runner.count('PERFGATE_MEASURED_RUNS="$PERFGATE_MEASURED_RUNS"') == 2
    assert 'cp "$same_spec_submission_dir/measurement.json"' in runner
    assert '"schema_version": "perfgate-runtime-provenance/v1"' in runner
    assert '"benchmark_runner_sha"' in runner
    assert '"runtime_manager_sha"' in runner
    assert '"cann_version"' in runner
    assert '"torch_npu_version"' in runner
    assert "NPU_MEMORY_EXIT_CODE=${NPU_MEMORY_EXIT_CODE:-87}" in runner
    assert "same_spec_server_log_indicates_npu_memory_pressure" in runner
    assert "ACL_ERROR_RT_MEMORY_ALLOCATION" in runner

    assert "plugin-run_leaderboard.json" in store
    assert "verify_raw_result_evidence warmup" in store
    assert "verify_raw_result_evidence per_run" in store
    assert "vllm_hust_benchmark.perfgate_baselines publish" in store
    assert '--measurement-file "$MEASUREMENT_FILE"' in store
    assert '--runtime-manager-sha "$RUNTIME_MANAGER_SHA"' in store
    assert "GIT_ASKPASS" in store
    assert "git worktree" not in store
    assert workflow.count("secrets.VLLM_ASCEND_HUST_CENTRAL_BASELINE_WRITER_TOKEN") == 1
    producer = workflow[workflow.index("- name: Run Plugin perfgate baseline producer") :]
    producer = producer[: producer.index("- name: Run benchmark CI and optional formal publish")]
    assert "cleanup_ascend_benchmark_processes.sh current" in producer
    assert "cleanup_ascend_ci_processes.sh" not in workflow


def test_plugin_scheme_c_is_producer_only() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")
    store = (SCRIPT_DIR / "perfgate_store_baseline.sh").read_text(encoding="utf-8")

    formal_step = workflow.index("- name: Run benchmark CI and optional formal publish")
    stage1_step = workflow.index("- name: Performance gate - Stage 1 comparison")
    stage2_step = workflow.index("- name: Performance gate - Stage 2 trial rebase and benchmark")
    final_compare_step = workflow.index("- name: Performance gate - two-stage comparison", stage2_step)
    for block in (
        workflow[formal_step:stage1_step],
        workflow[stage2_step:final_compare_step],
    ):
        assert "PERFGATE_WARMUP_RUNS:" not in block
        assert "PERFGATE_MEASURED_RUNS:" not in block
        assert "PERFGATE_AGGREGATION:" not in block

    assert "PERFGATE_WARMUP_RUNS=${PERFGATE_WARMUP_RUNS:-0}" in runner
    assert "PERFGATE_MEASURED_RUNS=${PERFGATE_MEASURED_RUNS:-1}" in runner
    assert "PERFGATE_AGGREGATION=${PERFGATE_AGGREGATION:-primary-median-run}" in runner
    assert "perfgate-measurement/v2" in store
    assert "warmup+primary-median-run" in store
    assert 'secondary_sort_key == "run_index"' in store


def test_benchmark_disables_huggingface_xet_download_path() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert 'HF_HUB_DISABLE_XET: "1"' in workflow
    assert "HF_ENDPOINT:" in workflow
    assert "HUGGINGFACE_HUB_CACHE:" in workflow
    assert "TRANSFORMERS_CACHE:" in workflow


def test_local_ascend_manager_fallback_bootstraps_pip() -> None:
    helper = MANAGER_HELPER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "hust_ensure_python_pip()" in helper
    assert '"${python_bin}" -m ensurepip --upgrade' in helper
    assert "https://bootstrap.pypa.io/get-pip.py" in helper
    assert '"${python_bin}" "${get_pip_script}" --user' in helper
    assert "${CI_HOME:-}" in helper
    assert '"${ci_home}/miniconda3/envs/${env_name}"' in helper
    prepare_step = workflow[workflow.index("Prepare Ascend runtime and install repos") :]
    assert "source scripts/hust_ascend_manager_helper.sh" in prepare_step
    assert prepare_step.index("source scripts/hust_ascend_manager_helper.sh") < prepare_step.index(
        'PYTHON_BIN="$(hust_resolve_python_bin)"'
    )
    assert 'export VLLM_HUST_PYTHON_BIN="$PYTHON_BIN"' in workflow
    assert "VLLM_HUST_PYTHON_BIN=$VLLM_HUST_PYTHON_BIN" in workflow
    assert "_hust_ascend_manager_command_needs_pip()" in helper
    assert "--install-python-stack|--install-plugin" in helper
    fallback = helper[helper.index("hust_ascend_manager_run()") :]
    assert 'if _hust_ascend_manager_command_needs_pip "$@"; then' in fallback
    assert 'hust_ensure_python_pip "${python_bin}" || return 1' in fallback
    assert '"${python_bin}" -m hust_ascend_manager.cli "$@"' in fallback


def test_single_ascend_env_falls_back_when_manager_env_fails() -> None:
    single_env = USE_SINGLE_ASCEND_ENV_SCRIPT.read_text(encoding="utf-8")

    assert "manager_env_status=0" in single_env
    assert 'manager_env="$(hust_ascend_manager_run env --shell' in single_env
    assert "manager_env_status=$?" in single_env
    assert 'if [[ "${manager_env_status}" -eq 0 ]]; then' in single_env
    assert 'eval "${manager_env}"' in single_env
    assert "falling back to local CANN set_env.sh discovery" in single_env
    assert "/usr/local/Ascend/cann-*/set_env.sh" in single_env
    assert '[[ -n "${ASCEND_HOME_PATH:-}" && -n "${ASCEND_OPP_PATH:-}" ]] && python_can_import_tbe' in single_env
    assert "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-<unset>}" in single_env


def test_local_plugin_editable_install_bootstraps_build_metadata_deps() -> None:
    install_script = INSTALL_PLUGIN_SCRIPT.read_text(encoding="utf-8")

    assert "import setuptools_scm" in install_script
    assert "import wheel.bdist_wheel" in install_script
    assert 'hust_run_pip install "setuptools-scm>=8"' in install_script
    assert 'hust_run_pip install "wheel"' in install_script
    assert 'hust_run_pip install -e "${PLUGIN_REPO}" --no-build-isolation --no-deps' in install_script


def test_benchmark_prepare_preserves_torch_npu_stack() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    prepare_step = workflow[workflow.index("Prepare Ascend runtime and install repos") :]
    prepare_step = prepare_step[: prepare_step.index("- name: Verify installation")]

    assert "install_ascend_benchmark_with_dev_hub.sh" in prepare_step
    assert "hust_ascend_manager_run setup --non-interactive" not in prepare_step
    assert "run_in_quickstart_env()" not in prepare_step
    assert 'mktemp "${RUNNER_TEMP:-/tmp}/benchmark-quickstart-env.' not in prepare_step
    assert '"$CONDA_BIN" run -n "vllm-hust-dev" bash "$inline_script"' not in prepare_step
    assert "find_library('stdc++')" in prepare_step
    assert 'PYTHON_BIN="${VLLM_HUST_PYTHON_BIN:-}"' in prepare_step
    assert 'echo "PYTHON_BIN=$PYTHON_BIN"' in prepare_step
    assert '"$PYTHON_BIN" - <<' in prepare_step
    assert 'echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"' in prepare_step
    assert '} >> "$GITHUB_ENV"' in prepare_step
    assert 'python -m pip install -e "$VLLM_HUST_BENCHMARK_REPO[publish]" jsonschema' not in prepare_step
    assert 'python -m pip install "huggingface_hub>=0.20"' not in prepare_step
    assert 'python -m pip install "numpy<2.0.0" scipy attrs decorator psutil' not in prepare_step
    assert (
        'python -m pip install -c "$torch_constraints" -r "$VLLM_HUST_REPO/requirements/common.txt"'
    ) not in prepare_step
    assert "VLLM_HUST_PYTHON_BIN" in prepare_step


def test_benchmark_bootstrap_supports_container_native_python() -> None:
    install_script = INSTALL_DEV_HUB_SCRIPT.read_text(encoding="utf-8")

    assert 'CONDA_BIN="$(resolve_conda_bin 2>/dev/null || true)"' in install_script
    assert "VLLM_HUST_PYTHON_BIN:-$(command -v python3" in install_script
    assert "Conda is unavailable; reusing container Python" in install_script
    assert "Skipping conda runtime library installation in container-native Python mode" in install_script
    assert 'marker_root="${CI_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}}/ascend-benchmark"' in install_script


def test_benchmark_verify_uses_resolved_python_not_conda_lookup() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    verify_step = workflow[workflow.index("Verify installation") :]
    verify_step = verify_step[: verify_step.index("- name: Performance gate - fetch Stage 1 baseline")]

    assert "source scripts/hust_ascend_manager_helper.sh" in verify_step
    assert 'PYTHON_BIN="${VLLM_HUST_PYTHON_BIN:-}"' in verify_step
    assert 'PYTHON_BIN="$(hust_resolve_python_bin)"' in verify_step
    assert 'export VLLM_HUST_PYTHON_BIN="$PYTHON_BIN"' in verify_step
    assert "source scripts/use_single_ascend_env.sh" in verify_step
    assert '"$PYTHON_BIN" --version' in verify_step
    assert '"$PYTHON_BIN" - <<' in verify_step
    assert "conda executable not found for Verify installation" not in verify_step
    assert 'CONDA_BIN="${CONDA_EXE:-}"' not in verify_step
    assert 'conda run -n "vllm-hust-dev"' not in verify_step


def test_benchmark_runner_auto_disables_sudo_when_unavailable() -> None:
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")

    assert 'if [[ "$ASCEND_BENCHMARK_USE_SUDO" == "auto" ]]; then' in runner_script
    assert 'if [[ "$(id -u)" == "0" ]]; then' in runner_script
    assert "current user is root" in runner_script
    assert "command -v sudo" in runner_script
    assert "Ascend benchmark sudo mode: disabled via auto detection" in runner_script
    assert "command not found" in runner_script[runner_script.index("runtime_ready_log_indicates_sudo_auth_failure") :]


def test_benchmark_server_uses_inferred_max_model_len_by_default() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")
    root_helper = (SCRIPT_DIR / "run_ascend_benchmark_root_helper.sh").read_text(encoding="utf-8")

    assert 'MAX_MODEL_LEN: ""' in workflow
    assert "MAX_MODEL_LEN=${MAX_MODEL_LEN:-}" in runner_script
    assert "max_model_len_args=()" in runner_script
    assert '"${max_model_len_args[@]}"' in runner_script
    assert runner_script.count('"${max_model_len_args[@]}"') == 1
    assert "max_model_len_args=()" in root_helper
    assert '"${max_model_len_args[@]}"' in root_helper
    assert "MAX_MODEL_LEN must be set" not in root_helper


def test_benchmark_server_uses_configurable_eager_and_completions_smoke() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")
    root_helper = (SCRIPT_DIR / "run_ascend_benchmark_root_helper.sh").read_text(encoding="utf-8")

    assert "ASCEND_BENCHMARK_ENFORCE_EAGER:" in workflow
    assert "VLLM_ASCEND_HUST_BENCHMARK_ENFORCE_EAGER || '0'" in workflow
    assert "ASCEND_BENCHMARK_ENFORCE_EAGER=${ASCEND_BENCHMARK_ENFORCE_EAGER:-0}" in runner_script
    assert "ASCEND_BENCHMARK_ENFORCE_EAGER" in runner_script[runner_script.index("SUDO_PRESERVE_ENV_VARS=(") :]
    assert "serve_extra_args=()" in runner_script
    assert "serve_extra_args+=(--enforce-eager)" in runner_script
    assert "run_completions_smoke()" in runner_script
    assert "wait_for_completions_smoke()" in runner_script
    assert "CHAT_SMOKE_TIMEOUT_SECONDS=${CHAT_SMOKE_TIMEOUT_SECONDS:-120}" in runner_script
    assert "CHAT_SMOKE_POLL_SECONDS=${CHAT_SMOKE_POLL_SECONDS:-5}" in runner_script
    assert "CHAT_SMOKE_REQUEST_TIMEOUT_SECONDS=${CHAT_SMOKE_REQUEST_TIMEOUT_SECONDS:-15}" in runner_script
    assert "/v1/completions" in runner_script
    assert "/v1/chat/completions" not in runner_script
    assert "completions_smoke.json" in runner_script
    assert "text.strip()" in runner_script
    assert "completion_tokens > 0" in runner_script
    assert "if wait_for_completions_smoke; then" in runner_script
    assert "Timed out waiting for completions smoke" in runner_script
    assert "--enforce-eager >" not in runner_script
    assert "serve_extra_args=()" in root_helper
    assert "serve_extra_args+=(--enforce-eager)" in root_helper
    assert '"${serve_extra_args[@]}"' in root_helper


def test_same_spec_benchmark_uses_persistent_cache_and_configurable_timeout() -> None:
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")

    assert "SAME_SPEC_READY_TIMEOUT_SECONDS=" in runner_script
    assert "CURRENT_VLLM_CACHE_ROOT=" in runner_script
    assert 'CURRENT_VLLM_CACHE_ROOT="$CURRENT_VLLM_CACHE_ROOT"' in runner_script
    assert 'READY_TIMEOUT_SECONDS="$SAME_SPEC_READY_TIMEOUT_SECONDS"' in runner_script
    assert "SAME_SPEC_READY_TIMEOUT_SECONDS" in runner_script[runner_script.index("SUDO_PRESERVE_ENV_VARS=(") :]


def test_stage2_trial_does_not_publish_benchmark_results() -> None:
    stage2_script = (SCRIPT_DIR / "perfgate_stage2_rebase_and_benchmark.sh").read_text(encoding="utf-8")

    assert "PUBLISH_TO_HF=0" in stage2_script
    assert "PUBLISH_TO_BENCHMARK_REPO=0" in stage2_script
    assert "SYNC_GITHUB_SNAPSHOTS=0" in stage2_script
    assert "BENCHMARK_RESULTS_ROOT" in stage2_script
    assert "install_ascend_benchmark_with_dev_hub.sh" in stage2_script
    assert "PERFGATE_STAGE2_DEV_HUB_QUICKSTART_CONDA" not in stage2_script
    assert "install_local_ascend_plugin.sh" not in stage2_script
    assert 'GIT_AUTHOR_NAME="vLLM-HUST Benchmark Bot"' in stage2_script
    assert 'GIT_AUTHOR_EMAIL="benchmark-bot@vllm-hust.local"' in stage2_script
    assert 'GIT_COMMITTER_NAME="vLLM-HUST Benchmark Bot"' in stage2_script
    assert 'GIT_COMMITTER_EMAIL="benchmark-bot@vllm-hust.local"' in stage2_script
    assert "git config --global" not in stage2_script


def test_dev_hub_install_wrapper_centralizes_custom_kernel_policy() -> None:
    install_script = INSTALL_DEV_HUB_SCRIPT.read_text(encoding="utf-8")

    assert "VLLM_HUST_REPO=" in install_script
    assert "VLLM_HUST_BENCHMARK_REPO=" in install_script
    assert "VLLM_HUST_DEV_HUB_REPO=" not in install_script
    assert "ascend-runtime-manager checkout not found" in install_script
    assert "detect_cann_major_version()" in install_script
    assert 'if [[ "$requested" == "auto" ]]; then' in install_script
    assert 'if [[ "$cann_major" == "9" ]] && ascend_custom_kernel_build_prereqs_present; then' in install_script
    assert "Using install-only repo bootstrap (no quickstart; editable --no-deps installs)" in install_script
    assert "COMPILE_CUSTOM_KERNELS=auto resolved to lightweight mode" in install_script
    assert "requirements/common.txt" in install_script
    assert 'run_env_pip install -r "$VLLM_HUST_REPO/requirements/common.txt"' not in install_script
    assert "read_requirement_specs_from_file()" in install_script
    assert 'ensure_python_requirements "vllm-hust runtime requirements"' in install_script
    assert "ASCEND_BENCHMARK_TRITON_ASCEND_INDEX_URL" in install_script
    assert "https://mirrors.huaweicloud.com/ascend/repos/pypi" in install_script
    assert "ensure_triton_ascend()" in install_script
    assert (
        'run_env_pip install --no-deps --index-url "$ASCEND_BENCHMARK_TRITON_ASCEND_INDEX_URL" "$triton_ascend_spec"'
    ) in install_script
    assert "Preinstall these packages on the self-hosted runner" not in install_script
    assert "ascend_custom_kernel_build_prereqs_present()" in install_script
    assert 'if [[ "$cann_major" == "9" ]] && ascend_custom_kernel_build_prereqs_present; then' in install_script
    assert 'install -e "$repo_path" --no-build-isolation --no-deps' in install_script
    assert 'bash "$VLLM_ASCEND_HUST_REPO/scripts/install_local_ascend_plugin.sh"' in install_script
    assert "ASCEND_BENCHMARK_STACK_MARKER_VERSION" in install_script
    assert "sha256sum" in install_script
    assert '"huggingface_hub>=0.20"' in install_script
    assert '"jsonschema>=4"' in install_script
    assert "HUST_DEV_HUB_SKIP_ASCEND_SYSTEM_APPLY=1" not in install_script
    assert 'bash "$VLLM_HUST_DEV_HUB_REPO/scripts/quickstart.sh"' not in install_script


def test_benchmark_workflow_masks_cross_service_credentials() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "Mask benchmark credentials" in workflow
    assert "::add-mask::" in workflow
    assert '"HF_TOKEN"' in workflow
    assert '"BENCHMARK_REPO_GH_TOKEN"' in workflow
    assert '"BENCHMARK_REPO_SSH_KEY"' in workflow


def test_benchmark_repo_publish_is_gated_and_reported() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    runner_script = (SCRIPT_DIR / "run_ascend_benchmark_ci.sh").read_text(encoding="utf-8")
    sync_script = (SCRIPT_DIR / "sync_benchmark_snapshots_to_github.sh").read_text(encoding="utf-8")

    assert "PUBLISH_TO_BENCHMARK_REPO:" in workflow
    assert "BENCHMARK_REPO_GH_TOKEN:" in workflow
    assert "BENCHMARK_REPO_SSH_KEY:" in workflow
    assert "VLLM_ASCEND_HUST_SYNC_BENCHMARK_SNAPSHOTS_TO_GITHUB || '0'" in workflow
    assert ("github.event_name != 'issue_comment') && secrets.VLLM_HUST_BENCHMARK_GH_TOKEN") in workflow
    assert "L3 Benchmark Repository Publication" in workflow

    assert "PUBLISH_TO_BENCHMARK_REPO=${PUBLISH_TO_BENCHMARK_REPO:-0}" in runner_script
    assert "PUBLISH_TO_BENCHMARK_REPO" in runner_script[runner_script.index("SUDO_PRESERVE_ENV_VARS=(") :]
    assert 'if [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then' in runner_script
    assert 'if [[ "$PUBLISH_TO_BENCHMARK_REPO" == "1" ]]; then' in runner_script
    assert 'elif [[ "$PUBLISH_TO_HF" == "1" ]]; then' not in runner_script
    assert 'elif [[ "$PUBLISH_TO_BENCHMARK_REPO" != "1" ]]; then' in runner_script
    assert 'BENCHMARK_REPO_GH_TOKEN="${BENCHMARK_REPO_GH_TOKEN:-}" \\' in runner_script
    assert 'BENCHMARK_REPO_SSH_KEY="${BENCHMARK_REPO_SSH_KEY:-}" \\' in runner_script

    assert "L3 benchmark repository publication is enabled" in sync_script
    assert "no cross-repository write credential is available" in sync_script
    assert "VLLM_ASCEND_HUST_BENCHMARK_SSH_KEY" in sync_script
    assert "VLLM_HUST_BENCHMARK_GH_TOKEN" in sync_script
    assert "Benchmark repo publish target:" in sync_script

    staging_index = sync_script.index("publication_staging_dir=$(mktemp -d")
    public_validator_index = sync_script.index("validate_public_leaderboard_snapshots.py")
    trend_validator_index = sync_script.index("validate-trend --input")
    git_add_index = sync_script.index('git -C "$BENCHMARK_REPO_DIR" add')
    git_commit_index = sync_script.index('git -C "$BENCHMARK_REPO_DIR" commit')
    git_push_index = sync_script.index('git -C "$BENCHMARK_REPO_DIR" push')
    verify_index = sync_script.index("verify_published_benchmark_repo_state", git_push_index)
    assert staging_index < public_validator_index < trend_validator_index < git_add_index
    assert git_add_index < git_commit_index < git_push_index
    assert git_push_index < verify_index
    assert "write_github_env GITHUB_SNAPSHOT_SYNC_STATUS rejected" in sync_script
    assert "required_submission_files=(leaderboard_manifest.json run_leaderboard.json STATUS)" in sync_script
    assert "reset_publication_staging()" in sync_script
    assert "reset_publication_staging || return $?" in sync_script
    submit_index = runner_script.index('"${PYTHON_BIN}" -m vllm_hust_benchmark.cli submit')
    status_index = runner_script.index("printf 'OK\\n' > \"$SUBMISSION_DIR/STATUS\"")
    sync_index = runner_script.index("sync_benchmark_publication_to_github", status_index)
    assert submit_index < status_index < sync_index


def _write_snapshot_sync_test_doubles(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    git_log = tmp_path / "git.log"

    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_GIT_LOG"
args=("$@")
if [[ "${args[0]:-}" == "-C" ]]; then
  args=("${args[@]:2}")
fi
if [[ "${args[0]:-}" == "fetch" && -n "${FAKE_GIT_FETCH_EXIT:-}" ]]; then
  exit "$FAKE_GIT_FETCH_EXIT"
fi
if [[ "${args[0]:-}" == "diff" ]]; then
  exit "${FAKE_GIT_DIFF_EXIT:-1}"
fi
if [[ "${args[0]:-}" == "rev-parse" ]]; then
  printf 'fake-publication-commit\\n'
fi
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)

    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail
if [[ "$*" == *"publish-website"* ]]; then
  while (( $# > 0 )); do
    if [[ "$1" == "--output-dir" ]]; then
      output_dir=$2
      break
    fi
    shift
  done
  mkdir -p "$output_dir"
  for snapshot in leaderboard_single.json leaderboard_multi.json leaderboard_compare.json last_updated.json; do
    printf '{"snapshot":"%s"}\\n' "$snapshot" > "$output_dir/$snapshot"
  done
  exit 0
fi
if [[ "$*" == *"validate_public_leaderboard_snapshots.py"* ]]; then
  exit "${FAKE_PUBLIC_VALIDATOR_EXIT:-0}"
fi
if [[ "$*" == *"validate-trend"* ]]; then
  exit "${FAKE_TREND_VALIDATOR_EXIT:-0}"
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    return fake_bin, git_log


def _snapshot_sync_env(tmp_path: Path, fake_bin: Path, git_log: Path) -> tuple[dict[str, str], Path]:
    benchmark_repo = tmp_path / "benchmark-repo"
    (benchmark_repo / ".git").mkdir(parents=True)
    (benchmark_repo / "submissions").mkdir()
    current_submission = tmp_path / "current-submission"
    current_submission.mkdir()
    (current_submission / "leaderboard_manifest.json").write_text("{}\n", encoding="utf-8")
    (current_submission / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")
    (current_submission / "STATUS").write_text("OK\n", encoding="utf-8")
    website_repo = tmp_path / "website-repo"
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text("", encoding="utf-8")
    hust_repo = tmp_path / "vllm-hust"
    hust_repo.mkdir()
    (hust_repo / "pyproject.toml").write_text("", encoding="utf-8")

    return (
        {
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "PYTHON_BIN": str(fake_bin / "python"),
            "FAKE_GIT_LOG": str(git_log),
            "BENCHMARK_REPO_DIR": str(benchmark_repo),
            "WEBSITE_REPO_DIR": str(website_repo),
            "VLLM_HUST_REPO_DIR": str(hust_repo),
            "CURRENT_SUBMISSION_DIR": str(current_submission),
            "RUN_ID": "test-run",
            "GITHUB_ACTIONS": "true",
            "BENCHMARK_REPO_GH_TOKEN": "test-token",
            "GITHUB_ENV": str(tmp_path / "github-env"),
        },
        benchmark_repo,
    )


def test_snapshot_sync_rejects_invalid_publication_before_git_writes(tmp_path: Path) -> None:
    fake_bin, git_log = _write_snapshot_sync_test_doubles(tmp_path)
    env, benchmark_repo = _snapshot_sync_env(tmp_path, fake_bin, git_log)
    env["FAKE_PUBLIC_VALIDATOR_EXIT"] = "2"

    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "sync_benchmark_snapshots_to_github.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    assert "publication admission failed at public snapshot validation" in result.stderr
    git_commands = git_log.read_text(encoding="utf-8")
    assert " add " not in f" {git_commands} "
    assert " commit " not in f" {git_commands} "
    assert " push " not in f" {git_commands} "
    assert not (benchmark_repo / "submissions" / "test-run").exists()
    assert not (benchmark_repo / "leaderboard-data" / "snapshots").exists()
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=rejected" in (tmp_path / "github-env").read_text(encoding="utf-8")


def test_snapshot_sync_stops_when_prepare_step_fails(tmp_path: Path) -> None:
    fake_bin, git_log = _write_snapshot_sync_test_doubles(tmp_path)
    env, benchmark_repo = _snapshot_sync_env(tmp_path, fake_bin, git_log)
    env["FAKE_GIT_FETCH_EXIT"] = "7"

    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "sync_benchmark_snapshots_to_github.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 7, result.stderr
    git_commands = git_log.read_text(encoding="utf-8")
    assert " checkout " not in f" {git_commands} "
    assert " add " not in f" {git_commands} "
    assert " commit " not in f" {git_commands} "
    assert " push " not in f" {git_commands} "
    assert not (benchmark_repo / "submissions" / "test-run").exists()
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=rejected" in (tmp_path / "github-env").read_text(encoding="utf-8")


def test_snapshot_sync_rejects_git_diff_errors(tmp_path: Path) -> None:
    fake_bin, git_log = _write_snapshot_sync_test_doubles(tmp_path)
    env, benchmark_repo = _snapshot_sync_env(tmp_path, fake_bin, git_log)
    env["FAKE_GIT_DIFF_EXIT"] = "128"

    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "sync_benchmark_snapshots_to_github.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 128, result.stderr
    git_commands = git_log.read_text(encoding="utf-8")
    assert " add " in f" {git_commands} "
    assert " commit " not in f" {git_commands} "
    assert " push " not in f" {git_commands} "
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=rejected" in (tmp_path / "github-env").read_text(encoding="utf-8")
    assert (benchmark_repo / "submissions" / "test-run").is_dir()


def test_snapshot_sync_publishes_only_after_validating_staged_output(tmp_path: Path) -> None:
    fake_bin, git_log = _write_snapshot_sync_test_doubles(tmp_path)
    env, benchmark_repo = _snapshot_sync_env(tmp_path, fake_bin, git_log)

    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "sync_benchmark_snapshots_to_github.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    submission_dir = benchmark_repo / "submissions" / "test-run"
    assert (submission_dir / "leaderboard_manifest.json").read_text(encoding="utf-8") == "{}\n"
    assert (submission_dir / "run_leaderboard.json").read_text(encoding="utf-8") == "{}\n"
    assert (submission_dir / "STATUS").read_text(encoding="utf-8") == "OK\n"
    snapshot_dir = benchmark_repo / "leaderboard-data" / "snapshots"
    for snapshot_name in (
        "leaderboard_single.json",
        "leaderboard_multi.json",
        "leaderboard_compare.json",
        "last_updated.json",
    ):
        assert (snapshot_dir / snapshot_name).read_text(encoding="utf-8") == f'{{"snapshot":"{snapshot_name}"}}\n'
    git_commands = git_log.read_text(encoding="utf-8")
    assert " add " in f" {git_commands} "
    assert " commit " in f" {git_commands} "
    assert " push " in f" {git_commands} "
    assert "GITHUB_SNAPSHOT_SYNC_STATUS=pushed" in (tmp_path / "github-env").read_text(encoding="utf-8")
    assert "GITHUB_SNAPSHOT_SYNC_VERIFICATION=verified" in (tmp_path / "github-env").read_text(encoding="utf-8")


def _run_git(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _init_retry_remote(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    remote.mkdir()
    _run_git(["git", "init", "--bare", str(remote)], tmp_path)
    _run_git(["git", "clone", str(remote), str(seed)], tmp_path)
    _run_git(["git", "config", "user.name", "Test"], seed)
    _run_git(["git", "config", "user.email", "test@example.com"], seed)
    (seed / "README.md").write_text("seed\n", encoding="utf-8")
    _run_git(["git", "add", "README.md"], seed)
    _run_git(["git", "commit", "-m", "seed"], seed)
    _run_git(["git", "push", "origin", "HEAD:main"], seed)
    return remote, seed


def _write_retry_test_doubles(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "retry-fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        """#!/bin/bash
set -euo pipefail

if [[ "$*" == *"validate_public_leaderboard_snapshots.py"* ]]; then
  exit 0
fi
if [[ "$*" == *"validate-trend"* ]]; then
  exit 0
fi

shift 3
source_dir=""
output_dir=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --source-dir)
      source_dir="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

mkdir -p "$output_dir"
if [[ -d "$source_dir/stale-ci" ]]; then
  stale_submission="stale-present"
else
  stale_submission="stale-absent"
fi
printf '{"stale_submission":"%s"}\\n' "$stale_submission" \\
  > "$output_dir/leaderboard_single.json"
printf '{}\\n' > "$output_dir/leaderboard_multi.json"
printf '{}\\n' > "$output_dir/leaderboard_compare.json"
printf '{}\\n' > "$output_dir/last_updated.json"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    fake_git = fake_bin / "git"
    fake_git.write_text(
        """#!/bin/bash
set -euo pipefail

for argument in "$@"; do
  if [[ "$argument" == "push" && ! -f "$FAKE_GIT_PUSH_STATE" ]]; then
    touch "$FAKE_GIT_PUSH_STATE"
    "$REAL_GIT" -C "$FAKE_GIT_SEED" rm -r submissions/stale-ci
    "$REAL_GIT" -C "$FAKE_GIT_SEED" commit -m "remove stale submission"
    "$REAL_GIT" -C "$FAKE_GIT_SEED" push origin HEAD:main
    exit 1
  fi
done

exec "$REAL_GIT" "$@"
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    return fake_bin, fake_python


def test_snapshot_sync_rebuilds_staging_before_push_retry(tmp_path: Path) -> None:
    remote, seed = _init_retry_remote(tmp_path)
    stale_submission = seed / "submissions" / "stale-ci"
    stale_submission.mkdir(parents=True)
    (stale_submission / "obsolete.txt").write_text("stale\n", encoding="utf-8")
    retained_submission = seed / "submissions" / "retained-ci"
    retained_submission.mkdir()
    (retained_submission / "result.txt").write_text("current\n", encoding="utf-8")
    _run_git(["git", "add", "submissions"], seed)
    _run_git(["git", "commit", "-m", "add stale submission"], seed)
    _run_git(["git", "push", "origin", "HEAD:main"], seed)

    benchmark_repo = tmp_path / "benchmark-repo"
    website_repo = tmp_path / "website-repo"
    hust_repo = tmp_path / "vllm-hust"
    current_submission = tmp_path / "current-submission"
    github_env = tmp_path / "github-env"
    fake_bin, fake_python = _write_retry_test_doubles(tmp_path)
    _run_git(["git", "clone", str(remote), str(benchmark_repo)], tmp_path)
    (website_repo / "scripts").mkdir(parents=True)
    (website_repo / "scripts" / "aggregate_results.py").write_text("# fake\n", encoding="utf-8")
    hust_repo.mkdir()
    (hust_repo / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")
    current_submission.mkdir()
    (current_submission / "leaderboard_manifest.json").write_text("{}\n", encoding="utf-8")
    (current_submission / "run_leaderboard.json").write_text("{}\n", encoding="utf-8")
    (current_submission / "STATUS").write_text("OK\n", encoding="utf-8")

    env = {
        **os.environ,
        "ALLOW_LOCAL_GIT_RESET": "1",
        "BENCHMARK_REPO_DIR": str(benchmark_repo),
        "BENCHMARK_REPO_REMOTE": "origin",
        "BENCHMARK_REPO_SLUG": "local/benchmark",
        "CURRENT_SUBMISSION_DIR": str(current_submission),
        "FAKE_GIT_PUSH_STATE": str(tmp_path / "first-push-failed"),
        "FAKE_GIT_SEED": str(seed),
        "GITHUB_ENV": str(github_env),
        "GITHUB_ACTIONS": "false",
        "BENCHMARK_REPO_GH_TOKEN": "",
        "BENCHMARK_REPO_SSH_KEY": "",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHON_BIN": str(fake_python),
        "REAL_GIT": shutil.which("git") or "git",
        "RUN_ID": "retry-ci-test",
        "SNAPSHOT_MAX_PUSH_ATTEMPTS": "2",
        "SNAPSHOT_PUSH_RETRY_SECONDS": "0",
        "SNAPSHOT_TARGET_BRANCH": "main",
        "VLLM_HUST_REPO_DIR": str(hust_repo),
        "WEBSITE_REPO_DIR": str(website_repo),
    }

    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "sync_benchmark_snapshots_to_github.sh")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "push failed; retrying with fresh origin/main" in result.stderr
    assert (tmp_path / "first-push-failed").is_file()
    assert (
        _run_git(
            [
                "git",
                "--git-dir",
                str(remote),
                "show",
                "main:leaderboard-data/snapshots/leaderboard_single.json",
            ],
            tmp_path,
        ).stdout.strip()
        == '{"stale_submission":"stale-absent"}'
    )
