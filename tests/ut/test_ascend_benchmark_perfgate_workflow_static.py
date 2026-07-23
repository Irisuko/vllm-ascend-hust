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
    ):
        assert (SCRIPT_DIR / script_name).is_file()


def test_ascend_benchmark_workflow_wires_two_stage_perfgate() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "PERFGATE_MODE" in workflow
    assert "PERFGATE_SPEC_FILE" in workflow
    assert "SOC_VERSION: ascend910b2" in workflow
    assert (
        "HARDWARE_CHIP_MODEL: ${{ github.event_name == 'workflow_dispatch' && inputs.hardware_chip_model || '910B2' }}"
        in workflow
    )
    assert "Resolve perfgate same-spec file" in workflow
    assert "Resolve main same-spec file" in workflow
    assert "resolve_perfgate_spec_file.py" in workflow
    assert "MAIN_SAME_SPEC_SPEC_FILE:" in workflow
    assert "github.event_name != 'pull_request' && github.event_name != 'issue_comment'" in workflow
    assert '--explicit-chip-model "${HARDWARE_CHIP_MODEL}"' in workflow
    assert '--benchmark-repo "${VLLM_HUST_BENCHMARK_REPO}"' in workflow
    assert '--explicit-same-spec-file ""' in workflow
    assert 'spec_file="${SAME_SPEC_SPEC_FILE:-$MAIN_SAME_SPEC_SPEC_FILE}"' in workflow
    assert "MAIN_BENCH_SCENARIO" in workflow
    assert '--scenario "${MAIN_BENCH_SCENARIO}"' in workflow
    assert '--repo-root "${GITHUB_WORKSPACE}/vllm-hust-benchmark"' in workflow
    assert "docs/official-baselines/perfgate-ascend-qwen25-3b-910b2.json" not in workflow
    assert "docs/official-baselines/perfgate-ascend-qwen25-3b-910b3.json" not in workflow
    assert "perfgate-ascend-qwen25-3b-910b3.json" not in workflow
    assert "VLLM_HUST_BENCHMARK_REF" in workflow
    assert "ref: ${{ env.VLLM_HUST_BENCHMARK_REF }}" in workflow
    assert 'hust_run_pip install -e "${VLLM_HUST_BENCHMARK_REPO}[publish]"' not in workflow
    assert "Detect PR fork point" in workflow
    assert "Performance gate - fetch Stage 1 baseline" in workflow
    assert "Performance gate - Stage 1 comparison" in workflow
    assert "Performance gate - Stage 2 trial rebase and benchmark" in workflow
    assert "Performance gate - two-stage comparison" in workflow
    assert "store-main-perfgate-baseline:" in workflow
    assert "Store main perfgate baseline" in workflow
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
        "(github.event_name == 'pull_request' || github.event_name == 'issue_comment') "
        "&& steps.resolve-scenario.outputs.BENCH_SCENARIO_COUNT == '1'"
    ) in workflow
    assert (
        "github.event_name != 'pull_request' && github.event_name != 'issue_comment' "
        "&& steps.resolve-scenario.outputs.BENCH_SCENARIO_COUNT == '1'"
    ) in workflow
    assert "vars.VLLM_ASCEND_HUST_MAIN_BENCHMARK_SCENARIOS == ''" in workflow
    assert "multi_scenario_results.tsv" in workflow
    assert "Perfgate comparison: `skipped for multi-scenario run" in workflow
    assert "os.environ.get('BENCH_SCENARIO_COUNT', '1') == '1'" in workflow
    assert "timeout-minutes: 60" in workflow
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
    stage1_script = (SCRIPT_DIR / "perfgate_stage1_compare.sh").read_text(
        encoding="utf-8"
    )
    stage2_script = (SCRIPT_DIR / "perfgate_stage2_rebase_and_benchmark.sh").read_text(
        encoding="utf-8"
    )

    assert 'write_env PERFGATE_STAGE1_COMPLETED 1' in stage1_script
    assert '"$MODE" == "enforce"' in stage1_script
    assert '"$MODE" != "enforce"' in stage2_script
    assert 'write_env PERFGATE_STAGE2_EXECUTED 1' in stage2_script
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
        runner_script.index('if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then') :
        runner_script.index('else', runner_script.index('if [[ "$SAME_SPEC_BENCHMARK_ENABLED" == "1" ]]; then'))
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
    validation_failure_block = runner_script[
        runner_script.index('if [[ "$validation_status" -ne 0 ]]; then') :
    ]
    validation_failure_block = validation_failure_block[
        : validation_failure_block.index("  fi")
    ]
    assert "print_same_spec_server_log_tail" in validation_failure_block


def test_pull_request_defaults_match_perfgate_spec_size() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert (
        "MODEL_NAME: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && "
        "(github.event_name == 'issue_comment' && "
        "needs.issue-comment-command.outputs.model_name || "
        "'Qwen/Qwen2.5-3B-Instruct') || "
        "(github.event_name == 'workflow_dispatch' && inputs.model_name || "
        "'Qwen/Qwen2.5-14B-Instruct') }}"
    ) in workflow
    assert (
        "MODEL_PARAMETERS: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && '3B' || '14B' }}"
    ) in workflow
    assert (
        "MODEL_PRECISION: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && 'BF16' || "
        "(github.event_name == 'workflow_dispatch' && inputs.model_precision || "
        "'FP16') }}"
    ) in workflow
    assert (
        "DTYPE: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && 'bfloat16' || "
        "(github.event_name == 'workflow_dispatch' && inputs.dtype || 'float16') }}"
    ) in workflow
    assert (
        "BENCH_RANDOM_INPUT_LEN: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && '64' || '1024' }}"
    ) in workflow
    assert (
        "BENCH_RANDOM_OUTPUT_LEN: ${{ (github.event_name == 'pull_request' || "
        "github.event_name == 'issue_comment') && '16' || '256' }}"
    ) in workflow


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
    assert "export VLLM_HUST_PYTHON_BIN=\"$PYTHON_BIN\"" in workflow
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
    assert 'manager_env_status=$?' in single_env
    assert 'if [[ "${manager_env_status}" -eq 0 ]]; then' in single_env
    assert 'eval "${manager_env}"' in single_env
    assert "falling back to local CANN set_env.sh discovery" in single_env
    assert "/usr/local/Ascend/cann-*/set_env.sh" in single_env
    assert '[[ -n "${ASCEND_HOME_PATH:-}" && -n "${ASCEND_OPP_PATH:-}" ]] && python_can_import_tbe' in single_env
    assert 'ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-<unset>}' in single_env


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
    assert 'run_in_quickstart_env()' not in prepare_step
    assert 'mktemp "${RUNNER_TEMP:-/tmp}/benchmark-quickstart-env.' not in prepare_step
    assert '"$CONDA_BIN" run -n "vllm-hust-dev" bash "$inline_script"' not in prepare_step
    assert "find_library('stdc++')" in prepare_step
    assert 'PYTHON_BIN="${VLLM_HUST_PYTHON_BIN:-}"' in prepare_step
    assert 'echo "PYTHON_BIN=$PYTHON_BIN" >> "$GITHUB_ENV"' in prepare_step
    assert '"$PYTHON_BIN" - <<' in prepare_step
    assert 'echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"' in prepare_step
    assert 'python -m pip install -e "$VLLM_HUST_BENCHMARK_REPO[publish]" jsonschema' not in prepare_step
    assert 'python -m pip install "huggingface_hub>=0.20"' not in prepare_step
    assert 'python -m pip install "numpy<2.0.0" scipy attrs decorator psutil' not in prepare_step
    assert 'python -m pip install -c "$torch_constraints" -r "$VLLM_HUST_REPO/requirements/common.txt"' not in prepare_step
    assert "VLLM_HUST_PYTHON_BIN" in prepare_step


def test_benchmark_verify_uses_resolved_python_not_conda_lookup() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    verify_step = workflow[workflow.index("Verify installation") :]
    verify_step = verify_step[: verify_step.index("- name: Performance gate - fetch Stage 1 baseline")]

    assert "source scripts/hust_ascend_manager_helper.sh" in verify_step
    assert 'PYTHON_BIN="${VLLM_HUST_PYTHON_BIN:-}"' in verify_step
    assert 'PYTHON_BIN="$(hust_resolve_python_bin)"' in verify_step
    assert 'export VLLM_HUST_PYTHON_BIN="$PYTHON_BIN"' in verify_step
    assert 'source scripts/use_single_ascend_env.sh' in verify_step
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
    assert runner_script.count('"${max_model_len_args[@]}"') == 2
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
    assert 'run_env_pip install --no-deps --index-url "$ASCEND_BENCHMARK_TRITON_ASCEND_INDEX_URL" "$triton_ascend_spec"' in install_script
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
