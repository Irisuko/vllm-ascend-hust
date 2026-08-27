from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def test_pr_smart_ut_is_hosted_cpu_only() -> None:
    text = (WORKFLOWS / "pr_smart_ut.yaml").read_text(encoding="utf-8")
    assert "pull_request:" in text
    assert "      - 'main'" in text
    assert "Restrict PR smart UT to hosted CPU tests" in text
    assert 'hosted_group["runner"] = "ubuntu-latest"' in text
    assert 'hosted_group["hosted_cpu"] = True' in text
    assert "cpu_scope.outputs.test_groups" in text


def test_selected_tests_uses_hosted_container_only_for_explicit_cpu_groups() -> None:
    text = (WORKFLOWS / "_selected_tests.yaml").read_text(encoding="utf-8")
    assert "matrix.group.hosted_cpu == true" in text
    assert "ubuntu:22.04" in text
    assert "matrix.group.hosted_cpu == true && 'https://pypi.org/simple'" in text
    assert 'if [ "${{ matrix.group.hosted_cpu }}" != "true" ]' in text
    assert "--device /dev/davinci1:/dev/davinci0" in text


def test_legacy_npu_pr_and_benchmark_entrypoints_are_removed() -> None:
    legacy_download_workflow = "la" + "bled_download_model_dataset.yaml"
    for name in (
        "ascend-benchmark-leaderboard.yml",
        "pr_test.yaml",
        "pr_e2e_command.yml",
        "pr_nightly_command.yml",
        "labeled_doctest.yaml",
        legacy_download_workflow,
        "schedule_nightly_test_a2.yaml",
        "schedule_nightly_test_a3.yaml",
        "schedule_weekly_test_a2.yaml",
        "schedule_weekly_test_a3.yaml",
        "schedule_vllm_e2e_test.yaml",
        "schedule_update_estimated_times.yaml",
    ):
        assert not (WORKFLOWS / name).exists(), name


def test_retired_hardware_slash_commands_are_not_registered() -> None:
    text = (WORKFLOWS / "slash_command_dispatch.yml").read_text(encoding="utf-8")
    for command in ("e2e", "nightly", "weekly"):
        assert f'"command": "{command}"' not in text
    commands = text.split("commands:", 1)[1].split("permission:", 1)[0]
    assert "rerun" in commands
    assert "cherry-pick" in commands
    assert "revert" in commands


def test_selected_tests_has_no_deleted_benchmark_checkout_or_targets() -> None:
    selected = (WORKFLOWS / "_selected_tests.yaml").read_text(encoding="utf-8")
    config = (WORKFLOWS / "scripts" / "test_config.yaml").read_text(encoding="utf-8")
    assert "vllm-hust-benchmark" not in selected
    for deleted_target in (
        "test_ascend_benchmark_perfgate_workflow_static.py",
        "test_perfgate_summary_conflict.py",
    ):
        assert deleted_target not in config
