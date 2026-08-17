# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from pathlib import Path

import regex as re
import yaml

REPO_ROOT = Path(__file__).parents[2]


def test_pyramidkv_uses_pinned_hust_core_in_github_hosted_ci() -> None:
    core_commit = (REPO_ROOT / ".github/vllm-hust-pyramidkv.commit").read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{40}", core_commit)

    pr_workflow = (REPO_ROOT / ".github/workflows/pr_test.yaml").read_text()
    assert "run-pyramidkv-selected-tests:" not in pr_workflow
    assert "vllm-hust-pyramidkv.commit" in pr_workflow
    assert "Checkout verified vLLM for mypy" in pr_workflow
    assert "repository: vllm-project/vllm" in pr_workflow
    assert "Checkout paired vLLM-HUST for PyramidKV mypy" in pr_workflow
    assert "Run paired-Core mypy for PyramidKV changes" in pr_workflow
    assert "changed_python_files" in pr_workflow
    assert ("!contains(needs.lint-and-select-tests.outputs.matched_modules, 'kv_cache_compression')") in pr_workflow
    assert pr_workflow.index("Select tests based on changed files") < pr_workflow.index(
        "Checkout paired vLLM-HUST for PyramidKV mypy"
    )
    assert "validate-hust-dual-editable:" in pr_workflow
    assert (
        "needs.lint-and-select-tests.outputs.packaging_changed == 'true' || "
        "contains(needs.lint-and-select-tests.outputs.matched_modules, "
        "'kv_cache_compression')"
    ) in pr_workflow


def test_pyramidkv_selective_test_module_is_complete() -> None:
    config_path = REPO_ROOT / ".github/workflows/scripts/test_config.yaml"
    modules, metadata = list(yaml.safe_load_all(config_path.read_text()))
    module = next(item for item in modules if item["name"] == "kv_cache_compression")

    assert module["optional"] is True
    assert module["tests"] == [
        "tests/ut/kv_cache_compression",
        "tests/ut/attention/test_attention_utils.py",
        "tests/ut/test_pyramidkv_ci_wiring.py",
        "tests/e2e/pull_request/one_card/test_pyramidkv.py",
    ]
    estimated_times = metadata["estimated_times"]
    assert estimated_times["tests/ut/kv_cache_compression/a2/test_pyramidkv_npu.py"] == 120
    assert estimated_times["tests/e2e/pull_request/one_card/test_pyramidkv.py"] == 1200


def test_pyramidkv_e2e_matrix_cannot_silently_skip_models() -> None:
    e2e_test = (REPO_ROOT / "tests/e2e/pull_request/one_card/test_pyramidkv.py").read_text()
    expected_tests = (
        "test_pyramidkv_llama_full_prefill_decode_batch_and_repeat",
        "test_pyramidkv_llama_full_graph_chunked_prefix_async",
        "test_pyramidkv_qwen_full_prefill_decode_batch_and_repeat",
        "test_pyramidkv_qwen_full_graph_chunked_prefix_async",
    )
    assert all(f"def {test_name}" in e2e_test for test_name in expected_tests)
    assert "pytest.skip" not in e2e_test
    assert '"cudagraph_mode": "FULL_DECODE_ONLY"' in e2e_test
