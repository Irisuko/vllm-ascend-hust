# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import runpy
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_MODULE = REPO_ROOT / "vllm_ascend" / "compilation" / "passes" / "qknorm_rope_pattern_specs.py"
FUSION_PASS = REPO_ROOT / "vllm_ascend" / "compilation" / "passes" / "qknorm_rope_fusion_pass.py"


def _spec_builder():
    return runpy.run_path(str(SPEC_MODULE))["iter_qknorm_rope_pattern_specs"]


def test_mixed_attention_shapes_expand_to_distinct_pattern_specs() -> None:
    target = SimpleNamespace(head_size=128, num_heads=32, num_kv_heads=8)
    duplicate_target = SimpleNamespace(head_size=128, num_heads=32, num_kv_heads=8)
    draft = SimpleNamespace(head_size=128, num_heads=16, num_kv_heads=8)
    unsupported = SimpleNamespace(head_size=64, num_heads=16, num_kv_heads=8)

    assert _spec_builder()([target, duplicate_target, draft, unsupported]) == (
        (128, 32, 8, 1e-6),
        (128, 32, 8, 1e-5),
        (128, 16, 8, 1e-6),
        (128, 16, 8, 1e-5),
    )


def test_pattern_specs_preserve_first_seen_order() -> None:
    draft = SimpleNamespace(head_size=128, num_heads=16, num_kv_heads=8)
    target = SimpleNamespace(head_size=128, num_heads=32, num_kv_heads=8)

    assert _spec_builder()([draft, target]) == (
        (128, 16, 8, 1e-6),
        (128, 16, 8, 1e-5),
        (128, 32, 8, 1e-6),
        (128, 32, 8, 1e-5),
    )


def test_fusion_pass_uses_host_validated_pattern_specs() -> None:
    source = FUSION_PASS.read_text(encoding="utf-8")

    assert "iter_qknorm_rope_pattern_specs(attn_layers.values())" in source
    assert "for head_dim, num_heads, num_kv_heads, epsilon in" in source
