# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from vllm_ascend.compilation.passes import qknorm_rope_fusion_pass as fusion_module


def test_registers_each_distinct_supported_attention_shape():
    vllm_config = MagicMock()
    vllm_config.model_config.dtype = torch.bfloat16

    target_layer = SimpleNamespace(head_size=128, num_heads=32, num_kv_heads=8)
    duplicate_target_layer = SimpleNamespace(head_size=128, num_heads=32, num_kv_heads=8)
    draft_layer = SimpleNamespace(head_size=128, num_heads=16, num_kv_heads=8)
    unsupported_layer = SimpleNamespace(head_size=64, num_heads=16, num_kv_heads=8)
    attention_layers = {
        "target": target_layer,
        "target_duplicate": duplicate_target_layer,
        "draft": draft_layer,
        "unsupported": unsupported_layer,
    }

    with (
        patch.object(fusion_module, "get_layers_from_vllm_config", return_value=attention_layers),
        patch.object(fusion_module, "PatternMatcherPass"),
        patch.object(fusion_module, "QKNormRopeFusionPattern") as pattern,
        patch.object(fusion_module, "QKNormRopeFusionPatternWithBias") as bias_pattern,
    ):
        fusion_module.QKNormRopeFusionPass(vllm_config)

    expected_calls = [
        call(
            vllm_config=vllm_config,
            head_dim=128,
            num_heads=32,
            num_kv_heads=8,
            eps=1e-6,
        ),
        call(
            vllm_config=vllm_config,
            head_dim=128,
            num_heads=32,
            num_kv_heads=8,
            eps=1e-5,
        ),
        call(
            vllm_config=vllm_config,
            head_dim=128,
            num_heads=16,
            num_kv_heads=8,
            eps=1e-6,
        ),
        call(
            vllm_config=vllm_config,
            head_dim=128,
            num_heads=16,
            num_kv_heads=8,
            eps=1e-5,
        ),
    ]
    assert pattern.call_args_list == expected_calls
    assert bias_pattern.call_args_list == expected_calls
    assert pattern.return_value.register.call_count == len(expected_calls)
    assert bias_pattern.return_value.register.call_count == len(expected_calls)
