from contextlib import ExitStack
from unittest import mock

import torch
from vllm.config import VllmConfig

from vllm_ascend.compilation.passes import allreduce_rmsnorm_fusion_pass as allreduce_pass
from vllm_ascend.compilation.passes import norm_quant_fusion_pass as norm_quant_pass
from vllm_ascend.compilation.passes import sequence_parallelism as sequence_parallel_pass
from vllm_ascend.compilation.passes import sequence_parallelism_moe as sequence_parallel_moe_pass


def _config() -> mock.MagicMock:
    config = mock.MagicMock(spec=VllmConfig)
    config.compilation_config = mock.MagicMock()
    config.compilation_config.splitting_ops = []
    config.compilation_config.use_inductor_graph_partition = False
    config.compilation_config.pass_config = mock.MagicMock()
    config.model_config = mock.MagicMock()
    config.model_config.dtype = torch.float16
    config.device_config = mock.MagicMock()
    config.device_config.device = "npu"
    return config


def test_norm_quant_pass_skips_add_rms_norm_patterns_when_op_is_unavailable():
    custom_patterns = (
        "AddRMSNormQuantPattern",
        "AddRMSNormQuantSPPattern",
        "AddRMSNormQuantPatternWithBias",
        "AddRMSNormQuantSPPatternWithBias",
        "AddRMSNormDynamicQuantPatternWithBias",
        "AddRMSNormDynamicQuantSPPatternWithBias",
    )
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(
                norm_quant_pass,
                "enable_add_rms_norm_bias_custom_op",
                return_value=False,
            )
        )
        stack.enter_context(
            mock.patch.object(
                norm_quant_pass,
                "is_add_rms_norm_dynamic_mx_quant_fusion_available",
                return_value=False,
            )
        )
        stack.enter_context(
            mock.patch.object(
                norm_quant_pass,
                "is_rms_norm_dynamic_mx_quant_fusion_available",
                return_value=False,
            )
        )
        stack.enter_context(mock.patch.object(norm_quant_pass.BasePattern, "register"))
        pattern_mocks = [stack.enter_context(mock.patch.object(norm_quant_pass, name)) for name in custom_patterns]
        norm_quant_pass.AddRMSNormQuantFusionPass(_config())

    for pattern_mock in pattern_mocks:
        pattern_mock.assert_not_called()


def test_allreduce_pass_skips_patterns_when_op_is_unavailable():
    with (
        mock.patch.object(allreduce_pass, "enable_add_rms_norm_bias_custom_op", return_value=False),
        mock.patch.object(allreduce_pass, "MiddleLayerMatmulAllReduceAddRMSNormPattern") as middle_pattern,
        mock.patch.object(allreduce_pass, "LastLayerMatmulAllReduceAddRMSNormPattern") as last_pattern,
    ):
        allreduce_pass.MatmulAllReduceAddRMSNormPass(_config())

    middle_pattern.assert_not_called()
    last_pattern.assert_not_called()


def test_sequence_parallel_pass_skips_patterns_when_op_is_unavailable():
    with (
        mock.patch.object(sequence_parallel_pass, "enable_add_rms_norm_bias_custom_op", return_value=False),
        mock.patch.object(sequence_parallel_pass, "get_sp_min_token_num", return_value=1),
        mock.patch.object(sequence_parallel_pass, "NoOpEliminationPass"),
        mock.patch.object(sequence_parallel_pass, "MiddleAllReduceRMSNormPattern") as middle_pattern,
        mock.patch.object(sequence_parallel_pass, "LastAllReduceRMSNormPattern") as last_pattern,
        mock.patch.object(sequence_parallel_pass, "Qwen3VLMiddleAllReduceRMSNormPattern") as qwen3_pattern,
    ):
        sequence_parallel_pass.SequenceParallelismPass(_config())

    middle_pattern.assert_not_called()
    last_pattern.assert_not_called()
    qwen3_pattern.assert_not_called()


def test_sequence_parallel_moe_pass_keeps_only_non_rmsnorm_pattern():
    with (
        mock.patch.object(sequence_parallel_moe_pass, "enable_add_rms_norm_bias_custom_op", return_value=False),
        mock.patch.object(sequence_parallel_moe_pass, "get_sp_min_token_num", return_value=1),
        mock.patch.object(sequence_parallel_moe_pass, "MiddleLayerAllgatherAddRMSNormPattern") as middle_pattern,
        mock.patch.object(sequence_parallel_moe_pass, "LastLayerAllgatherRMSNormPattern") as last_pattern,
        mock.patch.object(sequence_parallel_moe_pass, "Qwen3VLMiddleLayerAllgatherAddRMSNormPattern") as qwen3_pattern,
        mock.patch.object(sequence_parallel_moe_pass, "AllGatherChunkNoOpPattern") as noop_pattern,
    ):
        sequence_parallel_moe_pass.SequenceParallelismMoePass(_config())

    middle_pattern.assert_not_called()
    last_pattern.assert_not_called()
    qwen3_pattern.assert_not_called()
    assert noop_pattern.return_value.register.call_count == 1
