# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import ModelName


@pytest.mark.timeout(1000)
@pytest.mark.model(
    model_name=ModelName.QWEN3_06B,
    quantization=None,
    max_model_len=8192,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    enable_prefix_caching=False,
    max_num_seqs=32,
    tensor_parallel_size=1,
    distributed_executor_backend="mp",
    compilation_config={
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 32, 64],
    },
)
def test_qwen3_mixed_logprobs_widths(vllm_runner) -> None:
    requested_logprobs = [1, 20]
    sampling_params = [
        SamplingParams(max_tokens=5, temperature=0.0, logprobs=num_logprobs) for num_logprobs in requested_logprobs
    ]

    outputs = vllm_runner.model.generate(
        ["Hello, my name is", "The capital of France is"],
        sampling_params,
        use_tqdm=False,
    )

    for output, num_logprobs in zip(outputs, requested_logprobs):
        positions = output.outputs[0].logprobs
        assert positions
        for position in positions:
            assert num_logprobs <= len(position) <= num_logprobs + 1
            assert all(math.isfinite(item.logprob) for item in position.values())
