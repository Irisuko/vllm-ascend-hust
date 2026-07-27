# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import os
from pathlib import Path

import pytest

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free


DEFAULT_MODEL = Path(
    "/workspace/KVCache-Factory/models/Meta-Llama-3-8B-Instruct"
)
LONG_PROMPT = (
    "A coastal research station records the color of the sky every morning. "
    "The observer notes cloud cover, humidity, wind direction, and the angle "
    "of sunlight before writing a short explanation. "
) * 12 + (
    "Based on these notes, explain why a clear daytime sky usually appears "
    "blue in one concise sentence."
)
SHORT_PROMPT = "Explain why a clear daytime sky appears blue."
PROVIDER_CONFIG = json.dumps(
    {
        "schema_version": 1,
        "provider": "pyramidkv_ascend",
        "provider_config": {
            "max_capacity_prompt": 128,
            "window_size": 8,
            "kernel_size": 7,
            "pooling": "maxpool",
            "beta": 20,
            "kv_cache_granularity": "kv_head",
            "gqa_score_aggregation": "mean",
            "merge": None,
        },
    }
)


@wait_until_npu_memory_free()
def test_pyramidkv_full_prefill_decode_batch_and_repeat() -> None:
    model = Path(os.getenv("PYRAMIDKV_LLAMA_MODEL", DEFAULT_MODEL))
    if not model.is_dir():
        pytest.skip(f"local PyramidKV Llama model is unavailable: {model}")

    server_args = [
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "2",
        "--gpu-memory-utilization",
        "0.5",
        "--enforce-eager",
        "--block-size",
        "128",
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--generation-config",
        "vllm",
        "--kv-cache-compression-config",
        PROVIDER_CONFIG,
    ]
    env = {
        "VLLM_KNORM_ENABLED": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
    }

    with RemoteOpenAIServer(
        str(model), server_args, env_dict=env, seed=0
    ) as server:
        client = server.get_client()
        batched = client.completions.create(
            model=str(model),
            prompt=[LONG_PROMPT, SHORT_PROMPT],
            max_tokens=16,
            temperature=0,
            seed=0,
            logprobs=1,
        )
        assert len(batched.choices) == 2
        assert all(choice.finish_reason == "length" for choice in batched.choices)
        assert all(len(choice.logprobs.tokens) == 16 for choice in batched.choices)

        repeated = client.completions.create(
            model=str(model),
            prompt=[LONG_PROMPT, SHORT_PROMPT],
            max_tokens=16,
            temperature=0,
            seed=0,
            logprobs=1,
        )
        assert len(repeated.choices) == 2
        for first, second in zip(batched.choices, repeated.choices):
            assert first.logprobs.tokens == second.logprobs.tokens
            assert (
                first.logprobs.token_logprobs
                == second.logprobs.token_logprobs
            )
