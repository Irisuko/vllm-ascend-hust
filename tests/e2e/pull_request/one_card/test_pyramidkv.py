# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import os
from pathlib import Path

import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, wait_until_npu_memory_free

DEFAULT_LLAMA_MODEL = "LLM-Research/Meta-Llama-3-8B-Instruct"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-14B-Instruct"
LONG_PROMPT = (
    "A coastal research station records the color of the sky every morning. "
    "The observer notes cloud cover, humidity, wind direction, and the angle "
    "of sunlight before writing a short explanation. "
) * 12 + ("Based on these notes, explain why a clear daytime sky usually appears blue in one concise sentence.")
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


def _require_complete_model(model: Path, label: str) -> None:
    if not model.is_dir():
        pytest.fail(f"local PyramidKV {label} model is unavailable: {model}")
    index_file = model / "model.safetensors.index.json"
    required_files = [model / "config.json", model / "tokenizer.json", index_file]
    if any(not path.is_file() for path in required_files):
        pytest.fail(f"local PyramidKV {label} model is incomplete: {model}")
    weight_map = json.loads(index_file.read_text())["weight_map"]
    missing_weights = sorted({filename for filename in weight_map.values() if not (model / filename).is_file()})
    if missing_weights:
        pytest.fail(f"local PyramidKV {label} model has missing weights: {missing_weights}")


def _resolve_model(env_name: str, default_model: str) -> str:
    model = os.getenv(env_name, default_model)
    local_model = Path(model)
    if local_model.is_absolute() or local_model.exists():
        _require_complete_model(local_model, local_model.name)
    return model


def _run_pyramidkv_smoke(
    model: str,
    *,
    enforce_eager: bool,
    enable_chunked_prefill: bool,
    enable_prefix_caching: bool,
    async_scheduling: bool,
) -> None:
    server_port = get_open_port()

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
        "--max-num-batched-tokens",
        "256" if enable_chunked_prefill else "1024",
        "--gpu-memory-utilization",
        "0.8",
        "--block-size",
        "128",
        ("--enable-chunked-prefill" if enable_chunked_prefill else "--no-enable-chunked-prefill"),
        ("--enable-prefix-caching" if enable_prefix_caching else "--no-enable-prefix-caching"),
        "--async-scheduling" if async_scheduling else "--no-async-scheduling",
        "--generation-config",
        "vllm",
        "--port",
        str(server_port),
        "--kv-cache-compression-config",
        PROVIDER_CONFIG,
    ]
    if enforce_eager:
        server_args.append("--enforce-eager")
    else:
        server_args.extend(
            [
                "--compilation-config",
                json.dumps(
                    {
                        "cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [1, 2],
                    }
                ),
            ]
        )
    env = {
        "VLLM_KNORM_ENABLED": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
    }

    with RemoteOpenAIServer(
        model,
        server_args,
        server_host="127.0.0.1",
        server_port=server_port,
        env_dict=env,
        seed=0,
        auto_port=False,
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
            assert first.logprobs.token_logprobs == pytest.approx(
                second.logprobs.token_logprobs,
                abs=5e-2,
            )


@wait_until_npu_memory_free()
def test_pyramidkv_llama_full_prefill_decode_batch_and_repeat() -> None:
    model = _resolve_model("PYRAMIDKV_LLAMA_MODEL", DEFAULT_LLAMA_MODEL)
    _run_pyramidkv_smoke(
        model,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
    )


@wait_until_npu_memory_free()
def test_pyramidkv_llama_full_graph_chunked_prefix_async() -> None:
    model = _resolve_model("PYRAMIDKV_LLAMA_MODEL", DEFAULT_LLAMA_MODEL)
    _run_pyramidkv_smoke(
        model,
        enforce_eager=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        async_scheduling=True,
    )


@wait_until_npu_memory_free()
def test_pyramidkv_qwen_full_prefill_decode_batch_and_repeat() -> None:
    model = _resolve_model("PYRAMIDKV_QWEN_MODEL", DEFAULT_QWEN_MODEL)
    _run_pyramidkv_smoke(
        model,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        async_scheduling=False,
    )


@wait_until_npu_memory_free()
def test_pyramidkv_qwen_full_graph_chunked_prefix_async() -> None:
    model = _resolve_model("PYRAMIDKV_QWEN_MODEL", DEFAULT_QWEN_MODEL)
    _run_pyramidkv_smoke(
        model,
        enforce_eager=False,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        async_scheduling=True,
    )
