# PyramidKV KV cache compression (experimental)

PyramidKV KV cache compression is an opt-in experimental feature for one
validated Ascend configuration. It reduces the number of request-owned KV
cache blocks after a complete prefill. It is disabled unless
`--kv-cache-compression-config` is provided.

The first release keeps token IDs, sequence positions, RoPE, sampling, the KV
cache tensor shape, and the selected attention backend unchanged. During
prefill, every layer computes attention from the original full Q/K/V. With an
unchunked prefill it writes a compact K/V representation immediately. With
chunked prefill, intermediate chunks keep full K/V and the final chunk compacts
the complete prompt only after the whole model forward succeeds. The scheduler
releases only the common tail blocks that no layer needs. During decode, each
layer appends to its own compact physical length while the request continues to
use its semantic sequence length.

## Supported configuration

All of the following are required. Startup fails before formal KV cache
allocation if a requirement is not met:

- Ascend 910B2 with the required torch-npu cache-write and
  fused-infer-attention operators. Eager execution supports CANN 8.5.1 and
  CANN 9.0; graph execution requires CANN 9.0.
- V1 model runner with async scheduling and dual-batch overlap explicitly
  disabled. Execution must be eager (`NONE`), `PIECEWISE`, or
  `FULL_DECODE_ONLY` as described below.
- `LlamaForCausalLM` with 32 layers, 32 query heads, 8 KV heads, head dimension
  128, and BF16 model/KV cache data.
- Dense `AscendAttentionBackend`, one full-attention cache group, and KV block
  size 128.
- TP, PP, PCP, and DCP all equal to 1.
- Chunked prefill is supported on CANN 9.0 with default FIA. It continues to
  fail closed on CANN 8.5.1; unchunked eager execution remains supported there.
- Prefix caching, sliding-window attention, speculative decoding, KV transfer,
  KV offload, KNorm, and quantized KV cache disabled.
- Graph execution uses the default fused-infer-attention (FIA) path and an
  empty `pa_shape_list`. Paged-attention graph shapes, `FULL`, and
  `FULL_AND_PIECEWISE` are not supported.
- Complete or chunked prefill and ordinary one-token decode. A batch may
  contain committed compact decodes, intermediate/final prefill chunks, and
  requests below the compression threshold. Dynamic chunk sizes and a final
  chunk of one token are supported.

Other models, devices, backends, layouts, dtypes, or feature combinations are
not silently downgraded.

## Enabling the provider

Use the same JSON object for the CLI and the Python `EngineArgs`/
`VllmConfig` API:

```bash
VLLM_KNORM_ENABLED=0 VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve /path/to/Meta-Llama-3-8B-Instruct \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --enforce-eager \
  --block-size 128 \
  --no-async-scheduling \
  --enable-chunked-prefill \
  --max-num-batched-tokens 2048 \
  --no-enable-prefix-caching \
  --kv-cache-compression-config '{
    "schema_version": 1,
    "provider": "pyramidkv_ascend",
    "provider_config": {
      "max_capacity_prompt": 512,
      "window_size": 8,
      "kernel_size": 7,
      "pooling": "maxpool",
      "beta": 20,
      "kv_cache_granularity": "kv_head",
      "gqa_score_aggregation": "mean",
      "merge": null
    }
  }'
```

The command above selects eager chunked-prefill execution and therefore
requires CANN 9.0. For unchunked execution, use
`--no-enable-chunked-prefill`; that mode also supports eager CANN 8.5.1. On
CANN 9.0, remove
`--enforce-eager` and add one of the following compilation configurations to
enable graph execution:

```bash
# Piecewise capture around graph breaks; the existing per-layer Python hook is
# still used to prepare PyramidKV decode metadata.
--compilation-config '{
  "cudagraph_mode": "PIECEWISE",
  "cudagraph_capture_sizes": [1, 8, 16, 24, 32]
}'

# Full decode capture with fixed-address, per-layer slot and sequence-length
# metadata. Prefill continues outside the full-decode graph.
--compilation-config '{
  "cudagraph_mode": "FULL_DECODE_ONLY",
  "cudagraph_capture_sizes": [1, 8, 16, 24, 32]
}'
```

Do not combine either graph configuration with `--enforce-eager`. Startup
fails rather than silently falling back if the requested mode, CANN version,
or FIA/PA configuration is unsupported. `FULL_DECODE_ONLY` allocates stable
`int32[32, max_num_seqs]` slot-mapping and physical-length buffers. Each layer
uses its own row during replay, while padding rows use an invalid slot and
cannot write a real KV block.

`max_capacity_prompt` must be greater than `window_size`; `kernel_size` must be
positive and odd; `beta` must be positive. The other fields are fixed to the
values shown above in the first release. Unknown fields and invalid values are
errors.

The equivalent Python configuration uses the same typed object:

```python
from vllm.config import KVCacheCompressionConfig
from vllm.engine.arg_utils import EngineArgs

engine_args = EngineArgs(
    model="/path/to/Meta-Llama-3-8B-Instruct",
    kv_cache_compression_config=KVCacheCompressionConfig(
        provider="pyramidkv_ascend",
        provider_config={
            "max_capacity_prompt": 512,
            "window_size": 8,
            "kernel_size": 7,
            "pooling": "maxpool",
            "beta": 20,
            "kv_cache_granularity": "kv_head",
            "gqa_score_aggregation": "mean",
            "merge": None,
        },
    ),
)
```

The provider is implemented in vLLM Ascend and uses the PyTorch and torch-npu
packages already required by vLLM Ascend. There is no additional optional
package to install, and `KVCache-Factory` is not a runtime dependency. Keep the
vLLM-HUST and vLLM-Ascend-HUST revisions matched because their compression
schema and scheduler/worker transaction are versioned together.

At startup, logs include the provider, schema, actual platform/backend/model/
dtype/cache layout/block size, compatibility result, and complete provider
configuration. They also include whether chunked prefill is enabled and its
query-tail staging bound. After a successful final prefill chunk, the scheduler
logs the semantic and physical token lengths and released block IDs.

## Capacity and memory semantics

Token selection is independent per KV head. GQA query-head scores are averaged
within each KV-head group; scores use an fp32 softmax converted back to the
query dtype before max pooling and top-k.
The recent window remains in original order.

All full-attention layers in the first release share one request-level block
table. The scheduler therefore retains
`ceil(max(per_layer_physical_length) / 128)` blocks. Layers with shorter compact
lengths have unused slots inside those shared retained blocks. Do not estimate
device-memory savings by summing theoretical per-layer capacities, and do not
assume a performance or quality improvement without workload-specific
measurements.

A batch made entirely of prompts at or below the compression threshold uses
the original attention and cache-write path and does not emit a compression
plan. In `FULL_DECODE_ONLY`, the runner still computes an exact read-only slot
view for those requests so replay cannot use stale CPU slot metadata.

For a compressible chunked request, intermediate chunks write their full K/V
to the semantic cache positions and emit no compression plan. The provider
retains only the final `window_size` query vectors for each layer. With the
default Llama-3 shape this committed query tail is about 2 MiB per request.
Step-local transactional staging can coexist with the committed tail, giving a
worst-case bound of about 4 MiB per active request or 128 MiB at
`max_num_seqs=32`. The chunk end, block table, and query tail are committed only
after all 32 attention layers and the model forward succeed.

The final chunk reconstructs the complete key from paged cache plus the current
chunk and computes the same per-head indices as the unchunked path. It first
finishes attention with full history, then compacts the complete paged K/V in
place and emits the existing schema-v1 plan. No partial-prefill plan is sent to
the scheduler, and no request-long dense K/V copy is retained.

For a mixed prefill/decode batch, compact decodes use their layer-specific
physical slots and lengths. A new prefill remains complete in paged K/V while
the existing `PrefillCacheHit` attention runs. Only after the whole model
forward succeeds does the provider compact that prefill in place and emit its
plan. A decode block table may grow only by the exact monotonic tail allocation
required for its next physical token; shrinking, reordering, or adding excess
blocks is an error before the cache write.

For `FULL_DECODE_ONLY`, only all-decode scheduler steps enter the full graph.
Any semantic prefill, including a one-token final chunk, disables `FULL` for
that step and uses the dynamic/PIECEWISE path. Mixed prefill/decode steps remain
outside that graph. Once the plan receives its core commit acknowledgement,
ordinary decode may resume `FULL` replay. The request state advances only after
the complete model forward succeeds; graph preparation or execution failures
do not acknowledge a prefill chunk or decode step.

## Failure and rollback behavior

Compatibility and configuration failures occur before formal KV cache
allocation. Runtime state, block generation, full-layer completion, and the
scheduler commit acknowledgement are validated before tail blocks are freed or
compact decode begins. There is no CUDA, Triton, CPU, or ordinary-cache fallback
after a request starts compacting; a partial or stale transaction raises an
error instead of continuing with ambiguous cache contents.

To disable the feature, remove `--kv-cache-compression-config` (or set
`kv_cache_compression_config=None` in Python). The provider is then not
resolved, instantiated, or attached to attention metadata. No environment or
dependency change is required for rollback.

## Validation guidance

Compare three separately recorded runs: the original code, the new code with
compression disabled, and the new code with compression enabled. Fix the model,
backend, dtype, parallel settings, prompt/output lengths, seeds, concurrency,
warm-up, and repetition count. Record KV/device memory, TTFT, TPOT, throughput,
end-to-end latency, and a long-context quality metric. Report distributions,
not only the best run. CUDA results are not evidence for this provider.

The selection semantics were independently implemented from the observable
PyramidKV behavior in `KVCache-Factory` commit
`fc6f8f4c3d8ca7a1849a2ef67ff5fca8d285a6f0` (MIT). No Hugging Face monkey
patch, CUDA/Triton implementation, or source copy from that repository is used.
