# Layered Prefill

Layered prefill is an opt-in scheduling mode that propagates each prefill
token chunk through consecutive groups of transformer layers. Decode requests
in the same batch continue to traverse the complete model on every engine
iteration. This reduces the length of the individual prefill forward passes
that can block decode work.

## Usage

For an OpenAI-compatible server:

```bash
vllm serve Qwen/Qwen3-8B \
  --additional-config '{"enable_layered_prefill":true,"layered_prefill_num_stages":4}'
```

The equivalent offline configuration is:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-8B",
    additional_config={
        "enable_layered_prefill": True,
        "layered_prefill_num_stages": 4,
    },
)
```

For a MoE model, the switch is unchanged. For example:

```bash
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --additional-config '{"enable_layered_prefill":true,"layered_prefill_num_stages":4}'
```

Layer groups are balanced and contiguous. For example, 32 transformer layers
and four stages produce `[0, 8)`, `[8, 16)`, `[16, 24)`, and `[24, 32)`.

## Execution semantics

- A prefill chunk uses the same token positions and KV-cache slots in every
  stage. Its logical computed-token count advances only after the final stage.
- Intermediate hidden and residual states remain on the NPU between stages.
- Decode requests execute all transformer layers and sample normally during
  every stage.
- Prefill sampling is discarded until the chunk reaches its final layer stage.
- Prefix caching and chunked prefill remain available. Layered execution
  applies only to the uncached chunk selected by the scheduler.
- For MoE models, the router and selected experts still execute normally in
  each transformer layer. Before every partial layer range, the runner restores
  vLLM's fused-MoE registry cursor to the corresponding layer boundary. The
  cursor table is built once during model loading, so no layer-name scan is
  added to the per-layer execution path.

## Compatibility

The implementation supports text-only `Qwen3ForCausalLM` and the following
standard decoder-only MoE architectures using the native vLLM model
implementation and the Ascend V1 model runner:

- `Qwen3MoeForCausalLM`
- `GptOssForCausalLM`
- `MixtralForCausalLM`
- `Glm4MoeForCausalLM`
- `Ernie4_5_MoeForCausalLM`
- `DeepseekForCausalLM`, `DeepseekV2ForCausalLM`, and
  `DeepseekV3ForCausalLM`

Tensor parallelism, expert parallelism, quantization, prefix caching, and
chunked prefill can be used. DeepSeek checkpoints configured with
`llama_4_scaling` are rejected because that model-specific per-forward tensor
is not yet preserved by layered execution.

Hybrid or sparse-attention MoE architectures such as `Qwen3NextForCausalLM`
and `DeepseekV32ForCausalLM` are intentionally not accepted: their decoder
layers carry additional state that cannot be resumed using only hidden and
residual tensors.

The following combinations are rejected during startup: pipeline, data, or
context parallelism; DBO/microbatching; speculative decoding; non-FCFS
scheduling; LoRA; multimodal models; KV/EC connectors; custom,
balance, recompute, dynamic-batch, or profiling-chunk schedulers; routed-expert
output; Xlite/ACL graph execution; dynamic EPLB; multi-stream shared-expert or
gate overlap; layer sharding; sequence parallelism; FlashComm1/2; Ascend 310P;
and edge/cloud layer splitting.

Layered prefill forces synchronous scheduling and eager execution, and disables
cascade attention because the executed layer range changes between stages.

## Disabled-path isolation

`enable_layered_prefill` defaults to `false`. When disabled, vLLM Ascend keeps
the original upstream scheduler and the original `NPUModelRunner`; model
forward methods are not wrapped. The regular scheduler, attention, and model
forward hot paths contain no layered-prefill condition.
