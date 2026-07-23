# SPDX-License-Identifier: Apache-2.0
"""Architecture predicates shared by Ascend model compatibility paths."""

_QWEN2_ROPE_ARCHITECTURES = frozenset({"Qwen2ForCausalLM", "SliceGPTQwen2ForCausalLM"})


def uses_qwen2_rope(architectures: list[str] | None) -> bool:
    """Return whether a model architecture uses Qwen2 RoPE handling."""
    return bool(_QWEN2_ROPE_ARCHITECTURES.intersection(architectures or ()))
