# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections.abc import Iterable
from typing import Protocol


class AttentionShape(Protocol):
    head_size: int
    num_heads: int
    num_kv_heads: int


PatternSpec = tuple[int, int, int, float]


def iter_qknorm_rope_pattern_specs(
    attention_layers: Iterable[AttentionShape],
) -> tuple[PatternSpec, ...]:
    """Return first-seen, supported QKNorm/RoPE pattern specifications."""
    seen_shapes: set[tuple[int, int, int]] = set()
    pattern_specs: list[PatternSpec] = []

    for layer in attention_layers:
        if layer.head_size != 128:
            continue

        shape = (layer.head_size, layer.num_heads, layer.num_kv_heads)
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)

        for epsilon in (1e-6, 1e-5):
            pattern_specs.append((*shape, epsilon))

    return tuple(pattern_specs)
