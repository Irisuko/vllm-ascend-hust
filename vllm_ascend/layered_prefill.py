#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Control data shared by the opt-in layered prefill components."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True)
class LayeredPrefillModelAdapter:
    """How the isolated runner invokes a supported decoder model."""

    layer_call_order: Literal["positions_first", "hidden_first"]
    embedding_method: str = "embed_input_ids"
    is_moe: bool = False


# Keep this list deliberately narrow. Every entry uses a decoder stack whose
# layers return ``(hidden_states, residual)`` and can therefore be resumed at a
# layer boundary without model-specific state other than the MoE cursor.
LAYERED_PREFILL_MODEL_ADAPTERS = MappingProxyType(
    {
        "Qwen3ForCausalLM": LayeredPrefillModelAdapter("positions_first"),
        "Qwen3MoeForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "GptOssForCausalLM": LayeredPrefillModelAdapter(
            "hidden_first", is_moe=True
        ),
        "MixtralForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "Glm4MoeForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "Ernie4_5_MoeForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "DeepseekForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "DeepseekV2ForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "DeepseekV3ForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
    }
)


_MOE_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def get_moe_layer_cursors(
    moe_layer_names: Sequence[str],
    start_layer: int,
    end_layer: int,
) -> tuple[int, ...]:
    """Return the fused-MoE registry cursor at every layer boundary.

    vLLM's fast MoE cold-start path resolves the active expert module through
    a monotonically increasing cursor in ``ForwardContext``. A layered forward
    can start in the middle of the decoder, so its cursor must be restored to
    the number of registered MoE modules preceding that transformer layer.

    The returned tuple has ``end_layer - start_layer + 1`` entries. This also
    handles dense prefixes and models with more than one MoE module per layer.
    """
    if start_layer < 0 or end_layer < start_layer:
        raise ValueError(
            f"Invalid decoder layer range [{start_layer}, {end_layer})."
        )

    layer_indices: list[int] = []
    for name in moe_layer_names:
        match = _MOE_LAYER_PATTERN.search(name)
        if match is None:
            raise ValueError(
                "Layered prefill cannot map MoE module to a transformer "
                f"layer: {name!r}."
            )
        layer_index = int(match.group(1))
        if not start_layer <= layer_index < end_layer:
            raise ValueError(
                f"MoE module {name!r} is outside decoder layer range "
                f"[{start_layer}, {end_layer})."
            )
        layer_indices.append(layer_index)

    if layer_indices != sorted(layer_indices):
        raise ValueError(
            "MoE modules are not registered in transformer execution order."
        )

    cursors: list[int] = []
    moe_index = 0
    for boundary in range(start_layer, end_layer + 1):
        while (
            moe_index < len(layer_indices)
            and layer_indices[moe_index] < boundary
        ):
            moe_index += 1
        cursors.append(moe_index)
    return tuple(cursors)


@dataclass(frozen=True)
class LayeredPrefillRequestData:
    """A prefill token chunk propagating through transformer layer stages."""

    req_id: str
    start_token: int
    num_tokens: int


@dataclass(frozen=True)
class LayeredPrefillMetadata:
    """Per-step control data sent from the scheduler to the NPU runner."""

    stage: int
    num_stages: int
    requests: tuple[LayeredPrefillRequestData, ...]

    @property
    def is_final_stage(self) -> bool:
        return self.stage + 1 == self.num_stages


def get_layer_stage_range(
    num_layers: int,
    num_stages: int,
    stage: int,
) -> tuple[int, int]:
    """Return a balanced, non-overlapping layer range for one stage."""
    if not 0 <= stage < num_stages:
        raise ValueError(f"Invalid stage {stage} for {num_stages} stages.")
    if num_stages > num_layers:
        raise ValueError(
            f"Layered prefill has {num_stages} stages but only {num_layers} layers."
        )
    base, remainder = divmod(num_layers, num_stages)
    start = stage * base + min(stage, remainder)
    stop = start + base + (1 if stage < remainder else 0)
    return start, stop
