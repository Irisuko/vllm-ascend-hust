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
"""Dedicated NPU V1 model runner for opt-in layered prefill."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Any

import numpy as np
import torch
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.v1.attention.backends.utils import (
    reorder_batch_to_split_decodes_and_prefills,
)
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.layered_prefill import (
    LAYERED_PREFILL_MODEL_ADAPTERS,
    LayeredPrefillMetadata,
    get_layer_stage_range,
    get_moe_layer_cursors,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class LayeredPrefillNPUModelRunner(NPUModelRunner):
    """Run prefill through one layer group while decode traverses all layers."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._layered_metadata: LayeredPrefillMetadata | None = None
        self._layered_decode_reqs = 0
        self._layered_decode_tokens = 0
        self._layered_decode_attn_metadata: dict[str, Any] | None = None
        self._layered_intermediates: dict[
            str, tuple[torch.Tensor, torch.Tensor | None]
        ] = {}
        self._layered_original_forward: Callable[..., Any] | None = None
        self._layer_call_order = "positions_first"
        self._embedding_method = "embed_input_ids"
        self._moe_layer_registry: list[str] | None = None
        self._moe_layer_names: tuple[str, ...] = ()
        self._moe_layer_cursors: tuple[int, ...] | None = None

    def load_model(self) -> None:
        super().load_model()

        architecture = self.model_config.architecture
        adapter = LAYERED_PREFILL_MODEL_ADAPTERS.get(architecture)
        if adapter is None:
            raise ValueError(
                "Unsupported model architecture for layered prefill: "
                f"{architecture!r}"
            )
        self._layer_call_order = adapter.layer_call_order
        self._embedding_method = adapter.embedding_method

        outer_model = self.get_model()
        inner_model = getattr(outer_model, "model", None)
        if inner_model is None or not all(
            hasattr(inner_model, name)
            for name in (
                "layers",
                "norm",
                "start_layer",
                "end_layer",
            )
        ):
            raise TypeError(
                "Layered prefill requires a causal LM with a decoder in `.model`."
            )
        if getattr(inner_model, "edge_cloud_split_config", None) is not None:
            raise ValueError(
                "Layered prefill cannot be combined with edge/cloud layer splitting."
            )
        if self.use_aux_hidden_state_outputs:
            raise ValueError(
                "Layered prefill cannot be combined with auxiliary hidden-state "
                "outputs."
            )
        if not hasattr(inner_model, self._embedding_method):
            raise TypeError(
                "Layered prefill model adapter requires decoder method "
                f"{self._embedding_method!r}."
            )
        if architecture in {
            "DeepseekForCausalLM",
            "DeepseekV2ForCausalLM",
            "DeepseekV3ForCausalLM",
        } and getattr(inner_model.config, "llama_4_scaling", None) is not None:
            raise ValueError(
                "Layered prefill does not yet support DeepSeek models with "
                "llama_4_scaling."
            )

        num_layers = inner_model.end_layer - inner_model.start_layer
        num_stages = self.ascend_config.layered_prefill_num_stages
        get_layer_stage_range(num_layers, num_stages, 0)

        if adapter.is_moe:
            static_moe_registry = (
                self.vllm_config.compilation_config.static_all_moe_layers
            )
            static_moe_layers = tuple(static_moe_registry)
            if static_moe_layers:
                self._moe_layer_registry = static_moe_registry
                self._moe_layer_names = static_moe_layers
                self._moe_layer_cursors = get_moe_layer_cursors(
                    static_moe_layers,
                    inner_model.start_layer,
                    inner_model.end_layer,
                )

        self._layered_original_forward = inner_model.forward
        runner = self

        def layered_forward(
            model_self,
            input_ids,
            positions,
            intermediate_tensors=None,
            inputs_embeds=None,
        ):
            metadata = runner._layered_metadata
            if metadata is None:
                original_forward = runner._layered_original_forward
                assert original_forward is not None
                return original_forward(
                    input_ids,
                    positions,
                    intermediate_tensors,
                    inputs_embeds,
                )
            return runner._layered_forward(
                model_self,
                input_ids,
                positions,
                inputs_embeds,
                metadata,
            )

        # Patch only this runner's model instance. The upstream class and the
        # regular NPU runner are never modified.
        inner_model.forward = MethodType(layered_forward, inner_model)
        logger.info_once(
            "Layered prefill enabled with %d stages for %s.",
            num_stages,
            architecture or type(outer_model).__name__,
        )

    def execute_model(self, scheduler_output, intermediate_tensors=None):
        self._layered_metadata = getattr(
            scheduler_output,
            "layered_prefill",
            None,
        )
        for req_id in scheduler_output.finished_req_ids:
            self._layered_intermediates.pop(req_id, None)
        return super().execute_model(scheduler_output, intermediate_tensors)

    def _may_reorder_batch(self, scheduler_output: SchedulerOutput) -> None:
        metadata = getattr(scheduler_output, "layered_prefill", None)
        if metadata is None:
            super()._may_reorder_batch(scheduler_output)
            return

        reorder_batch_to_split_decodes_and_prefills(
            self.input_batch,
            scheduler_output,
            decode_threshold=1,
        )
        layered_ids = {request.req_id for request in metadata.requests}
        req_ids = self.input_batch.req_ids
        first_layered = next(
            (
                index
                for index, req_id in enumerate(req_ids)
                if req_id in layered_ids
            ),
            len(req_ids),
        )
        if any(req_id not in layered_ids for req_id in req_ids[first_layered:]):
            raise RuntimeError(
                "Layered prefill requests must form a contiguous batch suffix."
            )
        self._layered_decode_reqs = first_layered
        self._layered_decode_tokens = sum(
            scheduler_output.num_scheduled_tokens[req_id]
            for req_id in req_ids[:first_layered]
        )

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        result = super()._prepare_inputs(
            scheduler_output,
            num_scheduled_tokens,
        )
        metadata = getattr(scheduler_output, "layered_prefill", None)
        if metadata is not None and not metadata.is_final_stage:
            for request in metadata.requests:
                req_index = self.input_batch.req_id_to_index[request.req_id]
                self.discard_request_mask.np[req_index] = True

            num_reqs = self.input_batch.num_reqs
            discarded = np.nonzero(
                self.discard_request_mask.np[:num_reqs]
            )[0]
            self.num_discarded_requests = len(discarded)
            self.discard_request_indices.np[
                : self.num_discarded_requests
            ] = discarded
            self.discard_request_indices.copy_to_gpu(
                self.num_discarded_requests
            )
            self.discard_request_mask.copy_to_gpu(num_reqs)
        return result

    def _get_prompt_logprobs_dict(self, hidden_states, num_scheduled_tokens):
        metadata = self._layered_metadata
        if metadata is None or metadata.is_final_stage:
            return super()._get_prompt_logprobs_dict(
                hidden_states,
                num_scheduled_tokens,
            )

        layered_ids = {request.req_id for request in metadata.requests}
        filtered_tokens = {
            req_id: num_tokens
            for req_id, num_tokens in num_scheduled_tokens.items()
            if req_id not in layered_ids
        }
        return super()._get_prompt_logprobs_dict(
            hidden_states,
            filtered_tokens,
        )

    def _build_attention_metadata(self, *args, **kwargs):
        full_metadata, spec_metadata = super()._build_attention_metadata(
            *args,
            **kwargs,
        )
        self._layered_decode_attn_metadata = None
        if self._layered_metadata is None or self._layered_decode_tokens == 0:
            return full_metadata, spec_metadata

        # Ascend's current metadata path does not construct u-batch slices for
        # all backends. Since decode requests are reordered to a contiguous
        # prefix, rebuilding that prefix is backend-neutral and stays entirely
        # inside this opt-in runner.
        decode_kwargs = dict(kwargs)
        decode_kwargs.update(
            num_tokens=self._layered_decode_tokens,
            num_tokens_padded=self._layered_decode_tokens,
            num_reqs=self._layered_decode_reqs,
            num_reqs_padded=self._layered_decode_reqs,
            max_query_len=1,
            ubatch_slices=None,
            logits_indices=None,
            use_spec_decode=False,
            num_scheduled_tokens={
                req_id: 1
                for req_id in self.input_batch.req_ids[
                    : self._layered_decode_reqs
                ]
            },
            num_scheduled_tokens_np=np.ones(
                self._layered_decode_reqs,
                dtype=np.int32,
            ),
            cascade_attn_prefix_lens=None,
        )
        decode_metadata, _ = super()._build_attention_metadata(
            *args,
            **decode_kwargs,
        )
        if not isinstance(decode_metadata, dict):
            raise RuntimeError(
                "Layered prefill requires non-microbatched attention metadata."
            )
        self._layered_decode_attn_metadata = decode_metadata
        return full_metadata, spec_metadata

    @contextmanager
    def _decode_forward_context(
        self,
        input_ids: torch.Tensor | None,
    ) -> Iterator[None]:
        assert self._layered_decode_attn_metadata is not None
        with set_ascend_forward_context(
            self._layered_decode_attn_metadata,
            self.vllm_config,
            num_tokens=self._layered_decode_tokens,
            num_actual_tokens=self._layered_decode_tokens,
            model_instance=self.model,
            input_ids=input_ids,
            skip_compiled=True,
            has_sinks=self._has_sinks,
        ):
            yield

    def _call_layer(self, layer, positions, hidden_states, residual):
        if self._layer_call_order == "hidden_first":
            return layer(hidden_states, positions, residual)
        return layer(positions, hidden_states, residual)

    def _embed_input_ids(self, model, input_ids):
        return getattr(model, self._embedding_method)(input_ids)

    def _align_moe_cursor(self, model, layer_index: int) -> None:
        context = get_forward_context()
        context.layer_idx = layer_index
        context.is_first_layer = layer_index == model.start_layer
        if context.all_moe_layers is None:
            return
        context_registry = context.all_moe_layers
        if self._moe_layer_cursors is None:
            context_names = tuple(context_registry)
            self._moe_layer_registry = context_registry
            self._moe_layer_names = context_names
            self._moe_layer_cursors = get_moe_layer_cursors(
                context_names,
                model.start_layer,
                model.end_layer,
            )
        elif context_registry is not self._moe_layer_registry:
            if tuple(context_registry) != self._moe_layer_names:
                raise RuntimeError(
                    "MoE layer registry changed after layered-prefill model "
                    "load."
                )
            self._moe_layer_registry = context_registry
        context.moe_layer_index = self._moe_layer_cursors[
            layer_index - model.start_layer
        ]

    def _run_layers(
        self,
        model,
        start: int,
        stop: int,
        positions,
        hidden_states,
        residual,
    ):
        self._align_moe_cursor(model, start)
        context = get_forward_context()
        for layer_index in range(start, stop):
            context.layer_idx = layer_index
            context.is_first_layer = layer_index == model.start_layer
            hidden_states, residual = self._call_layer(
                model.layers[layer_index],
                positions,
                hidden_states,
                residual,
            )
        if context.all_moe_layers is not None:
            assert self._moe_layer_cursors is not None
            expected_cursor = self._moe_layer_cursors[
                stop - model.start_layer
            ]
            if context.moe_layer_index != expected_cursor:
                raise RuntimeError(
                    "MoE execution order diverged from the layer registry: "
                    f"expected cursor {expected_cursor} after layer {stop}, "
                    f"got {context.moe_layer_index}."
                )
        return hidden_states, residual

    def _layered_forward(
        self,
        model,
        input_ids,
        positions,
        inputs_embeds,
        metadata: LayeredPrefillMetadata,
    ):
        decode_tokens = self._layered_decode_tokens
        req_ids = self.input_batch.req_ids[self._layered_decode_reqs :]
        request_by_id = {
            request.req_id: request for request in metadata.requests
        }
        if set(req_ids) != set(request_by_id):
            raise RuntimeError(
                "Worker and scheduler layered request sets diverged."
            )

        num_layers = model.end_layer - model.start_layer
        relative_start, relative_stop = get_layer_stage_range(
            num_layers,
            metadata.num_stages,
            metadata.stage,
        )
        stage_start = model.start_layer + relative_start
        stage_stop = model.start_layer + relative_stop

        if metadata.stage == 0:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self._embed_input_ids(model, input_ids)
            prefill_hidden = hidden_states[decode_tokens:]
            prefill_residual = None
        else:
            try:
                saved = [
                    self._layered_intermediates[req_id]
                    for req_id in req_ids
                ]
            except KeyError as error:
                raise RuntimeError(
                    "Missing layered prefill intermediate activation."
                ) from error
            prefill_hidden = torch.cat([item[0] for item in saved], dim=0)
            residuals = [item[1] for item in saved]
            if any(residual is None for residual in residuals):
                raise RuntimeError("Missing layered prefill residual state.")
            prefill_residual = torch.cat(  # type: ignore[arg-type]
                residuals,
                dim=0,
            )

            if decode_tokens == 0:
                hidden_states = prefill_hidden[:0]
            elif inputs_embeds is not None:
                hidden_states = inputs_embeds[:decode_tokens]
            else:
                hidden_states = self._embed_input_ids(
                    model,
                    input_ids[:decode_tokens]
                )

        decode_hidden = hidden_states[:decode_tokens]
        decode_positions = positions[..., :decode_tokens]
        decode_input_ids = (
            input_ids[:decode_tokens] if input_ids is not None else None
        )
        decode_residual = None
        if decode_tokens:
            with self._decode_forward_context(decode_input_ids):
                decode_hidden, decode_residual = self._run_layers(
                    model,
                    model.start_layer,
                    stage_start,
                    decode_positions,
                    decode_hidden,
                    decode_residual,
                )

        prefill_positions = positions[..., decode_tokens:]
        if decode_tokens:
            stage_hidden = torch.cat(
                (decode_hidden, prefill_hidden),
                dim=0,
            )
            if decode_residual is None and prefill_residual is None:
                stage_residual = None
            else:
                assert decode_residual is not None
                assert prefill_residual is not None
                stage_residual = torch.cat(
                    (decode_residual, prefill_residual),
                    dim=0,
                )
            stage_positions = positions
        else:
            stage_hidden = prefill_hidden
            stage_residual = prefill_residual
            stage_positions = prefill_positions

        stage_hidden, stage_residual = self._run_layers(
            model,
            stage_start,
            stage_stop,
            stage_positions,
            stage_hidden,
            stage_residual,
        )
        decode_hidden = stage_hidden[:decode_tokens]
        prefill_hidden = stage_hidden[decode_tokens:]
        decode_residual = (
            stage_residual[:decode_tokens]
            if stage_residual is not None
            else None
        )
        prefill_residual = (
            stage_residual[decode_tokens:]
            if stage_residual is not None
            else None
        )

        if not metadata.is_final_stage:
            offset = 0
            for req_id in req_ids:
                num_tokens = request_by_id[req_id].num_tokens
                end = offset + num_tokens
                self._layered_intermediates[req_id] = (
                    prefill_hidden[offset:end].detach().clone(),
                    None
                    if prefill_residual is None
                    else prefill_residual[offset:end].detach().clone(),
                )
                offset = end
            if offset != prefill_hidden.shape[0]:
                raise RuntimeError(
                    "Layered prefill token split does not match the NPU batch."
                )

        if decode_tokens:
            with self._decode_forward_context(decode_input_ids):
                decode_hidden, decode_residual = self._run_layers(
                    model,
                    stage_stop,
                    model.end_layer,
                    decode_positions,
                    decode_hidden,
                    decode_residual,
                )

        if metadata.is_final_stage:
            for req_id in req_ids:
                self._layered_intermediates.pop(req_id, None)
            if decode_tokens:
                hidden_states = torch.cat(
                    (decode_hidden, prefill_hidden),
                    dim=0,
                )
                assert decode_residual is not None
                assert prefill_residual is not None
                residual = torch.cat(
                    (decode_residual, prefill_residual),
                    dim=0,
                )
            else:
                hidden_states = prefill_hidden
                residual = prefill_residual
            hidden_states, _ = model.norm(hidden_states, residual)
            return hidden_states

        if decode_tokens:
            decode_hidden, _ = model.norm(
                decode_hidden,
                decode_residual,
            )
            return torch.cat((decode_hidden, prefill_hidden), dim=0)
        return prefill_hidden
