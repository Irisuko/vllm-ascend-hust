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
"""Opt-in scheduler that advances prefill chunks one layer stage at a time."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.layered_prefill import (
    LayeredPrefillMetadata,
    LayeredPrefillRequestData,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass(frozen=True)
class _ActiveChunk:
    start_token: int
    num_tokens: int


class LayeredPrefillScheduler(Scheduler):
    """Schedule the same prefill token chunk across consecutive layer stages."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        additional_config = self.vllm_config.additional_config or {}
        self._num_stages = int(
            additional_config.get("layered_prefill_num_stages", 2)
        )
        self._stage = 0
        self._active_chunks: dict[str, _ActiveChunk] = {}
        self._active_token_lists: dict[str, list[int]] = {}

    def schedule(self) -> SchedulerOutput:
        if self._active_chunks:
            # A request can be aborted between stages. The regular finished ID
            # path releases its worker-side intermediate activation.
            self._active_chunks = {
                req_id: chunk
                for req_id, chunk in self._active_chunks.items()
                if req_id in self.requests
            }

        if self._active_chunks and self._pause_state == PauseState.PAUSED_ALL:
            return super().schedule()

        if not self._active_chunks:
            self._stage = 0
            output = super().schedule()
        else:
            output = self._schedule_later_stage()

        if self._active_chunks:
            requests = tuple(
                LayeredPrefillRequestData(
                    req_id=req_id,
                    start_token=chunk.start_token,
                    num_tokens=chunk.num_tokens,
                )
                for req_id, chunk in self._active_chunks.items()
            )
            # Keep SchedulerOutput and its construction untouched on the
            # disabled path. Python's pickle transports this opt-in attribute.
            output.layered_prefill = LayeredPrefillMetadata(  # type: ignore[attr-defined]
                stage=self._stage,
                num_stages=self._num_stages,
                requests=requests,
            )

            if self._stage + 1 == self._num_stages:
                self._active_chunks.clear()
                self._stage = 0
            else:
                self._stage += 1

        return output

    def _schedule_later_stage(self) -> SchedulerOutput:
        active_ids = set(self._active_chunks)
        original_running = self.running
        original_waiting = self.waiting
        original_skipped_waiting = self.skipped_waiting

        active = [request for request in original_running if request.request_id in active_ids]
        decodes = [
            request
            for request in original_running
            if request.request_id not in active_ids
            and request.num_computed_tokens >= request.num_prompt_tokens
        ]
        hidden_prefills = [
            request
            for request in original_running
            if request.request_id not in active_ids
            and request.num_computed_tokens < request.num_prompt_tokens
        ]

        # Active chunks are placed first so a transient decode batch cannot
        # shrink them. Temporarily cap each request at its recorded chunk end
        # so freed budget cannot expand it either. The worker already cached
        # the complete token list when the request first entered the batch.
        self.running = active + decodes
        self.waiting = create_request_queue(self.policy)
        self.skipped_waiting = create_request_queue(self.policy)
        self._cap_active_token_lists(active)
        try:
            output = super().schedule()
            temporary_waiting = self.waiting
            temporary_skipped = self.skipped_waiting
            scheduled_running = self.running
        finally:
            self._restore_active_token_lists()
            self.waiting = original_waiting
            self.skipped_waiting = original_skipped_waiting

        # Preserve any preemptions the base scheduler performed.
        for request in reversed(list(temporary_waiting)):
            self.waiting.prepend_request(request)
        for request in reversed(list(temporary_skipped)):
            self.skipped_waiting.prepend_request(request)
        surviving_ids = {id(request) for request in scheduled_running}
        hidden_ids = {id(request) for request in hidden_prefills}
        self.running = [
            request
            for request in original_running
            if id(request) in surviving_ids or id(request) in hidden_ids
        ]

        for req_id, chunk in self._active_chunks.items():
            actual = output.num_scheduled_tokens.get(req_id)
            if actual != chunk.num_tokens:
                raise RuntimeError(
                    "Layered prefill could not reschedule the complete active "
                    f"chunk for request {req_id!r}: expected {chunk.num_tokens}, "
                    f"scheduled {actual}."
                )
        return output

    def _cap_active_token_lists(self, active: list["Request"]) -> None:
        for request in active:
            chunk = self._active_chunks[request.request_id]
            chunk_end = chunk.start_token + chunk.num_tokens
            if request.num_tokens > chunk_end:
                self._active_token_lists[request.request_id] = (
                    request._all_token_ids
                )
                request._all_token_ids = request._all_token_ids[:chunk_end]

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Advance a prefill chunk's logical tokens only at its final stage."""
        # Base schedule() invokes this method before returning, while later
        # stages may still have their temporary token-list caps installed.
        self._restore_active_token_lists()
        if not self._active_chunks:
            for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
                request = self.requests[req_id]
                if request.num_computed_tokens < request.num_prompt_tokens:
                    self._active_chunks[req_id] = _ActiveChunk(
                        start_token=request.num_computed_tokens,
                        num_tokens=num_tokens,
                    )

        if not self._active_chunks or self._stage + 1 == self._num_stages:
            super()._update_after_schedule(scheduler_output)
            return

        active_tokens = {
            req_id: scheduler_output.num_scheduled_tokens.pop(req_id)
            for req_id in self._active_chunks
            if req_id in scheduler_output.num_scheduled_tokens
        }
        hidden_token_count = sum(active_tokens.values())
        scheduler_output.total_num_scheduled_tokens -= hidden_token_count
        try:
            # Decode requests retain all regular scheduler bookkeeping.
            super()._update_after_schedule(scheduler_output)
        finally:
            scheduler_output.num_scheduled_tokens.update(active_tokens)
            scheduler_output.total_num_scheduled_tokens += hidden_token_count
            self.prev_step_scheduled_req_ids.update(active_tokens)

    def _restore_active_token_lists(self) -> None:
        for req_id, token_ids in self._active_token_lists.items():
            request = self.requests.get(req_id)
            if request is not None:
                request._all_token_ids = token_ids
        self._active_token_lists.clear()
