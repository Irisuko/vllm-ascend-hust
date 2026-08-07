"""Experimental, fail-fast worker-local mapped KV offload handler.

This module targets vLLM's :class:`OffloadingWorker` contract and keeps one
object responsible for tensors, registrations, and both transfer directions.
It is selected explicitly through
``vllm_ascend.kv_offload.npu.MappedOffloadingSpec``:

* one ``MappedOffloadingHandler`` owns the CPU tensors and registrations;
* CPU-to-NPU transfers always use mapped-host gather;
* NPU-to-CPU transfers always use the normal span-copy primitive;
* unsupported layouts or missing runtime support fail during construction.

The same handler instance is registered for both router directions. There is
no fallback handler and no duplicated CPU-pool ownership.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from typing_extensions import override
from vllm.logger import logger
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from vllm_ascend.custom_op_package import activate_kv_cache_block_gather_runtime

DIRECTION_D2H = 1
MAPPED_GATHER_ALIGNMENT_BYTES = 32
MAPPED_GATHER_ID_BUFFER_CACHE_SIZE = 4
MAPPED_GATHER_MAX_CACHED_IDS = 256 * 1024


def _get_custom_op(name: str):
    """Return an optional in-tree custom op without triggering dispatch."""
    namespace = getattr(torch.ops, "_C_ascend", None)
    return None if namespace is None else getattr(namespace, name, None)


def _resolve_mapped_gather_ops():
    gather_op = _get_custom_op("kv_cache_block_gather")
    register_op = _get_custom_op("register_kv_cache_block_gather_host_pool")
    unregister_op = _get_custom_op("unregister_kv_cache_block_gather_host_pool")
    runtime_op = _get_custom_op("has_kv_cache_block_gather_runtime")
    if gather_op is None or register_op is None or unregister_op is None:
        return gather_op, register_op, unregister_op
    if runtime_op is None:
        raise RuntimeError("vllm_ascend extension is missing the kv_cache_block_gather runtime capability check")
    if not runtime_op():
        raise RuntimeError(
            "packaged custom-op library does not expose both kv_cache_block_gather ACLNN runtime symbols"
        )
    return gather_op, register_op, unregister_op


def _load_mapped_gather_ops():
    """Load the packaged custom op and resolve its data/lifecycle API."""
    import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401

    activate_kv_cache_block_gather_runtime(torch)
    return _resolve_mapped_gather_ops()


def _validate_mapped_gather_layout(
    npu_tensors: list[torch.Tensor],
    cpu_tensors: list[torch.Tensor],
    kv_cache_groups_data_refs: list[list[CanonicalKVCacheRef]],
) -> None:
    """Reject any layout outside the intentionally narrow mapped domain."""
    if len(npu_tensors) != len(cpu_tensors) or not npu_tensors:
        raise ValueError("canonical CPU/NPU tensor lists are empty or mismatched")

    for tensor_idx, (cpu_tensor, npu_tensor) in enumerate(zip(cpu_tensors, npu_tensors)):
        if not cpu_tensor.is_contiguous() or not npu_tensor.is_contiguous():
            raise ValueError(f"canonical tensor {tensor_idx} is not contiguous")
        if cpu_tensor.stride(0) != cpu_tensor.shape[1] or npu_tensor.stride(0) != npu_tensor.shape[1]:
            raise ValueError(f"canonical tensor {tensor_idx} has a padded/strided row")
        page_bytes = int(npu_tensor.shape[1])
        if page_bytes % MAPPED_GATHER_ALIGNMENT_BYTES != 0:
            raise ValueError(
                f"canonical tensor {tensor_idx} page size {page_bytes} is not "
                f"{MAPPED_GATHER_ALIGNMENT_BYTES}-byte aligned"
            )

    for group_idx, group_data_refs in enumerate(kv_cache_groups_data_refs):
        for data_ref in group_data_refs:
            if data_ref.tensor_idx < 0 or data_ref.tensor_idx >= len(npu_tensors):
                raise ValueError(f"KV group {group_idx} references missing tensor {data_ref.tensor_idx}")
            row_bytes = int(npu_tensors[data_ref.tensor_idx].shape[1])
            if data_ref.page_size_bytes != row_bytes:
                raise ValueError(
                    f"KV group {group_idx} has unpadded page size "
                    f"{data_ref.page_size_bytes}, but canonical row size is "
                    f"{row_bytes}"
                )


def _new_descriptor_buffers(
    num_copy_ops: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pin_memory = is_pin_memory_available()
    return (
        torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=pin_memory),
        torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=pin_memory),
        torch.empty(num_copy_ops, dtype=torch.int64, pin_memory=pin_memory),
    )


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    block_size_factor: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
) -> None:
    """Compute byte pointers for sub-blocks of the given block IDs."""
    assert skip_count < block_size_factor

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if block_size_factor == 1:
        output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride
        return

    assert tensor.shape[1] % block_size_factor == 0
    sub_block_size = tensor.shape[1] // block_size_factor
    sub_offsets = np.arange(block_size_factor, dtype=np.uint64) * sub_block_size
    all_ptrs = (base_ptr + block_ids.astype(np.uint64)[:, np.newaxis] * row_stride) + sub_offsets[np.newaxis, :]
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


@dataclass
class _MappedIDBuffers:
    """One ID bundle whose lifetime is tied to one in-flight H2D job."""

    src_host: torch.Tensor
    dst_host: torch.Tensor
    src_device: torch.Tensor
    dst_device: torch.Tensor
    capacity: int


@dataclass
class _Transfer:
    job_id: int
    stream: torch.npu.Stream
    start_event: torch.npu.Event
    end_event: torch.npu.Event
    num_bytes: int
    batch_src: torch.Tensor | None
    batch_dst: torch.Tensor | None
    batch_sizes: torch.Tensor | None
    mapped_id_buffers: _MappedIDBuffers | None
    success: bool = True
    needs_stream_sync: bool = False


@dataclass
class _AsyncState:
    """Queues and reusable synchronization objects for one direction."""

    transfer_events: dict[int, torch.npu.Event] = field(default_factory=dict)
    transfers: deque[_Transfer] = field(default_factory=deque)
    stream_pool: list[torch.npu.Stream] = field(default_factory=list)
    event_pool: list[torch.npu.Event] = field(default_factory=list)
    poisoned: bool = False


class MappedOffloadingHandler(OffloadingWorker):
    """Worker-local offload with mandatory mapped gather for CPU-to-NPU loads."""

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        block_size_factor: int,
        num_cpu_blocks: int,
    ):
        if block_size_factor != 1:
            raise ValueError("MappedOffloadingHandler requires block_size_factor == 1")
        if num_cpu_blocks <= 0:
            raise ValueError("num_cpu_blocks must be greater than zero")

        self._closed = False
        self.kv_cache_groups_data_refs = kv_caches.group_data_refs
        self._gather_op, register_op, self._unregister_op = _load_mapped_gather_ops()
        if self._gather_op is None or register_op is None or self._unregister_op is None:
            raise RuntimeError("mapped-gather op or explicit host-pool lifecycle ops are unavailable")
        self._mapped_handles: list[int] = []

        pin_memory = is_pin_memory_available()
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        self.npu_tensors: list[torch.Tensor] = []
        self.cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            page_size_bytes = kv_cache_tensor.page_size_bytes
            npu_tensor = kv_cache_tensor.tensor.view(torch.int8).view((-1, page_size_bytes))

            started = time.monotonic()
            cpu_tensor = torch.zeros(
                (num_cpu_blocks, page_size_bytes),
                dtype=torch.int8,
                device="cpu",
                pin_memory=pin_memory,
            )
            logger.debug(
                "torch.zeros pinned tensor %d x %d (%.2f GB): %.3f s",
                num_cpu_blocks,
                page_size_bytes,
                num_cpu_blocks * page_size_bytes / 1e9,
                time.monotonic() - started,
            )
            self.npu_tensors.append(npu_tensor)
            self.cpu_tensors.append(cpu_tensor)

        _validate_mapped_gather_layout(
            self.npu_tensors,
            self.cpu_tensors,
            self.kv_cache_groups_data_refs,
        )

        # Directional queues remain independent so loads and stores do not
        # accidentally serialize each other.
        self._store_state = _AsyncState()
        self._load_state = _AsyncState()
        self._states = (self._store_state, self._load_state)
        self._store_descriptor_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._load_id_pool: list[_MappedIDBuffers] = []
        self._load_jobs = 0
        self._load_pages = 0
        self._load_bytes = 0
        self._store_jobs = 0
        self._store_pages = 0
        self._store_bytes = 0

        self._register_mapped_pool(register_op)
        logger.info(
            "Mapped %d worker-local CPU KV tensors for direct H2D gather",
            len(self.cpu_tensors),
        )

    def _register_mapped_pool(self, register_op) -> None:
        """Register all CPU tensors transactionally under this worker."""
        try:
            for tensor in self.cpu_tensors:
                handle = int(register_op(tensor))
                if handle <= 0:
                    raise RuntimeError(f"host-pool registration returned invalid handle {handle}")
                self._mapped_handles.append(handle)
        except Exception:
            try:
                self._close_mapped_pool()
            except Exception:
                logger.exception("Failed to roll back a partial mapped CPU pool")
            raise

    def _close_mapped_pool(self) -> None:
        """Release every mapping lease, preserving failed handles for retry."""
        if not self._mapped_handles:
            self._gather_op = None
            self._unregister_op = None
            return
        if self._unregister_op is None:
            raise RuntimeError("mapped CPU pool has handles but no unregister op")

        handles, self._mapped_handles = self._mapped_handles, []
        failed_handles: list[int] = []
        first_error: Exception | None = None
        for handle in reversed(handles):
            try:
                if not self._unregister_op(handle):
                    raise RuntimeError(f"mapped CPU pool handle {handle} was not active")
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                failed_handles.append(handle)
                logger.warning(
                    "Failed to unregister mapped CPU pool handle %d",
                    handle,
                    exc_info=True,
                )

        self._mapped_handles = failed_handles
        if failed_handles:
            raise RuntimeError(f"failed to unregister {len(failed_handles)} mapped CPU pool leases") from first_error

        self._gather_op = None
        self._unregister_op = None

    def transfer_async(
        self,
        job_id: int,
        transfer_spec: tuple[LoadStoreSpec, LoadStoreSpec],
    ) -> bool:
        """Compatibility wrapper for the pre-worker vLLM offload API."""
        src_spec, dst_spec = transfer_spec
        if isinstance(src_spec, GPULoadStoreSpec) and isinstance(dst_spec, CPULoadStoreSpec):
            return self.submit_store(job_id, src_spec, dst_spec)
        if isinstance(src_spec, CPULoadStoreSpec) and isinstance(dst_spec, GPULoadStoreSpec):
            return self.submit_load(job_id, src_spec, dst_spec)
        raise TypeError("mapped offload supports only NPU-to-CPU stores and CPU-to-NPU loads")

    @override
    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        return self._submit_store(job_id, src_spec, dst_spec)

    @override
    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        return self._submit_load(job_id, src_spec, dst_spec)

    def _submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        """Submit an NPU-to-CPU span copy."""
        if self._closed:
            raise RuntimeError("cannot submit to a shut down mapped worker")
        if not isinstance(dst_spec, BlockIDsLoadStoreSpec):
            raise TypeError("CPU KV store requires a block-ID destination spec")

        (
            src_blocks,
            dst_blocks,
            group_sizes,
            num_copy_ops,
            num_transfer_bytes,
        ) = self._make_transfer_plan(src_spec, dst_spec, src_spec)
        (
            batch_src,
            batch_dst,
            batch_sizes,
            copy_src,
            copy_dst,
            copy_sizes,
        ) = self._prepare_store_descriptors(
            src_blocks,
            dst_blocks,
            group_sizes,
            num_copy_ops,
        )

        state = self._store_state
        stream, start_event, end_event = self._acquire_async_resources(
            state,
            wait_for_current_stream=True,
        )
        transfer = _Transfer(
            job_id=job_id,
            stream=stream,
            start_event=start_event,
            end_event=end_event,
            num_bytes=num_transfer_bytes,
            batch_src=batch_src,
            batch_dst=batch_dst,
            batch_sizes=batch_sizes,
            mapped_id_buffers=None,
        )

        work_may_be_inflight = False
        completion_recorded = False
        try:
            with torch.npu.stream(stream):
                start_event.record(stream)
                # Event recording itself is asynchronous stream work.  If the
                # completion event fails, synchronize before reusing these
                # objects even when this is an otherwise empty transfer.
                work_may_be_inflight = True
                if num_copy_ops > 0:
                    torch.ops._C_ascend.swap_blocks_batch(
                        copy_src,
                        copy_dst,
                        copy_sizes,
                        DIRECTION_D2H,
                    )
                end_event.record(stream)
                completion_recorded = True
        except Exception:
            self._preserve_failed_submission(
                state,
                transfer,
                work_may_be_inflight,
                completion_recorded,
            )
            raise

        self._track_submission(state, transfer)
        self._store_jobs += 1
        self._store_pages += num_copy_ops
        self._store_bytes += num_transfer_bytes
        return True

    def _submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        """Submit a CPU-to-NPU mapped gather; no copy fallback exists."""
        if self._closed:
            raise RuntimeError("cannot submit to a shut down mapped worker")
        if not isinstance(src_spec, BlockIDsLoadStoreSpec):
            raise TypeError("CPU KV load requires a block-ID source spec")

        (
            src_blocks,
            dst_blocks,
            group_sizes,
            num_copy_ops,
            num_transfer_bytes,
        ) = self._make_transfer_plan(src_spec, dst_spec, dst_spec)
        if sum(group_sizes) != len(src_blocks) or len(src_blocks) != len(dst_blocks):
            raise ValueError("mapped gather requires one canonical ID per block")

        mapped_id_buffers = None
        if len(src_blocks) > 0:
            self._validate_mapped_ids(src_blocks, dst_blocks, group_sizes)
            mapped_id_buffers = self._acquire_mapped_ids(len(src_blocks))

        state = self._load_state
        stream, start_event, end_event = self._acquire_async_resources(
            state,
            wait_for_current_stream=False,
        )
        transfer = _Transfer(
            job_id=job_id,
            stream=stream,
            start_event=start_event,
            end_event=end_event,
            num_bytes=num_transfer_bytes,
            batch_src=None,
            batch_dst=None,
            batch_sizes=None,
            mapped_id_buffers=mapped_id_buffers,
        )

        work_may_be_inflight = False
        completion_recorded = False
        try:
            with torch.npu.stream(stream):
                start_event.record(stream)
                # Keep the event/stream lifecycle conservative for empty jobs
                # too; the start-event record may still be queued on device.
                work_may_be_inflight = True
                if num_copy_ops > 0:
                    assert mapped_id_buffers is not None
                    self._run_mapped_gather(
                        src_blocks,
                        dst_blocks,
                        group_sizes,
                        mapped_id_buffers,
                    )
                end_event.record(stream)
                completion_recorded = True
        except Exception:
            self._preserve_failed_submission(
                state,
                transfer,
                work_may_be_inflight,
                completion_recorded,
            )
            raise

        self._track_submission(state, transfer)
        self._load_jobs += 1
        self._load_pages += num_copy_ops
        self._load_bytes += num_transfer_bytes
        return True

    def _make_transfer_plan(
        self,
        src_spec: BlockIDsLoadStoreSpec,
        dst_spec: BlockIDsLoadStoreSpec,
        gpu_spec: GPULoadStoreSpec,
    ) -> tuple[np.ndarray, np.ndarray, list[int], int, int]:
        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        if src_blocks.ndim != 1 or dst_blocks.ndim != 1:
            raise ValueError("source and destination block IDs must be 1D")

        group_sizes = gpu_spec.group_sizes
        if len(group_sizes) != len(self.kv_cache_groups_data_refs):
            raise ValueError("group_sizes does not match canonical KV groups")
        # The mapped path requires block_size_factor == 1, so block_indices is
        # structural group metadata only; there is no sub-block expansion.
        if len(gpu_spec.block_indices) != len(self.kv_cache_groups_data_refs):
            raise ValueError("block_indices does not match canonical KV groups")
        for group_idx, group_size in enumerate(group_sizes):
            if group_size < 0:
                raise ValueError(f"KV group {group_idx} has negative size {group_size}")

        num_copy_ops = sum(
            group_size * len(group_refs)
            for group_size, group_refs in zip(
                group_sizes,
                self.kv_cache_groups_data_refs,
            )
        )
        num_transfer_bytes = sum(
            group_size * sum(ref.page_size_bytes for ref in group_refs)
            for group_size, group_refs in zip(
                group_sizes,
                self.kv_cache_groups_data_refs,
            )
        )
        return (
            src_blocks,
            dst_blocks,
            group_sizes,
            num_copy_ops,
            num_transfer_bytes,
        )

    @staticmethod
    def _acquire_async_resources(
        state: _AsyncState,
        *,
        wait_for_current_stream: bool,
    ) -> tuple[torch.npu.Stream, torch.npu.Event, torch.npu.Event]:
        if state.poisoned:
            raise RuntimeError(
                "cannot submit after a failed transfer left device work unsynchronized; retry handler shutdown"
            )
        stream = state.stream_pool.pop() if state.stream_pool else torch.npu.Stream()
        start_event = state.event_pool.pop() if state.event_pool else torch.npu.Event(enable_timing=True)
        end_event = state.event_pool.pop() if state.event_pool else torch.npu.Event(enable_timing=True)
        if wait_for_current_stream:
            stream.wait_stream(torch.npu.current_stream())
        if state.transfers:
            stream.wait_event(state.transfers[-1].end_event)
        return stream, start_event, end_event

    @staticmethod
    def _track_submission(state: _AsyncState, transfer: _Transfer) -> None:
        state.transfer_events[transfer.job_id] = transfer.end_event
        state.transfers.append(transfer)

    def _recycle_transfer_resources(
        self,
        state: _AsyncState,
        transfer: _Transfer,
    ) -> None:
        state.stream_pool.append(transfer.stream)
        state.event_pool.extend((transfer.end_event, transfer.start_event))
        if transfer.batch_src is not None:
            if transfer.batch_dst is None or transfer.batch_sizes is None:
                raise RuntimeError("store transfer is missing descriptor buffers")
            self._store_descriptor_pool.append(
                (
                    transfer.batch_src,
                    transfer.batch_dst,
                    transfer.batch_sizes,
                )
            )
        if transfer.mapped_id_buffers is not None:
            self._release_mapped_ids(transfer.mapped_id_buffers)

    def _preserve_failed_submission(
        self,
        state: _AsyncState,
        transfer: _Transfer,
        work_may_be_inflight: bool,
        completion_recorded: bool,
    ) -> None:
        transfer.success = False
        if work_may_be_inflight and not completion_recorded:
            try:
                transfer.end_event.record(transfer.stream)
                completion_recorded = True
            except Exception:
                logger.exception(
                    "Failed to record a completion event for a failed mapped KV transfer; synchronizing its stream"
                )
                try:
                    transfer.stream.synchronize()
                except Exception:
                    # The handler can no longer prove that resources referenced
                    # by this submission are idle. Keep the complete transfer
                    # reachable and reject more work until shutdown retries the
                    # stream synchronization successfully.
                    logger.exception(
                        "Failed to synchronize a failed mapped KV transfer; "
                        "the handler is poisoned until shutdown succeeds"
                    )
                    transfer.needs_stream_sync = True
                    state.poisoned = True
                    state.transfers.append(transfer)
                    return
                self._recycle_transfer_resources(state, transfer)
                return
        if completion_recorded:
            self._track_submission(state, transfer)
        else:
            # No device work was submitted, so all resources are immediately
            # reusable even though the submission itself raised.
            self._recycle_transfer_resources(state, transfer)

    def _prepare_store_descriptors(
        self,
        src_blocks: np.ndarray,
        dst_blocks: np.ndarray,
        group_sizes: list[int],
        num_copy_ops: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_src, batch_dst, batch_sizes = (
            self._store_descriptor_pool.pop() if self._store_descriptor_pool else _new_descriptor_buffers(num_copy_ops)
        )
        if batch_src.numel() < num_copy_ops:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_copy_ops)

        copy_src = batch_src[:num_copy_ops]
        copy_dst = batch_dst[:num_copy_ops]
        copy_sizes = batch_sizes[:num_copy_ops]
        all_src = copy_src.numpy()
        all_dst = copy_dst.numpy()
        all_sizes = copy_sizes.numpy()

        src_offset = dst_offset = op_idx = 0
        for group_idx, (group_size, group_refs) in enumerate(
            zip(
                group_sizes,
                self.kv_cache_groups_data_refs,
            )
        ):
            if group_size == 0:
                continue

            src_end = src_offset + group_size
            dst_end = dst_offset + group_size
            if src_end > len(src_blocks) or dst_end > len(dst_blocks):
                raise ValueError(f"store group {group_idx} exceeds its block-ID array")

            group_src = src_blocks[src_offset:src_end]
            group_dst = dst_blocks[dst_offset:dst_end]
            for data_ref in group_refs:
                tensor_idx = data_ref.tensor_idx
                op_end = op_idx + group_size
                compute_sub_block_ptrs(
                    group_src,
                    1,
                    all_src[op_idx:op_end],
                    self.npu_tensors[tensor_idx],
                )
                compute_sub_block_ptrs(
                    group_dst,
                    1,
                    all_dst[op_idx:op_end],
                    self.cpu_tensors[tensor_idx],
                )
                all_sizes[op_idx:op_end] = data_ref.page_size_bytes
                op_idx = op_end

            src_offset = src_end
            dst_offset = dst_end

        if src_offset != len(src_blocks) or dst_offset != len(dst_blocks):
            raise ValueError("store groups do not consume all block IDs")
        if op_idx != num_copy_ops:
            raise ValueError("copy descriptor count does not match the plan")
        return (
            batch_src,
            batch_dst,
            batch_sizes,
            copy_src,
            copy_dst,
            copy_sizes,
        )

    def _validate_mapped_ids(
        self,
        src_blocks: np.ndarray,
        dst_blocks: np.ndarray,
        group_sizes: list[int],
    ) -> None:
        if not np.issubdtype(src_blocks.dtype, np.integer) or not np.issubdtype(
            dst_blocks.dtype,
            np.integer,
        ):
            raise ValueError("mapped-gather block IDs must have an integer dtype")

        int32_max = np.iinfo(np.int32).max
        src_offset = dst_offset = 0
        for group_idx, (group_size, group_refs) in enumerate(zip(group_sizes, self.kv_cache_groups_data_refs)):
            src_end = src_offset + group_size
            dst_end = dst_offset + group_size
            if src_end > len(src_blocks) or dst_end > len(dst_blocks):
                raise ValueError(f"mapped-gather group {group_idx} exceeds its block-ID array")

            if group_size:
                group_src = src_blocks[src_offset:src_end]
                group_dst = dst_blocks[dst_offset:dst_end]
                src_min, src_max = int(group_src.min()), int(group_src.max())
                dst_min, dst_max = int(group_dst.min()), int(group_dst.max())
                if src_min < 0 or dst_min < 0:
                    raise ValueError(f"mapped-gather group {group_idx} contains negative block IDs")
                if src_max > int32_max or dst_max > int32_max:
                    raise ValueError(f"mapped-gather group {group_idx} block IDs exceed int32 range")
                for data_ref in group_refs:
                    tensor_idx = data_ref.tensor_idx
                    src_rows = int(self.cpu_tensors[tensor_idx].shape[0])
                    dst_rows = int(self.npu_tensors[tensor_idx].shape[0])
                    if src_max >= src_rows:
                        raise ValueError(
                            f"mapped-gather group {group_idx} source block ID "
                            f"{src_max} is out of bounds for tensor {tensor_idx} "
                            f"with {src_rows} rows"
                        )
                    if dst_max >= dst_rows:
                        raise ValueError(
                            f"mapped-gather group {group_idx} destination block ID "
                            f"{dst_max} is out of bounds for tensor {tensor_idx} "
                            f"with {dst_rows} rows"
                        )

            src_offset = src_end
            dst_offset = dst_end

        if src_offset != len(src_blocks) or dst_offset != len(dst_blocks):
            raise ValueError("mapped-gather group sizes do not consume all block IDs")

    def _new_mapped_ids(self, capacity: int) -> _MappedIDBuffers:
        pin_memory = is_pin_memory_available()
        device = self.npu_tensors[0].device
        return _MappedIDBuffers(
            src_host=torch.empty(
                capacity,
                dtype=torch.int32,
                device="cpu",
                pin_memory=pin_memory,
            ),
            dst_host=torch.empty(
                capacity,
                dtype=torch.int32,
                device="cpu",
                pin_memory=pin_memory,
            ),
            src_device=torch.empty(capacity, dtype=torch.int32, device=device),
            dst_device=torch.empty(capacity, dtype=torch.int32, device=device),
            capacity=capacity,
        )

    def _acquire_mapped_ids(self, num_ids: int) -> _MappedIDBuffers:
        best_idx: int | None = None
        for idx, buffers in enumerate(self._load_id_pool):
            if buffers.capacity >= num_ids and (
                best_idx is None or buffers.capacity < self._load_id_pool[best_idx].capacity
            ):
                best_idx = idx
        if best_idx is not None:
            return self._load_id_pool.pop(best_idx)

        capacity = num_ids
        if num_ids <= MAPPED_GATHER_MAX_CACHED_IDS:
            capacity = min(
                1 << (num_ids - 1).bit_length(),
                MAPPED_GATHER_MAX_CACHED_IDS,
            )
        return self._new_mapped_ids(capacity)

    def _release_mapped_ids(self, buffers: _MappedIDBuffers) -> None:
        if (
            buffers.capacity <= MAPPED_GATHER_MAX_CACHED_IDS
            and len(self._load_id_pool) < MAPPED_GATHER_ID_BUFFER_CACHE_SIZE
        ):
            self._load_id_pool.append(buffers)

    def _run_mapped_gather(
        self,
        src_blocks: np.ndarray,
        dst_blocks: np.ndarray,
        group_sizes: list[int],
        buffers: _MappedIDBuffers,
    ) -> None:
        gather_op = self._gather_op
        if gather_op is None or not self._mapped_handles:
            raise RuntimeError("mapped CPU pool is not active")

        num_ids = len(src_blocks)
        buffers.src_host[:num_ids].numpy()[:] = src_blocks
        buffers.dst_host[:num_ids].numpy()[:] = dst_blocks
        src_ids = buffers.src_device[:num_ids]
        dst_ids = buffers.dst_device[:num_ids]
        src_ids.copy_(buffers.src_host[:num_ids], non_blocking=True)
        dst_ids.copy_(buffers.dst_host[:num_ids], non_blocking=True)

        offset = 0
        for group_size, group_refs in zip(
            group_sizes,
            self.kv_cache_groups_data_refs,
        ):
            if group_size == 0:
                continue
            end = offset + group_size
            group_src_ids = src_ids[offset:end]
            group_dst_ids = dst_ids[offset:end]
            for data_ref in group_refs:
                tensor_idx = data_ref.tensor_idx
                gather_op(
                    group_src_ids,
                    self.cpu_tensors[tensor_idx],
                    group_dst_ids,
                    self.npu_tensors[tensor_idx],
                )
            offset = end

        if offset != num_ids:
            raise ValueError("mapped-gather groups do not consume all block IDs")

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        for state in self._states:
            while state.transfers and not state.transfers[0].needs_stream_sync and state.transfers[0].end_event.query():
                transfer = state.transfers.popleft()
                transfer_time = transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
                results.append(
                    TransferResult(
                        job_id=transfer.job_id,
                        success=transfer.success,
                        transfer_size=transfer.num_bytes,
                        transfer_time=transfer_time,
                    )
                )
                self._recycle_transfer_resources(state, transfer)
                state.transfer_events.pop(transfer.job_id, None)
        return results

    def wait(self, job_ids: set[int]) -> None:
        for state in self._states:
            for job_id in job_ids:
                event = state.transfer_events.get(job_id)
                if event is not None:
                    event.synchronize()

    def shutdown(self) -> None:
        if self._closed:
            return

        # Do not pop before synchronization succeeds.  A failed shutdown keeps
        # every resource reachable so a later call can retry safely.
        for state in self._states:
            while state.transfers:
                transfer = state.transfers[0]
                if transfer.needs_stream_sync:
                    transfer.stream.synchronize()
                else:
                    transfer.end_event.synchronize()
                state.transfers.popleft()
                state.transfer_events.pop(transfer.job_id, None)
            state.poisoned = False

        # There can be no device read of host mappings after both queues drain.
        # Unregister before releasing the tensors or shared-memory views.
        self._close_mapped_pool()

        logger.info(
            "Mapped KV offload summary: load_jobs=%d load_pages=%d "
            "load_bytes=%d store_jobs=%d store_pages=%d store_bytes=%d",
            self._load_jobs,
            self._load_pages,
            self._load_bytes,
            self._store_jobs,
            self._store_pages,
            self._store_bytes,
        )

        for state in self._states:
            state.transfer_events.clear()
            state.stream_pool.clear()
            state.event_pool.clear()
        self._store_descriptor_pool.clear()
        self._load_id_pool.clear()

        self.cpu_tensors.clear()
        self.npu_tensors.clear()
        self._closed = True


class MappedOffloadingSpec(CPUOffloadingSpec):
    """Explicit factory entry for selecting the fail-fast mapped handler."""

    @override  # type: ignore[misc]  # mypy skips the external vLLM base
    def create_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        handler = getattr(self, "_mapped_handler", None)
        if handler is None:
            handler = MappedOffloadingHandler(
                kv_caches=kv_caches,
                block_size_factor=self.block_size_factor,
                num_cpu_blocks=self.num_blocks,
            )
            self._mapped_handler = handler

        return handler
