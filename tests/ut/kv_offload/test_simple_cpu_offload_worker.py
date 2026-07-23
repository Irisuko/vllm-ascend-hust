"""验证 NPU Simple CPU Offload worker 的 store 事件传递。

测试配对的 vllm-hust 提交：
53decfd3a41eba482031e9a7ac92f14585fb2d54。
"""

from unittest.mock import MagicMock, patch

from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadMetadata
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

from vllm_ascend.simple_kv_offload.copy_backend import NPUDmaCopyBackend
from vllm_ascend.simple_kv_offload.worker import SimpleCPUOffloadNPUWorker


def test_npu_worker_store_records_and_forwards_compute_event() -> None:
    """继承的生产 store 路径应记录并转交计算完成事件。"""
    worker = SimpleCPUOffloadNPUWorker.__new__(SimpleCPUOffloadNPUWorker)
    backend = MagicMock(spec=NPUDmaCopyBackend)
    worker._backend = backend
    worker._connector_metadata = SimpleCPUOffloadMetadata(
        store_event=7,
        store_gpu_blocks=[3, 4],
        store_cpu_blocks=[11, 12],
    )
    worker._store_compute_done = None
    worker._load_events = []
    worker._store_events = []
    worker._pending_load_event_indices = set()
    worker._pending_store_event_indices = set()

    compute_event = MagicMock()
    compute_stream = object()
    with (
        patch(
            "vllm.v1.simple_kv_offload.worker.torch.Event",
            return_value=compute_event,
        ),
        patch(
            "vllm.v1.simple_kv_offload.worker.torch.cuda.current_stream",
            return_value=compute_stream,
        ),
    ):
        worker.get_finished(set())

    assert SimpleCPUOffloadNPUWorker.get_finished is SimpleCPUOffloadWorker.get_finished
    compute_event.record.assert_called_once_with(compute_stream)
    backend.launch_copy.assert_called_once_with(
        [3, 4],
        [11, 12],
        is_store=True,
        event_idx=7,
        events_list=worker._store_events,
        wait_event=compute_event,
    )
