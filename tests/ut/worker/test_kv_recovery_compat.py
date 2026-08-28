# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.

from types import SimpleNamespace
from unittest.mock import Mock

from vllm_ascend.worker.kv_recovery import observe_first_compute_if_supported


def test_stock_core_without_kv_recovery_hook_is_noop() -> None:
    observe_first_compute_if_supported(SimpleNamespace(), object())


def test_runtime_kv_recovery_hook_receives_scheduler_output() -> None:
    scheduler_output = object()
    observe_first_compute = Mock()
    model_runner = SimpleNamespace(
        observe_kv_recovery_first_compute=observe_first_compute,
    )

    observe_first_compute_if_supported(model_runner, scheduler_output)

    observe_first_compute.assert_called_once_with(scheduler_output)
