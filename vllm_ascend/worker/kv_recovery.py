# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.


def observe_first_compute_if_supported(
    model_runner: object,
    scheduler_output: object,
) -> None:
    """Forward KV-recovery observation when the paired core supports it."""
    observe_first_compute = getattr(
        model_runner,
        "observe_kv_recovery_first_compute",
        None,
    )
    if callable(observe_first_compute):
        observe_first_compute(scheduler_output)
