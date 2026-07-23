#!/usr/bin/env python3

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
#

"""Fail-closed validation for Ascend devices exposed to CI containers."""

from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

NPU_SMI_CANDIDATES = (
    Path("/usr/local/bin/npu-smi"),
    Path("/usr/local/Ascend/driver/tools/npu-smi"),
    Path("/usr/local/Ascend/driver/tools/npu-smi/npu-smi"),
)
COMMON_DEVICE_NAMES = ("davinci_manager", "devmm_svm", "hisi_hdc")


class RuntimeContractError(RuntimeError):
    """Raised when the container does not satisfy the Ascend runtime contract."""


def parse_visible_device_ids(value: str, expected_npus: int) -> tuple[int, ...]:
    if expected_npus < 1:
        raise RuntimeContractError("expected_npus must be positive for an NPU job")
    if not value.strip():
        raise RuntimeContractError("ASCEND_RT_VISIBLE_DEVICES is not set")

    raw_ids = [item.strip() for item in value.split(",")]
    try:
        device_ids = tuple(int(item) for item in raw_ids)
    except ValueError as exc:
        raise RuntimeContractError(f"ASCEND_RT_VISIBLE_DEVICES must contain integer IDs: {value!r}") from exc
    if any(device_id < 0 for device_id in device_ids):
        raise RuntimeContractError("Ascend device IDs must be non-negative")
    if len(set(device_ids)) != len(device_ids):
        raise RuntimeContractError("ASCEND_RT_VISIBLE_DEVICES contains duplicate IDs")
    if len(device_ids) != expected_npus:
        raise RuntimeContractError(
            "Visible Ascend device count does not match the selected test group: "
            f"expected {expected_npus}, got {len(device_ids)}"
        )
    return device_ids


def resolve_npu_smi(candidates: Sequence[Path] = NPU_SMI_CANDIDATES) -> Path:
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise RuntimeContractError(f"npu-smi is unavailable; checked: {checked}")


def validate_device_nodes(
    device_root: Path,
    device_ids: Sequence[int],
    *,
    is_char_device: Callable[[Path], bool] | None = None,
) -> tuple[Path, ...]:
    if is_char_device is None:
        is_char_device = lambda path: stat.S_ISCHR(path.stat().st_mode)

    required = tuple(device_root / name for name in COMMON_DEVICE_NAMES) + tuple(
        device_root / f"davinci{device_id}" for device_id in device_ids
    )
    invalid = []
    for path in required:
        try:
            valid = is_char_device(path)
        except FileNotFoundError:
            valid = False
        if not valid:
            invalid.append(str(path))
    if invalid:
        raise RuntimeContractError("Required Ascend character devices are unavailable: " + ", ".join(invalid))
    return required


def probe_npu_smi(
    npu_smi: Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    try:
        result = run(
            [str(npu_smi), "info"],
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeContractError("npu-smi info timed out after 60 seconds") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise RuntimeContractError(f"npu-smi info failed with exit code {result.returncode}: {detail}")


def validate_runtime(
    *,
    expected_npus: int,
    visible_devices: str,
    device_root: Path = Path("/dev"),
    npu_smi_candidates: Sequence[Path] = NPU_SMI_CANDIDATES,
    is_char_device: Callable[[Path], bool] | None = None,
    run_probe: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[Path, tuple[int, ...]]:
    device_ids = parse_visible_device_ids(visible_devices, expected_npus)
    npu_smi = resolve_npu_smi(npu_smi_candidates)
    validate_device_nodes(device_root, device_ids, is_char_device=is_char_device)
    probe_npu_smi(npu_smi, run=run_probe)
    return npu_smi, device_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Ascend device exposure inside a CI job container.")
    parser.add_argument("--expected-npus", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        npu_smi, device_ids = validate_runtime(
            expected_npus=args.expected_npus,
            visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        )
    except RuntimeContractError as exc:
        print(f"Ascend container runtime preflight failed: {exc}", file=sys.stderr)
        return 1

    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as env_file:
            env_file.write(f"NPU_SMI_BIN={npu_smi}\n")
    print(f"Ascend container runtime ready: devices={','.join(map(str, device_ids))}, npu_smi={npu_smi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
