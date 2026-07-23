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

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "scripts" / "check_ascend_container_runtime.py"
)
SPEC = importlib.util.spec_from_file_location("check_ascend_container_runtime", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def _successful_probe(*args, **kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")


def test_runtime_contract_accepts_exact_visible_device_set(tmp_path: Path) -> None:
    npu_smi = tmp_path / "npu-smi"
    npu_smi.write_text("#!/bin/sh\n", encoding="utf-8")
    npu_smi.chmod(0o755)
    device_root = tmp_path / "dev"
    device_root.mkdir()
    expected_nodes = {
        device_root / "davinci_manager",
        device_root / "devmm_svm",
        device_root / "hisi_hdc",
        device_root / "davinci1",
    }

    resolved_binary, device_ids = runtime.validate_runtime(
        expected_npus=1,
        visible_devices="1",
        device_root=device_root,
        npu_smi_candidates=(npu_smi,),
        is_char_device=lambda path: path in expected_nodes,
        run_probe=_successful_probe,
    )

    assert resolved_binary == npu_smi
    assert device_ids == (1,)


@pytest.mark.parametrize("visible_devices", ["", "0,1", "1,1", "invalid"])
def test_runtime_contract_rejects_invalid_visibility(tmp_path: Path, visible_devices: str) -> None:
    npu_smi = tmp_path / "npu-smi"
    npu_smi.write_text("#!/bin/sh\n", encoding="utf-8")
    npu_smi.chmod(0o755)

    with pytest.raises(runtime.RuntimeContractError):
        runtime.validate_runtime(
            expected_npus=1,
            visible_devices=visible_devices,
            npu_smi_candidates=(npu_smi,),
            is_char_device=lambda _path: True,
            run_probe=_successful_probe,
        )


def test_runtime_contract_rejects_missing_device_node(tmp_path: Path) -> None:
    npu_smi = tmp_path / "npu-smi"
    npu_smi.write_text("#!/bin/sh\n", encoding="utf-8")
    npu_smi.chmod(0o755)

    with pytest.raises(runtime.RuntimeContractError, match="davinci1"):
        runtime.validate_runtime(
            expected_npus=1,
            visible_devices="1",
            npu_smi_candidates=(npu_smi,),
            is_char_device=lambda path: path.name != "davinci1",
            run_probe=_successful_probe,
        )


def test_runtime_contract_propagates_npu_smi_failure(tmp_path: Path) -> None:
    npu_smi = tmp_path / "npu-smi"
    npu_smi.write_text("#!/bin/sh\n", encoding="utf-8")
    npu_smi.chmod(0o755)

    def failed_probe(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args[0], returncode=7, stdout="", stderr="driver unavailable")

    with pytest.raises(runtime.RuntimeContractError, match="driver unavailable"):
        runtime.validate_runtime(
            expected_npus=1,
            visible_devices="1",
            npu_smi_candidates=(npu_smi,),
            is_char_device=lambda _path: True,
            run_probe=failed_probe,
        )


def test_cli_fails_closed_without_visible_device_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--expected-npus", "1"])

    assert runtime.main() == 1
    assert "ASCEND_RT_VISIBLE_DEVICES is not set" in capsys.readouterr().err
