# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE = REPO_ROOT / "tools" / "docker" / "verify_baked_custom_ops.sh"


def _custom_ops_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    vendor = tmp_path / "vendors" / "custom_transformer"
    header = vendor / "op_api/include/aclnnop/aclnn_moe_init_routing_custom.h"
    library = vendor / "op_api/lib/libcust_opapi.so"
    version = vendor / "version.info"
    cann_version = tmp_path / "cann" / "opp" / "version.info"
    for path in (header, library, version, cann_version):
        path.parent.mkdir(parents=True, exist_ok=True)
    header.write_text("header\n", encoding="utf-8")
    library.write_bytes(b"library\n")
    version.write_text("custom_opp_compiler_version=9.0.0\n", encoding="utf-8")
    cann_version.write_text("Version=9.0.0\n", encoding="utf-8")
    return vendor, header, library, cann_version


def _run_probe(
    vendor: Path | None,
    cann_version: Path,
    *,
    compile_mode: str | None = "1",
    library_path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in ("COMPILE_CUSTOM_KERNELS", "ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH"):
        env.pop(name, None)
    if compile_mode is not None:
        env["COMPILE_CUSTOM_KERNELS"] = compile_mode
    if vendor is not None:
        env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)
        env["LD_LIBRARY_PATH"] = library_path or str(vendor / "op_api/lib")
    return subprocess.run(
        ["bash", str(PROBE), str(cann_version)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_enabled_custom_ops_report_paths_versions_and_checksums(tmp_path: Path) -> None:
    vendor, header, library, cann_version = _custom_ops_fixture(tmp_path)

    result = _run_probe(vendor, cann_version)

    assert result.returncode == 0
    assert "custom_ops.mode=enabled" in result.stdout
    assert "custom_ops.compile_custom_kernels=1" in result.stdout
    assert f"custom_ops.vendor_path={vendor}" in result.stdout
    assert "custom_ops.custom_version=9.0.0" in result.stdout
    assert "custom_ops.cann_version=9.0.0" in result.stdout
    assert f"custom_ops.header_sha256={hashlib.sha256(header.read_bytes()).hexdigest()}" in result.stdout
    assert f"custom_ops.library_sha256={hashlib.sha256(library.read_bytes()).hexdigest()}" in result.stdout
    assert "custom_ops.status=ready" in result.stdout


def test_enabled_mode_can_be_inferred_from_legacy_image_paths(tmp_path: Path) -> None:
    vendor, _, _, cann_version = _custom_ops_fixture(tmp_path)

    result = _run_probe(vendor, cann_version, compile_mode=None)

    assert result.returncode == 0
    assert "custom_ops.compile_custom_kernels=inferred-enabled" in result.stdout


def test_disabled_mode_does_not_advertise_custom_ops(tmp_path: Path) -> None:
    cann_version = tmp_path / "version.info"
    cann_version.write_text("Version=9.0.0\n", encoding="utf-8")

    result = _run_probe(None, cann_version, compile_mode="0")

    assert result.returncode == 0
    assert result.stdout == (
        "custom_ops.mode=disabled\ncustom_ops.compile_custom_kernels=0\ncustom_ops.status=not-built\n"
    )


def test_disabled_mode_rejects_exported_vendor_path(tmp_path: Path) -> None:
    vendor, _, _, cann_version = _custom_ops_fixture(tmp_path)

    result = _run_probe(vendor, cann_version, compile_mode="0")

    assert result.returncode == 1
    assert "COMPILE_CUSTOM_KERNELS=0 but ASCEND_CUSTOM_OPP_PATH is still exported" in result.stderr


@pytest.mark.parametrize("missing", ["header", "library", "version"])
def test_enabled_mode_fails_closed_when_an_artifact_is_missing(tmp_path: Path, missing: str) -> None:
    vendor, header, library, cann_version = _custom_ops_fixture(tmp_path)
    paths = {"header": header, "library": library, "version": vendor / "version.info"}
    paths[missing].unlink()

    result = _run_probe(vendor, cann_version)

    assert result.returncode == 1


def test_enabled_mode_requires_the_op_api_library_path(tmp_path: Path) -> None:
    vendor, _, _, cann_version = _custom_ops_fixture(tmp_path)

    result = _run_probe(
        vendor,
        cann_version,
        library_path=str(tmp_path / "unrelated-lib"),
    )

    assert result.returncode == 1
    assert "custom op API library is not activated" in result.stderr


def test_enabled_mode_rejects_cann_version_mismatch(tmp_path: Path) -> None:
    vendor, _, _, cann_version = _custom_ops_fixture(tmp_path)
    cann_version.write_text("Version=8.5.0\n", encoding="utf-8")

    result = _run_probe(vendor, cann_version)

    assert result.returncode == 1
    assert "custom-op compiler version 9.0.0 does not match CANN 8.5.0" in result.stderr
