# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKE_FILE = REPO_ROOT / "csrc" / "mc2" / "matmul_allreduce_add_rmsnorm" / "op_host" / "CMakeLists.txt"


def _compile_options_block() -> str:
    source = CMAKE_FILE.read_text(encoding="utf-8")
    match = re.search(
        r"add_ops_compile_options\(\s*OP_NAME\s+MatmulAllreduceAddRmsnorm\s+"
        r"OPTIONS(?P<options>.*?)\n\)",
        source,
        flags=re.DOTALL,
    )
    assert match is not None, "matmul-allreduce compile-options block is missing"
    return match.group(0)


def test_matmul_allreduce_compile_options_use_registered_op_and_catlass() -> None:
    source = CMAKE_FILE.read_text(encoding="utf-8")
    block = _compile_options_block()

    assert "OP_NAME MatmulAllreduceAddRmsnorm\n" in block
    assert "-I${CMAKE_SOURCE_DIR}/third_party/catlass/include" in block
    assert "MatmulAllreduceAddRmsnormTensorList" not in source
