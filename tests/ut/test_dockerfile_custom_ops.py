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
#

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES = ("Dockerfile", "Dockerfile.openEuler")
CUSTOM_OPP_PATH = "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"


def _stage(dockerfile: str, name: str) -> str:
    marker = f" AS {name}\n"
    start = dockerfile.index(marker) + len(marker)
    end = dockerfile.find("\nFROM ", start)
    return dockerfile[start:] if end == -1 else dockerfile[start:end]


@pytest.mark.parametrize("dockerfile_name", DOCKERFILES)
def test_custom_kernel_image_stage_exports_and_validates_baked_package(
    dockerfile_name: str,
) -> None:
    dockerfile = (REPO_ROOT / dockerfile_name).read_text(encoding="utf-8")
    enabled = _stage(dockerfile, "custom-kernels-1")

    assert "ENV COMPILE_CUSTOM_KERNELS=1" in enabled
    assert f"ASCEND_CUSTOM_OPP_PATH={CUSTOM_OPP_PATH}" in enabled
    assert f"LD_LIBRARY_PATH={CUSTOM_OPP_PATH}/op_api/lib:$LD_LIBRARY_PATH" in enabled
    assert ('test -f "$ASCEND_CUSTOM_OPP_PATH/op_api/include/aclnnop/aclnn_moe_init_routing_custom.h"') in enabled
    assert 'test -f "$ASCEND_CUSTOM_OPP_PATH/op_api/lib/libcust_opapi.so"' in enabled


@pytest.mark.parametrize("dockerfile_name", DOCKERFILES)
def test_lightweight_image_stage_does_not_advertise_custom_ops(
    dockerfile_name: str,
) -> None:
    dockerfile = (REPO_ROOT / dockerfile_name).read_text(encoding="utf-8")
    disabled = _stage(dockerfile, "custom-kernels-0")

    assert "ENV COMPILE_CUSTOM_KERNELS=0" in disabled
    assert "ASCEND_CUSTOM_OPP_PATH" not in disabled
    assert "op_api/lib" not in disabled
    assert "image custom ops were not built (COMPILE_CUSTOM_KERNELS=0)" in disabled


@pytest.mark.parametrize("dockerfile_name", DOCKERFILES)
def test_compile_custom_kernels_selects_the_final_runtime_stage(
    dockerfile_name: str,
) -> None:
    dockerfile = (REPO_ROOT / dockerfile_name).read_text(encoding="utf-8")

    assert dockerfile.index("ARG COMPILE_CUSTOM_KERNELS=1") < dockerfile.index("FROM ")
    assert "FROM custom-kernels-${COMPILE_CUSTOM_KERNELS}\n" in dockerfile
