# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def test_build_requirements_only_contain_setup_dependencies():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    build_requirements = tomllib.loads(pyproject.read_text())["build-system"]["requires"]

    package_names = {
        requirement.split("[", 1)[0].split("=", 1)[0].split("<", 1)[0].lower() for requirement in build_requirements
    }
    runtime_only_packages = {
        "fastapi",
        "torch",
        "torch-npu",
        "transformers",
        "triton-ascend",
    }

    assert package_names.isdisjoint(runtime_only_packages)


def test_custom_op_metadata_uses_opc_discovery_filename():
    root = Path(__file__).resolve().parents[2]
    build_functions = (root / "csrc/cmake/func.cmake").read_text()

    assert (
        "set(CUSTOM_OPS_INFO_JSON ${CUSTOM_OPS_INFO_DIR}/aic-${OPINFO_COMPUTE_UNIT}-ops-info.json)" in build_functions
    )
    assert "copy_if_different ${OPS_INFO_JSON} ${CUSTOM_OPS_INFO_JSON}" in build_functions
    assert "DEPENDS ${CUSTOM_OPS_INFO_JSON}" in build_functions


def test_dual_editable_documentation_uses_target_specific_flow():
    root = Path(__file__).resolve().parents[2]
    required = (
        "requirements/common.txt",
        "/path/to/vllm-ascend-hust/requirements.txt",
        "requirements/build/empty.txt",
        "VLLM_TARGET_DEVICE=empty uv pip install -e .",
        "--no-build-isolation --no-deps",
        "COMPILE_CUSTOM_KERNELS=0 uv pip install -e . --no-deps",
    )

    for readme in ("README.md", "README.zh.md"):
        contents = (root / readme).read_text()
        for command_fragment in required:
            assert command_fragment in contents


def test_custom_op_builder_normalizes_supported_device_family_shorthands():
    root = Path(__file__).resolve().parents[2]
    builder = (root / "csrc/build_aclnn.sh").read_text()

    assert "910b) SOC_VERSION=ascend910b1" in builder
    assert "910c) SOC_VERSION=ascend910_9392" in builder
    assert "310p) SOC_VERSION=ascend310p1" in builder
    assert "input_SOC_VERSION=${INPUT_SOC_VERSION:-<unset>}" in builder
