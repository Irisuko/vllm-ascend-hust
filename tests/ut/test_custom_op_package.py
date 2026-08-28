from pathlib import Path
from types import SimpleNamespace

from vllm_ascend.custom_op_package import (
    CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH,
    CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR,
    CUSTOM_OP_GATHER_KERNEL_MANIFEST,
    CUSTOM_OPAPI_RELATIVE_PATH,
    CUSTOM_OPP_ENV,
    activate_kv_cache_block_gather_runtime,
    bootstrap_custom_op_package_env,
    bundled_custom_op_vendor_path,
    resolve_custom_op_package,
)


def _create_bundled_package(package_dir: Path) -> Path:
    opapi = bundled_custom_op_vendor_path(package_dir) / CUSTOM_OPAPI_RELATIVE_PATH
    opapi.parent.mkdir(parents=True)
    opapi.touch()
    vendor = bundled_custom_op_vendor_path(package_dir)
    gather_header = vendor / CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH
    gather_header.parent.mkdir(parents=True)
    gather_header.touch()
    gather_manifest = (
        vendor / CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR / "ascend910b" / CUSTOM_OP_GATHER_KERNEL_MANIFEST
    )
    gather_manifest.parent.mkdir(parents=True)
    gather_manifest.touch()
    return opapi


def test_resolve_bundled_custom_op_package(tmp_path: Path):
    opapi = _create_bundled_package(tmp_path)

    resolution = resolve_custom_op_package(package_dir=tmp_path)

    assert resolution.available
    assert resolution.source == "bundled"
    assert resolution.vendor_path == bundled_custom_op_vendor_path(tmp_path)
    assert resolution.opapi_library == opapi
    assert "using bundled custom-op package" in resolution.reason


def test_resolve_reports_missing_vendor_directory(tmp_path: Path):
    resolution = resolve_custom_op_package(package_dir=tmp_path)

    assert not resolution.available
    assert resolution.vendor_path is None
    assert resolution.opapi_library is None
    assert resolution.reason == (
        f"bundled custom-op vendor directory is missing: {bundled_custom_op_vendor_path(tmp_path)}"
    )


def test_resolve_reports_missing_bundled_opapi(tmp_path: Path):
    vendor_path = bundled_custom_op_vendor_path(tmp_path)
    vendor_path.mkdir(parents=True)

    resolution = resolve_custom_op_package(package_dir=tmp_path)

    assert not resolution.available
    assert resolution.vendor_path == vendor_path
    assert resolution.opapi_library is None
    assert resolution.reason == (
        f"bundled custom-op opapi library is missing: {vendor_path / CUSTOM_OPAPI_RELATIVE_PATH}"
    )


def test_resolve_reports_missing_bundled_gather_header(tmp_path: Path):
    vendor_path = bundled_custom_op_vendor_path(tmp_path)
    opapi = vendor_path / CUSTOM_OPAPI_RELATIVE_PATH
    opapi.parent.mkdir(parents=True)
    opapi.touch()

    resolution = resolve_custom_op_package(package_dir=tmp_path)

    assert not resolution.available
    assert resolution.vendor_path == vendor_path
    assert resolution.opapi_library is None
    assert resolution.reason == (
        f"bundled kv_cache_block_gather opapi header is missing: {vendor_path / CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH}"
    )


def test_resolve_reports_missing_bundled_gather_manifest(tmp_path: Path):
    vendor_path = bundled_custom_op_vendor_path(tmp_path)
    opapi = vendor_path / CUSTOM_OPAPI_RELATIVE_PATH
    opapi.parent.mkdir(parents=True)
    opapi.touch()
    gather_header = vendor_path / CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH
    gather_header.parent.mkdir(parents=True)
    gather_header.touch()

    resolution = resolve_custom_op_package(package_dir=tmp_path)

    assert not resolution.available
    assert resolution.vendor_path == vendor_path
    assert resolution.opapi_library is None
    assert resolution.reason == (
        "bundled kv_cache_block_gather kernel manifest is missing under: "
        f"{vendor_path / CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR}"
    )


def test_bootstrap_selects_bundled_artifacts_without_manual_paths(tmp_path: Path):
    opapi = _create_bundled_package(tmp_path)
    old_opp = tmp_path / "existing-opp"
    environ = {CUSTOM_OPP_ENV: str(old_opp)}

    resolution = bootstrap_custom_op_package_env(
        package_dir=tmp_path,
        include_vendor_lib=True,
        environ=environ,
    )

    vendor = bundled_custom_op_vendor_path(tmp_path)
    assert resolution.available
    assert resolution.opapi_library == opapi
    assert environ[CUSTOM_OPP_ENV].split(":") == [str(vendor), str(old_opp)]
    assert environ["LD_LIBRARY_PATH"] == str(vendor / "op_api" / "lib")


def test_activate_uses_registered_extension_loader(tmp_path: Path):
    opapi = _create_bundled_package(tmp_path)
    loaded = []
    namespace = SimpleNamespace(
        load_kv_cache_block_gather_runtime=lambda path: loaded.append(path) or True,
        has_kv_cache_block_gather_runtime=lambda: True,
    )
    fake_torch = SimpleNamespace(ops=SimpleNamespace(_C_ascend=namespace))

    selected = activate_kv_cache_block_gather_runtime(
        fake_torch,
        package_dir=tmp_path,
    )

    assert selected == opapi
    assert loaded == [str(opapi)]


def test_packaging_uses_generated_vendor_name_and_checks_wheel_payload():
    repo_root = Path(__file__).resolve().parents[2]
    binding = (repo_root / "csrc/kv_cache_block_gather_binding.cpp").read_text(encoding="utf-8")
    setup = (repo_root / "setup.py").read_text(encoding="utf-8")

    assert "load_kv_cache_block_gather_runtime" in binding
    assert '"custom_transformer",' in setup
    assert '"libcust_opapi.so"' in setup
    assert '"aclnn_kv_cache_block_gather.h",' in setup
    assert '"kv_cache_block_gather.json",' in setup
    assert "Custom-op build did not produce the complete packaged" in setup
    assert "shutil.copytree(src_cann_ops_custom, dst_cann_ops_custom)" in setup
