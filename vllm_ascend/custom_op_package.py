"""Locate custom-op artifacts bundled with the vLLM Ascend package.

Keep this module dependency-free: packaging checks and early worker startup may
need it before torch/CANN initialization is safe.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

CUSTOM_OP_VENDOR_NAME = "custom_transformer"
CUSTOM_OPAPI_RELATIVE_PATH = Path("op_api/lib/libcust_opapi.so")
CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH = Path("op_api/include/aclnnop/aclnn_kv_cache_block_gather.h")
CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR = Path("op_impl/ai_core/tbe/kernel/config")
CUSTOM_OP_GATHER_KERNEL_MANIFEST = "kv_cache_block_gather.json"
CUSTOM_OPP_ENV = "ASCEND_CUSTOM_OPP_PATH"


@dataclass(frozen=True)
class CustomOpPackageResolution:
    """Result of resolving the opapi used by in-tree custom operators."""

    available: bool
    vendor_path: Path | None
    opapi_library: Path | None
    source: str
    reason: str


def bundled_custom_op_vendor_path(package_dir: str | Path | None = None) -> Path:
    """Return the expected vendor root for a source, editable, or wheel install."""
    base_dir = Path(__file__).resolve().parent if package_dir is None else Path(package_dir)
    return base_dir / "_cann_ops_custom" / "vendors" / CUSTOM_OP_VENDOR_NAME


def resolve_custom_op_package(
    *,
    package_dir: str | Path | None = None,
) -> CustomOpPackageResolution:
    """Resolve the custom-op package bundled with vLLM Ascend."""
    vendor_path = bundled_custom_op_vendor_path(package_dir)
    if not vendor_path.is_dir():
        return CustomOpPackageResolution(
            available=False,
            vendor_path=None,
            opapi_library=None,
            source="bundled",
            reason=f"bundled custom-op vendor directory is missing: {vendor_path}",
        )

    opapi_library = vendor_path / CUSTOM_OPAPI_RELATIVE_PATH
    if not opapi_library.is_file():
        return CustomOpPackageResolution(
            available=False,
            vendor_path=vendor_path,
            opapi_library=None,
            source="bundled",
            reason=f"bundled custom-op opapi library is missing: {opapi_library}",
        )

    gather_header = vendor_path / CUSTOM_OP_GATHER_HEADER_RELATIVE_PATH
    if not gather_header.is_file():
        return CustomOpPackageResolution(
            available=False,
            vendor_path=vendor_path,
            opapi_library=None,
            source="bundled",
            reason=f"bundled kv_cache_block_gather opapi header is missing: {gather_header}",
        )

    kernel_config_dir = vendor_path / CUSTOM_OP_GATHER_KERNEL_CONFIG_RELATIVE_DIR
    gather_manifests = kernel_config_dir.glob(f"*/{CUSTOM_OP_GATHER_KERNEL_MANIFEST}")
    if not any(path.is_file() for path in gather_manifests):
        return CustomOpPackageResolution(
            available=False,
            vendor_path=vendor_path,
            opapi_library=None,
            source="bundled",
            reason=(f"bundled kv_cache_block_gather kernel manifest is missing under: {kernel_config_dir}"),
        )

    return CustomOpPackageResolution(
        available=True,
        vendor_path=vendor_path,
        opapi_library=opapi_library,
        source="bundled",
        reason=f"using bundled custom-op package: {vendor_path}",
    )


def _prepend_env_path(environ: MutableMapping[str, str], name: str, path: Path) -> None:
    path_str = os.fspath(path)
    entries = [entry for entry in environ.get(name, "").split(os.pathsep) if entry]
    if path_str not in entries:
        entries.insert(0, path_str)
        environ[name] = os.pathsep.join(entries)


def bootstrap_custom_op_package_env(
    *,
    package_dir: str | Path | None = None,
    include_vendor_lib: bool = False,
    environ: MutableMapping[str, str] | None = None,
) -> CustomOpPackageResolution:
    """Expose bundled kernels to CANN and return the capability decision."""
    target_environ = os.environ if environ is None else environ
    resolution = resolve_custom_op_package(package_dir=package_dir)

    if resolution.vendor_path is not None:
        _prepend_env_path(target_environ, CUSTOM_OPP_ENV, resolution.vendor_path)
        vendor_lib = resolution.vendor_path / "op_api" / "lib"
        if include_vendor_lib and vendor_lib.is_dir():
            _prepend_env_path(target_environ, "LD_LIBRARY_PATH", vendor_lib)

    return resolution


def activate_kv_cache_block_gather_runtime(
    torch_module,
    *,
    opapi_library: str | Path | None = None,
    package_dir: str | Path | None = None,
) -> Path:
    """Load the gather OPAPI library through the registered Torch adapter.

    ``opapi_library`` is a development-only argument used by the benchmark and
    smoke test. Production callers omit it and use the wheel-bundled package.
    No process-wide user configuration or silent fallback is involved.
    """
    resolution = bootstrap_custom_op_package_env(package_dir=package_dir)
    if opapi_library is None:
        if not resolution.available or resolution.opapi_library is None:
            raise RuntimeError(resolution.reason)
        selected_library = resolution.opapi_library
    else:
        selected_library = Path(opapi_library).expanduser().resolve()
        if not selected_library.is_file():
            raise RuntimeError(f"kv_cache_block_gather OPAPI library is not a file: {selected_library}")

    namespace = getattr(torch_module.ops, "_C_ascend", None)
    loader = None if namespace is None else getattr(namespace, "load_kv_cache_block_gather_runtime", None)
    capability = None if namespace is None else getattr(namespace, "has_kv_cache_block_gather_runtime", None)
    if loader is None or capability is None:
        raise RuntimeError("vllm_ascend extension is missing the kv_cache_block_gather runtime loader")
    if not loader(os.fspath(selected_library)) or not capability():
        raise RuntimeError(f"custom-op library does not expose kv_cache_block_gather: {selected_library}")
    return selected_library
