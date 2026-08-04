# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
OP_ROOT = REPO_ROOT / "csrc" / "attention" / "kv_cache_block_gather"


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_int8_is_declared_by_graph_and_host_contracts() -> None:
    graph_proto = (OP_ROOT / "op_graph" / "kv_cache_block_gather_proto.h").read_text(encoding="utf-8")
    host_def = (OP_ROOT / "op_host" / "kv_cache_block_gather_def.cpp").read_text(encoding="utf-8")

    assert graph_proto.count("TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8})") == 2
    assert host_def.count("{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}") == 2


def test_int8_has_a_distinct_tiling_specialization() -> None:
    tiling = (OP_ROOT / "op_host" / "kv_cache_block_gather_tiling.cpp").read_text(encoding="utf-8")
    tiling_key = (OP_ROOT / "op_kernel" / "kv_cache_block_gather_tiling_key.h").read_text(encoding="utf-8")
    kernel = (OP_ROOT / "op_kernel" / "kv_cache_block_gather.cpp").read_text(encoding="utf-8")

    assert "payloadDtype == ge::DT_INT8" in tiling
    assert "GET_TPL_TILING_KEY(KV_CACHE_BLOCK_GATHER_INT8_MODE)" in tiling
    assert "#define KV_CACHE_BLOCK_GATHER_INT8_MODE 4" in tiling_key
    assert "KvCacheBlockGather<int8_t>" in kernel


def test_all_pages_require_datacopy_block_alignment() -> None:
    tiling = (OP_ROOT / "op_host" / "kv_cache_block_gather_tiling.cpp").read_text(encoding="utf-8")

    assert "constexpr int64_t DATA_COPY_BLOCK_BYTES = 32" in tiling
    assert "GetDataCopyAlignmentElems(payloadDtype)" in tiling
    assert "elemsPerBlock % alignmentElems != 0" in tiling


def test_torch_binding_accepts_signed_byte_pages() -> None:
    binding = _read("csrc/kv_cache_block_gather_binding.cpp")

    assert "src_pages.scalar_type() == at::ScalarType::Char" in binding
    assert "float32, float16, bfloat16, or int8" in binding
    assert "bytes_per_block % data_copy_block_bytes == 0" in binding
    assert "for block-aligned DataCopy" in binding


def test_torch_binding_rejects_cross_device_id_tensors_before_acl_conversion() -> None:
    binding = _read("csrc/kv_cache_block_gather_binding.cpp")

    src_check = "src_block_ids.device() == out.device()"
    dst_check = "dst_block_ids.device() == out.device()"
    first_descriptor = "ConvertType(src_block_ids)"
    assert src_check in binding
    assert dst_check in binding
    assert binding.index(src_check) < binding.index(first_descriptor)
    assert binding.index(dst_check) < binding.index(first_descriptor)
