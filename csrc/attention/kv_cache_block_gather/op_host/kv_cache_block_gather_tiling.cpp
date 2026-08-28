#include <algorithm>
#include <cstdint>

#include "kv_cache_block_gather_tiling_data.h"
#include "kv_cache_block_gather_tiling_key.h"
#include "kv_cache_block_gather_compat.h"
#include "log/ops_log.h"
#include "register/op_impl_kernel_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

namespace {
// Host-side tiling is the launch planner.  It validates runtime metadata and
// serializes a small KvCacheBlockGatherTilingData record for the device kernel.
// It does not copy KV payloads itself.

// The current ACLNN contract asks the runtime for 16 MiB even though the
// kernel currently ignores its workspace pointer.  Treat this as part of the
// existing runtime contract; removing it requires separate CANN/ACLNN
// validation rather than an assumption that zero is always safe.
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
// Maximum payload elements moved through UB in one loop trip.  This is an
// element count, so the byte count depends on the payload dtype.
constexpr int64_t DEFAULT_TILE_ELEMS = 1024;
// Every specialization deliberately uses only block-aligned DataCopy
// transfers. Canonical KV pages satisfy this naturally; other layouts are
// rejected instead of relying on an implicit tail copy.
constexpr int64_t DATA_COPY_BLOCK_BYTES = 32;
constexpr uint32_t IDX_SRC_BLOCK_IDS = 0;
constexpr uint32_t IDX_SRC_PAGES = 1;
constexpr uint32_t IDX_DST_BLOCK_IDS = 2;
constexpr uint32_t IDX_OUT = 0;
} // namespace

struct KvCacheBlockGatherCompileInfo {
};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeInfo(
    gert::TilingContext* context, int64_t& selectedBlocks, int64_t& elemsPerBlock)
{
    // The kernel treats every KV page as one flat contiguous payload.  Only
    // dim 0 identifies pages; all remaining output dimensions are folded into
    // elemsPerBlock.
    auto srcBlockIdsShape = context->GetInputShape(IDX_SRC_BLOCK_IDS);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcBlockIdsShape);
    auto srcPagesShape = context->GetInputShape(IDX_SRC_PAGES);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcPagesShape);
    auto dstBlockIdsShape = context->GetInputShape(IDX_DST_BLOCK_IDS);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstBlockIdsShape);
    auto outShape = context->GetOutputShape(IDX_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    auto srcBlockIdsStorageShape = srcBlockIdsShape->GetStorageShape();
    auto srcPagesStorageShape = srcPagesShape->GetStorageShape();
    auto dstBlockIdsStorageShape = dstBlockIdsShape->GetStorageShape();
    auto outStorageShape = outShape->GetStorageShape();

    OP_CHECK_IF(srcBlockIdsStorageShape.GetDimNum() != 1,
        OP_LOGE(context, "src_block_ids must be 1D, got %zuD", srcBlockIdsStorageShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(dstBlockIdsStorageShape.GetDimNum() != 1,
        OP_LOGE(context, "dst_block_ids must be 1D, got %zuD", dstBlockIdsStorageShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outStorageShape.GetDimNum() < 1,
        OP_LOGE(context, "out must have at least 1 dimension"), return ge::GRAPH_FAILED);

    selectedBlocks = srcBlockIdsStorageShape.GetDim(0);
    OP_CHECK_IF(selectedBlocks <= 0, OP_LOGE(context, "selectedBlocks must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dstBlockIdsStorageShape.GetDim(0) != selectedBlocks,
        OP_LOGE(context, "dst_block_ids dim0 %ld must equal src_block_ids dim0 %ld",
            dstBlockIdsStorageShape.GetDim(0), selectedBlocks),
        return ge::GRAPH_FAILED);

    int64_t outElems = 1;
    for (size_t i = 0; i < outStorageShape.GetDimNum(); ++i) {
        outElems *= outStorageShape.GetDim(i);
    }
    int64_t dstBlocks = outStorageShape.GetDim(0);
    OP_CHECK_IF(dstBlocks <= 0, OP_LOGE(context, "out dim0 must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outElems % dstBlocks != 0,
        OP_LOGE(context, "out element count %ld is not divisible by out dim0 %ld", outElems, dstBlocks),
        return ge::GRAPH_FAILED);

    elemsPerBlock = outElems / dstBlocks;
    OP_CHECK_IF(elemsPerBlock <= 0, OP_LOGE(context, "elemsPerBlock must be positive"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(elemsPerBlock % 8 != 0,
        OP_LOGE(context, "elemsPerBlock %ld must be a multiple of 8 elements", elemsPerBlock),
        return ge::GRAPH_FAILED);

    int64_t srcElems = 1;
    for (size_t i = 0; i < srcPagesStorageShape.GetDimNum(); ++i) {
        srcElems *= srcPagesStorageShape.GetDim(i);
    }
    OP_CHECK_IF(srcElems < elemsPerBlock,
        OP_LOGE(context, "src_pages element count %ld must be at least one block of %ld elements",
            srcElems, elemsPerBlock),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(srcElems % elemsPerBlock != 0,
        OP_LOGE(context, "src_pages element count %ld must be divisible by elemsPerBlock %ld",
            srcElems, elemsPerBlock),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static bool IsSupportedPayloadDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 ||
        dtype == ge::DT_INT8;
}

static int64_t GetDataCopyAlignmentElems(ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT) {
        return DATA_COPY_BLOCK_BYTES / sizeof(float);
    }
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        return DATA_COPY_BLOCK_BYTES / sizeof(uint16_t);
    }
    return DATA_COPY_BLOCK_BYTES / sizeof(int8_t);
}

static ge::graphStatus CheckDtypes(gert::TilingContext* context, ge::DataType& payloadDtype)
{
    auto srcBlockIdsDesc = context->GetInputDesc(IDX_SRC_BLOCK_IDS);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcBlockIdsDesc);
    auto srcPagesDesc = context->GetInputDesc(IDX_SRC_PAGES);
    OP_CHECK_NULL_WITH_CONTEXT(context, srcPagesDesc);
    auto dstBlockIdsDesc = context->GetInputDesc(IDX_DST_BLOCK_IDS);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstBlockIdsDesc);
    auto outDesc = context->GetOutputDesc(IDX_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);

    OP_CHECK_IF(srcBlockIdsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(context, "src_block_ids dtype must be int32"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(dstBlockIdsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(context, "dst_block_ids dtype must be int32"), return ge::GRAPH_FAILED);

    payloadDtype = srcPagesDesc->GetDataType();
    OP_CHECK_IF(!IsSupportedPayloadDtype(payloadDtype),
        OP_LOGE(context, "src_pages dtype must be float, float16, bf16, or int8"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outDesc->GetDataType() != payloadDtype,
        OP_LOGE(context, "out dtype must match src_pages dtype"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetTilingKeyByDtype(ge::DataType payloadDtype)
{
    // Select one precompiled template specialization.  The device kernel does
    // not perform a runtime dtype branch for every copied element.
    if (payloadDtype == ge::DT_FLOAT16) {
        return GET_TPL_TILING_KEY(KV_CACHE_BLOCK_GATHER_FLOAT16_MODE);
    }
    if (payloadDtype == ge::DT_BF16) {
        return GET_TPL_TILING_KEY(KV_CACHE_BLOCK_GATHER_BF16_MODE);
    }
    if (payloadDtype == ge::DT_INT8) {
        return GET_TPL_TILING_KEY(KV_CACHE_BLOCK_GATHER_INT8_MODE);
    }
    return GET_TPL_TILING_KEY(KV_CACHE_BLOCK_GATHER_FLOAT_MODE);
}

static ge::graphStatus KvCacheBlockGatherTilingFunc(gert::TilingContext* context)
{
    // 1. Discover AIV resources and validate tensor metadata.
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);
    ge::DataType payloadDtype = ge::DT_UNDEFINED;
    OP_CHECK_IF(CheckDtypes(context, payloadDtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "CheckDtypes failed"), return ge::GRAPH_FAILED);

    int64_t selectedBlocks = 0;
    int64_t elemsPerBlock = 0;
    OP_CHECK_IF(GetShapeInfo(context, selectedBlocks, elemsPerBlock) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeInfo failed"), return ge::GRAPH_FAILED);
    int64_t alignmentElems = GetDataCopyAlignmentElems(payloadDtype);
    OP_CHECK_IF(elemsPerBlock % alignmentElems != 0,
        OP_LOGE(context, "elemsPerBlock %ld must be a multiple of %ld elements for block-aligned DataCopy",
            elemsPerBlock, alignmentElems),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    // 2. Write the host-to-kernel ABI record consumed through the `tiling`
    //    pointer in the AscendC entry point.
    KvCacheBlockGatherTilingData* tiling = context->GetTilingData<KvCacheBlockGatherTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(KvCacheBlockGatherTilingData), 0, sizeof(KvCacheBlockGatherTilingData)) != EOK,
        OP_LOGE(context, "set tiling data failed"), return ge::GRAPH_FAILED);
    tiling->selectedBlocks = selectedBlocks;
    tiling->elemsPerBlock = elemsPerBlock;
    tiling->tileElems = std::min<int64_t>(DEFAULT_TILE_ELEMS, elemsPerBlock);

    // 3. Launch at most one AIV core per selected block.  If there are more
    //    pairs than cores, the kernel's grid-stride loop handles the remainder.
    int64_t blockDim = std::min<int64_t>(coreNum, selectedBlocks);
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->SetTilingKey(GetTilingKeyByDtype(payloadDtype));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForKvCacheBlockGather(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KvCacheBlockGather)
    .Tiling(KvCacheBlockGatherTilingFunc)
    .TilingParse<KvCacheBlockGatherCompileInfo>(TilingParseForKvCacheBlockGather);

} // namespace optiling
