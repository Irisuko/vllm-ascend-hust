#include "kv_cache_block_gather.h"

enum class KvCacheBlockGatherTilingKey : uint32_t {
    TILING_KEY_FLOAT = 1,
    TILING_KEY_FLOAT16 = 2,
    TILING_KEY_BF16 = 3,
    TILING_KEY_INT8 = 4,
};

template <uint32_t schMode>
__global__ __aicore__ void kv_cache_block_gather(
    GM_ADDR srcBlockIds, GM_ADDR srcPages, GM_ADDR dstBlockIds, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // Device entry point.  CANN appends workspace and tiling to the logical
    // tensor arguments.  The host has already selected schMode and serialized
    // KvCacheBlockGatherTilingData before this kernel is launched.
    REGISTER_TILING_DEFAULT(KvCacheBlockGatherTilingData);
    GET_TILING_DATA_WITH_STRUCT(KvCacheBlockGatherTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(KvCacheBlockGatherTilingKey::TILING_KEY_FLOAT)) {
        NsKvCacheBlockGather::KvCacheBlockGather<float> op;
        op.Init(srcBlockIds, srcPages, dstBlockIds, out, workspace, tiling, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(KvCacheBlockGatherTilingKey::TILING_KEY_FLOAT16)) {
        NsKvCacheBlockGather::KvCacheBlockGather<half> op;
        op.Init(srcBlockIds, srcPages, dstBlockIds, out, workspace, tiling, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(KvCacheBlockGatherTilingKey::TILING_KEY_BF16)) {
        NsKvCacheBlockGather::KvCacheBlockGather<bfloat16_t> op;
        op.Init(srcBlockIds, srcPages, dstBlockIds, out, workspace, tiling, &tilingData);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(KvCacheBlockGatherTilingKey::TILING_KEY_INT8)) {
        NsKvCacheBlockGather::KvCacheBlockGather<int8_t> op;
        op.Init(srcBlockIds, srcPages, dstBlockIds, out, workspace, tiling, &tilingData);
        op.Process();
    }
}
