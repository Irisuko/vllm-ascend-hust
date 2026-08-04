#ifndef KV_CACHE_BLOCK_GATHER_TILING_DATA_H
#define KV_CACHE_BLOCK_GATHER_TILING_DATA_H

#include <cstdint>

// Serialized ABI shared by op_host/tiling.cpp and the device kernel.  Changing
// field type, order, or alignment changes the bytes read by the kernel and must
// therefore be updated on both sides together.
struct KvCacheBlockGatherTilingData {
    // Number of (src block, dst block) mapping pairs.
    int64_t selectedBlocks;
    // Flattened element count in one KV block.
    int64_t elemsPerBlock;
    // Maximum elements copied through UB per loop trip.
    int64_t tileElems;
};

#endif
