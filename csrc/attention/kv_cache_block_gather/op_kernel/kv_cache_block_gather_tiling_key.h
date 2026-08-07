#ifndef KV_CACHE_BLOCK_GATHER_TILING_KEY_H
#define KV_CACHE_BLOCK_GATHER_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// Tiling keys identify separately compiled dtype specializations.  The host
// selects one key after inspecting src_pages; the kernel entry then instantiates
// the corresponding C++ payload type.
#define KV_CACHE_BLOCK_GATHER_FLOAT_MODE 1
#define KV_CACHE_BLOCK_GATHER_FLOAT16_MODE 2
#define KV_CACHE_BLOCK_GATHER_BF16_MODE 3
#define KV_CACHE_BLOCK_GATHER_INT8_MODE 4

ASCENDC_TPL_ARGS_DECL(KvCacheBlockGather,
    ASCENDC_TPL_UINT_DECL(schMode, 3, ASCENDC_TPL_UI_LIST, KV_CACHE_BLOCK_GATHER_FLOAT_MODE,
        KV_CACHE_BLOCK_GATHER_FLOAT16_MODE, KV_CACHE_BLOCK_GATHER_BF16_MODE,
        KV_CACHE_BLOCK_GATHER_INT8_MODE)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, KV_CACHE_BLOCK_GATHER_FLOAT_MODE,
            KV_CACHE_BLOCK_GATHER_FLOAT16_MODE, KV_CACHE_BLOCK_GATHER_BF16_MODE,
            KV_CACHE_BLOCK_GATHER_INT8_MODE)));

#endif
