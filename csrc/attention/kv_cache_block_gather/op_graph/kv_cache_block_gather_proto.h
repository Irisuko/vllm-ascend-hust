#ifndef OPS_OP_PROTO_INC_KV_CACHE_BLOCK_GATHER_H_
#define OPS_OP_PROTO_INC_KV_CACHE_BLOCK_GATHER_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

// Graph-facing operator schema: this is the public contract seen by GE during
// graph construction/compilation.  It declares names and legal dtypes only;
// no data is moved here.  src_block_ids[i] and dst_block_ids[i] form one page
// mapping, while src_pages and out contain the page payloads.
REG_OP(KvCacheBlockGather)
    .INPUT(src_block_ids, TensorType({DT_INT32, DT_INT32, DT_INT32, DT_INT32}))
    .INPUT(src_pages, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .INPUT(dst_block_ids, TensorType({DT_INT32, DT_INT32, DT_INT32, DT_INT32}))
    .OUTPUT(out, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT8}))
    .OP_END_FACTORY_REG(KvCacheBlockGather)

} // namespace ge

#endif // OPS_OP_PROTO_INC_KV_CACHE_BLOCK_GATHER_H_
