#include "kv_cache_block_gather_compat.h"
#include "log/ops_log.h"
#include "register/op_impl_registry.h"

namespace ops {
static constexpr int64_t IDX_0 = 0;

// The current integration is out-parameter based: the Torch binding allocates
// `out` and passes its shape into ACLNN.  Consequently there is no new shape to
// derive here; this hook only verifies that an output shape is present.  A
// functional graph API that allocated its own output would need a real shape
// relation in this function.
static ge::graphStatus InferShapeKvCacheBlockGather(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferShapeKvCacheBlockGather");

    const gert::Shape* outShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    OP_LOGD(context->GetNodeName(), "End InferShapeKvCacheBlockGather");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCacheBlockGather).InferShape(InferShapeKvCacheBlockGather);
} // namespace ops
