#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_SRC_PAGES = 1;

// Graph inference runs on the host before the device kernel.  The gather does
// not change payload type, so the output dtype follows src_pages.  This rule is
// metadata for graph compilation, not part of the runtime copy path.
static ge::graphStatus InferDataTypeKvCacheBlockGather(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferDataTypeKvCacheBlockGather");
    context->SetOutputDataType(IDX_0, context->GetInputDataType(IDX_SRC_PAGES));
    OP_LOGD(context->GetNodeName(), "End InferDataTypeKvCacheBlockGather");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(KvCacheBlockGather).InferDataType(InferDataTypeKvCacheBlockGather);
} // namespace ops
