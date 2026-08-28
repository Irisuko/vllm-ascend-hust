#ifndef KV_CACHE_BLOCK_GATHER_H
#define KV_CACHE_BLOCK_GATHER_H

#include "kernel_operator.h"
#include "kv_cache_block_gather_tiling_data.h"
#include "kv_cache_block_gather_tiling_key.h"

namespace NsKvCacheBlockGather {
using namespace AscendC;

// This file contains the actual device-side gather algorithm.  GlobalTensor
// wraps device-visible GM addresses; for srcPagesGM_ that address may refer to
// host RAM registered with ACL_HOST_REGISTER_MAPPED by the Torch binding.
constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KvCacheBlockGather {
public:
    __aicore__ inline KvCacheBlockGather() {}
    __aicore__ inline void Init(GM_ADDR srcBlockIds, GM_ADDR srcPages,
        GM_ADDR dstBlockIds, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling,
        const KvCacheBlockGatherTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyBlock(int64_t pairIdx);

private:
    // Separate VECIN and VECOUT queues preserve the conventional
    // CopyIn -> local copy -> CopyOut pipeline.  Although the middle copy looks
    // redundant, measurements in README.md show that TQueBind was slower on
    // this mapped-host gather path.  BUFFER_NUM=2 gives each stage ping-pong UB
    // storage; increasing it to 4 also regressed performance.
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;

    GlobalTensor<int32_t> srcBlockIdsGM_;
    GlobalTensor<int32_t> dstBlockIdsGM_;
    GlobalTensor<T> srcPagesGM_;
    GlobalTensor<T> outGM_;

    int64_t selectedBlocks_ = 0;
    int64_t elemsPerBlock_ = 0;
    int64_t tileElems_ = 0;
};

template <typename T>
__aicore__ inline void KvCacheBlockGather<T>::Init(
    GM_ADDR srcBlockIds, GM_ADDR srcPages, GM_ADDR dstBlockIds, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling,
    const KvCacheBlockGatherTilingData* tilingData)
{
    selectedBlocks_ = tilingData->selectedBlocks;
    elemsPerBlock_ = tilingData->elemsPerBlock;
    tileElems_ = tilingData->tileElems;
    // Workspace is present in the ACLNN launch ABI but is not consumed by the
    // current copy algorithm.  Host tiling still requests it, so do not infer
    // from this cast alone that workspace=0 is valid for every runtime.
    (void)workspace;
    (void)tiling;

    srcBlockIdsGM_.SetGlobalBuffer((__gm__ int32_t*)srcBlockIds, selectedBlocks_);
    dstBlockIdsGM_.SetGlobalBuffer((__gm__ int32_t*)dstBlockIds, selectedBlocks_);
    srcPagesGM_.SetGlobalBuffer((__gm__ T*)srcPages);
    outGM_.SetGlobalBuffer((__gm__ T*)out);

    pipe_.InitBuffer(inputQueue_, BUFFER_NUM, tileElems_ * sizeof(T));
    pipe_.InitBuffer(outputQueue_, BUFFER_NUM, tileElems_ * sizeof(T));
}

template <typename T>
__aicore__ inline void KvCacheBlockGather<T>::CopyBlock(int64_t pairIdx)
{
    // The two ID arrays are the page table for this operation.  One mapping
    // pair says: copy src_pages[srcPageIdx] into out[dstPageIdx].
    int32_t srcPageIdx = srcBlockIdsGM_.GetValue(pairIdx);
    int32_t dstPageIdx = dstBlockIdsGM_.GetValue(pairIdx);
    // GetValue issues a scalar-pipeline load from GM (possibly served by L2)
    // into a scalar register.  It needs no explicit UB buffer or TQue, but it
    // is still a real memory transaction, so this form suits small metadata;
    // bulk payload movement should use DataCopy through UB instead.
    int64_t srcBase = static_cast<int64_t>(srcPageIdx) * elemsPerBlock_;
    int64_t dstBase = static_cast<int64_t>(dstPageIdx) * elemsPerBlock_;

    // Move one flat block in tileElems-sized chunks:
    //   mapped host/device GM -> input UB -> output UB -> NPU GM.
    int64_t copied = 0;
    while (copied < elemsPerBlock_) {
        int64_t remain = elemsPerBlock_ - copied;
        int64_t copyElems = remain < tileElems_ ? remain : tileElems_;

        LocalTensor<T> inLocal = inputQueue_.AllocTensor<T>();
        DataCopy(inLocal, srcPagesGM_[srcBase + copied], copyElems);
        inputQueue_.EnQue(inLocal);

        inLocal = inputQueue_.DeQue<T>();
        LocalTensor<T> outLocal = outputQueue_.AllocTensor<T>();
        DataCopy(outLocal, inLocal, copyElems);
        outputQueue_.EnQue(outLocal);
        inputQueue_.FreeTensor(inLocal);

        outLocal = outputQueue_.DeQue<T>();
        DataCopy(outGM_[dstBase + copied], outLocal, copyElems);
        outputQueue_.FreeTensor(outLocal);

        copied += copyElems;
    }
}

template <typename T>
__aicore__ inline void KvCacheBlockGather<T>::Process()
{
    // Grid-stride distribution across AIV cores.  For example, with 40 cores,
    // core 0 handles pairs 0, 40, 80, ... and core 1 handles 1, 41, 81, ... .
    for (int64_t pair = static_cast<int64_t>(AscendC::GetBlockIdx());
         pair < selectedBlocks_;
         pair += static_cast<int64_t>(AscendC::GetBlockNum())) {
        CopyBlock(pair);
    }
}

} // namespace NsKvCacheBlockGather

#endif
