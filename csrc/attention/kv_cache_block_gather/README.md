# KV Cache Block Gather

> [!IMPORTANT]
> 这是一个 **experimental、显式启用、fail-fast** 的 A2/A3 CPU KV restore
> 路径。默认 `CPUOffloadingSpec` 行为不变；算子只接受由显式 host-pool lease
> 注册并在异步执行期间保持存活的 worker-local CPU tensor。

本文记录 `kv_cache_block_gather` 的设备侧搬运流水实验：我们尝试用
`TQueBind` 去掉显式的 UB→UB 拷贝，但实测性能反而下降；随后又对双队列的
`BUFFER_NUM` 做了小范围 sweep。这里把实验、结论和最终选择留在算子旁边，
避免以后仅凭“少一次拷贝一定更快”的直觉重复走一遍。

## 算子的数据路径

算子根据两组 block ID 完成离散 gather：

```text
out[dst_block_ids[i]] = src_pages[src_block_ids[i]]
```

设备 kernel 的 `src_pages` GM ABI 本身不区分 device HBM 与 device-visible
host 映射；但本实验的 Torch adapter 刻意只接受经
`aclrtHostRegister(..., ACL_HOST_REGISTER_MAPPED)` 注册、并持有显式 lease 的
CPU tensor。每个 block 会被切成最多 1024 个元素的 tile，由多个 AIV
core 按 grid-stride 方式处理。

当前 kernel 使用传统的三段流水：

```text
mapped-host/device GM -> VECIN UB -> VECOUT UB -> device GM
                          CopyIn      local copy      CopyOut
```

对应两个独立队列：

```cpp
TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
```

## 为什么曾经考虑 `TQueBind`

这个 gather 不做 Add、Cast 等数学运算，中间只有：

```cpp
DataCopy(outLocal, inLocal, copyElems);
```

从数据语义看，这次 UB→UB 拷贝确实是恒等操作。因此一个很自然的简化是：

```cpp
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue_;
```

让同一个 `LocalTensor` 从 VECIN 直接交给 VECOUT，数据路径缩短为：

```text
mapped-host/device GM -> bound UB -> device GM
```

`TQueBind` 仍然表达 MTE2→MTE3 的事件依赖，但不再分配独立的输入、输出
UB，也没有显式的 local-to-local `DataCopy`。CANN 自带的纯搬运实现中也能
看到这种写法，所以它在语义和 API 使用上都是成立的，不是 correctness
问题。

## 实验方法

使用 [`tools/benchmark_kv_gather_vs_span.py`](../../../tools/benchmark_kv_gather_vs_span.py)
测量，主要配置如下：

- Ascend 910B2，CANN 9.0；
- `float16`；
- 512 个 selected blocks，K/V 两个 part；
- block 大小为 4 KiB、16 KiB、64 KiB；
- 每种 block 大小覆盖 8 种 span 碎片程度；
- 每个 case warmup 5 次、正式测量 30 次；
- 每个版本都运行 `span-first` 和 `mapped-first` 两种 backend 顺序；
- gather-vs-span 的最终对比让两个 backend 共享同一个 PyTorch pinned host
  allocation，避免 pageable copy staging 扭曲 baseline；
- host 注册时间排除在 steady-state kernel 时间之外；
- 每个 case 校验完整的非零输出。

这里的 span baseline 是直接对每个 CPU/NPU 连续 block-pair run 调用一次
`copy_` 的 **Python per-span microbenchmark**。它用于隔离碎片化的搬运开销，
不是当前生产 connector 的 native transfer backend，也不等价于其
staging+scatter 路径，不能单独预测高并发服务吞吐。

下面的队列布局 sweep 来自原型阶段的普通 CPU allocation + 显式 host
registration，作用是比较同一 backing 下不同 kernel 结构的相对差异；它不作为
production pinned-pool 的绝对带宽数据。文末 gather-vs-span 数值则已使用当前
工具的 pinned allocation 重新测量。

下表使用两种 backend 顺序、8 种 mapping 的 **mapped-gather device event
mean** 聚合值。百分比以原双队列 `BUFFER_NUM=1` 为基准；负数表示回退。

## `TQueBind` sweep 结果

| 实现 | `BUFFER_NUM` | 4 KiB event ms | 相对基线 | 16 KiB event ms | 相对基线 | 64 KiB event ms | 相对基线 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 双队列 + UB copy | 1 | 0.2670 | 基线 | 0.7998 | 基线 | 2.9343 | 基线 |
| `TQueBind` | 1 | 0.3119 | -14.4% | 0.8332 | -4.0% | 2.9820 | -1.6% |
| `TQueBind` | 2 | 0.2912 | -8.3% | 0.8104 | -1.3% | 2.9566 | -0.8% |
| `TQueBind` | 4 | 0.2870 | -7.0% | 0.8212 | -2.6% | 2.9646 | -1.0% |

结论：

1. `TQueBind<..., 1>` 的确受单 buffer 限制，特别是 4 KiB 小块；
2. `BUFFER_NUM=2` 补回了大部分损失，是最均衡的 `TQueBind` 选择；
3. 增加到 4 没有稳定收益；
4. 即使选择 2，`TQueBind` 仍然没有超过原来的双队列。

wall-clock 聚合值呈现相同方向，因此这个结论不是单纯由 event/wall
计时口径差异造成的。

## 为什么最终不用 `TQueBind`

“少搬一次数据”并不等于“关键路径更短”。`TQue`/`TQueBind` 不只是 UB
内存容器，还描述生产者、消费者之间的流水位置、事件同步和 buffer
生命周期。当前结果说明：

- 独立 VECIN/VECOUT 队列形成的流水，对这个 mapped-host 搬运路径更有利；
- 中间 UB copy 的局部带宽成本很低，可能被其他搬运阶段部分隐藏；
- `TQueBind` 改变了 MTE2→MTE3 的依赖路径，省掉 payload copy 的收益不足以
  抵消新的同步或串行化成本；
- block 越大，mapped-host/GM 外部带宽越占主导，所以两者差距缩小到约 1%。

上面关于流水和事件的解释是由结果推导出的工作假设，还不是 profiler
证明的微架构归因。但“`TQueBind` 在当前目标环境更慢”已经由两种 backend
顺序和完整 correctness 校验重复确认，因此工程决策不需要等待更昂贵的
profiling。

## 双队列 `BUFFER_NUM` sweep

恢复双队列后，又测试了每个队列 1、2、4 个 buffer：

| 双队列 `BUFFER_NUM` | 4 KiB event ms | 相对 1 | 16 KiB event ms | 相对 1 | 64 KiB event ms | 相对 1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2670 | 基线 | 0.7998 | 基线 | 2.9343 | 基线 |
| **2** | **0.2653** | **+0.7%** | **0.8007** | **-0.1%** | **2.8886** | **+1.6%** |
| 4 | 0.2914 | -8.4% | 0.8149 | -1.9% | 2.9421 | -0.3% |

`BUFFER_NUM=2` 在 4 KiB 基本持平、16 KiB 属于测量噪声范围，在 64 KiB
则在两种 backend 顺序中都小幅领先。增加到 4 明显没有价值，4 KiB 反而
出现较大回退。因此最终保留：

```cpp
constexpr int32_t BUFFER_NUM = 2;
TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
```

双队列的 payload UB 占用近似为：

```text
2 queues * BUFFER_NUM * tileElems * sizeof(T)
```

`tileElems=1024`、`float16` 时，`BUFFER_NUM=2` 每个 core 使用约 8 KiB
payload UB；相对于本算子的 mapped-host 访问成本，这个空间换取小幅流水收益
是可以接受的。

## 最终决策

- 保留独立 `inputQueue_` 和 `outputQueue_`；
- 保留显式 UB→UB `DataCopy`，不要仅凭代码观感删除；
- 使用 `BUFFER_NUM=2`；
- 不把 `TQueBind` 版本作为性能优化合入；
- 如果未来更换 CANN、SoC 或 tile 策略，应重新测量，而不是把本结论当成所有
  Ascend 平台的永久规律。

## 集成状态与端到端结论

在 production-shaped pinned host allocation 上重新运行最终矩阵后，512 个完全
离散的 16 KiB block（K/V 两个 part）中，mapped gather 的 wall-clock 有效带宽
约为 17.83 GB/s，逐 span copy 约为 0.43 GB/s（约 41x）；当 512 个 block
完全连续、只需要一次 span copy 时，两者约为 18.04 GB/s 与 14.97 GB/s
（约 1.2x）。数值取 `span-first` 和 `mapped-first` 两种顺序的 mean 平均值，
registration 不计入 steady state。该结果只说明碎片严重时能消除大量小 copy
提交，并不代表服务吞吐会同比增长。

在高并发 Mooncake trace 的端到端 A/B 中，两条路径分别为 208.77 s 与
208.89 s，基本持平：并发 copy 能与 decode 重叠，restore 并非稳定关键路径。
因此本实现进入主线的定位是可复现的研究基础设施，而不是默认性能优化。

显式选择方式：

```text
spec_module_path = vllm_ascend.kv_offload.npu
spec_name = MappedOffloadingSpec
```

当前只支持 `block_size_factor == 1`；缺少 packaged operator、页面布局不兼容或
host registration 失败都会在 handler 构造阶段报错，不会静默回退到 copy。
