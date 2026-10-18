# Cosmos3 Reasoner Decode 优化与 T2V Green Context 共置说明

本文档记录当前仓库中针对 Cosmos3 Reasoner 自回归 decode 的优化、Green Context（GC）分区方式、decode 与 T2V 共置 runtime、测量口径、正确性边界、已发现的问题以及复现实验的方法。

本文档的目的不是只给出一组命令，而是解释：

1. Cosmos3 原版 decode 为什么慢；
2. 当前优化具体改变了哪些计算与状态；
3. 为什么优化后的 scaling 不能与原版 eager scaling 直接比较；
4. decode 和 T2V 如何真正进入两个互不重叠的 GC stream；
5. 哪些历史结果有效、哪些结果已经因实现错误作废；
6. 如何以正确性可验证的方式继续做 scaling 和 colocation。

## 1. 当前实验目标

目标 workload 是同一个 Cosmos3-Edge 常驻模型上的两个路径：

- Reasoner：prompt prefill 后的 autoregressive text decode；
- Generator：完整 T2V，或其中的 diffusion sampling / VAE decode 阶段。

实验希望将两个 workload 放入同一张 GPU 上两个不重叠的 Green Context：

```text
GPU 170 SM
├── Generator GC stream：例如 160 SM
└── Reasoner GC stream：例如 10 SM
```

测量问题包括：

- decode 在不同实际 SM 数量下的 scaling；
- T2V 在指定分区下的 alone 时间；
- 两者同时执行时各自的 slowdown；
- 共置 makespan 相对同分区串行执行的吞吐收益；
- 输出 token 是否保持一致。

这里的重点是同一模型实例、同一进程、两个 CPU worker、两个 GC stream。不是两个独立模型进程，也不是将权重复制两份。

## 1.1 实现与实验版本演进

为了避免把不同实现、不同 warm 状态和曾经的错误 CUDA Graph
结果混在一起，当前实验应至少区分以下版本。

| 版本 | Decode 实现 | 并发状态 | 主要结论 |
|---|---|---|---|
| V0 | Cosmos3 原生 eager | pipeline 数据依赖尚未拆开 | 同一请求的 Reasoner 后接 Generator，本质串行；不能用于评价 GPU 共置 |
| V1 | Cosmos3 原生 eager | 初次/冷状态，且早期存在 FlashAttention call_once 等问题 | 曾表现为近似串行或 decode 大幅膨胀；不是干净的稳态 launch-overhead 结论 |
| V1-warm | Cosmos3 原生 eager | call_once 处理后，先 warmup，再测量 | 已经能够较好重合；说明冷启动和一次性初始化是早期异常的重要部分 |
| V2-invalid | static KV + CUDA Graph | graph capture 未显式绑定 GC stream | Graph 实际跑在 170-SM primary context；所有该阶段 GC scaling/共置结果作废 |
| V3 | static KV + CUDA Graph | 显式 capture/replay 于 GC stream | Decode 时间大幅下降；T2V 共置 slowdown 仍很小 |
| V4 | V3 + persistent graph + non-fixed-length 控制路径 | 当前主实验 | 加入 token hash、alone、warmup 和实际 SM 校验；用于当前 scaling 和完整 T2V 共置 |

### 1.1.1 V1：原生 Cosmos3 eager decode 的早期共置

原生路径包含 dynamic torch.cat KV、imaginaire_attention 和 Python
逐 token 提交 kernel。

早期实验曾出现 decode colocated 约 44--72 s、T2V colocated 约
35--44 s。从表面上看，两侧接近串行。当时确实可以从 Nsys 中看到较大的
launch gap 和 host-side 等待，但这批结果同时混有：

1. 最初 pipeline 使用了同一请求内部的 Reasoner → Generator 数据依赖；
2. FlashAttention 首次调用的 call_once/初始化会跨线程阻塞；
3. T2V 首次执行还包含 compile、autotune、allocator/cache materialization 等冷启动；
4. CUDA_DEVICE_MAX_CONNECTIONS=1 会进一步限制可并行提交；从 1 调到 8
   有约数秒改善，而 8 到 32 收益很小。

所以不能把早期近似串行全部归因于原生 decode 的 kernel 太碎或 launch
开销太大。Launch 开销确实存在，但早期数据不是只改变 launch
机制的受控实验。

### 1.1.2 原生 eager 是否测过 warm

测过。保留日志：

    profiling_results/colocation_stage_sweep/eager_warm_sm136_diffusion_sampling.log

实际分区为 Reasoner 34 SM、Generator 136 SM。同一次运行的首次
colocated warmup 为：

    reasoner_decode = 44.887 s
    generator       = 36.256 s

随后 partitioned-alone 与正式 warm colocation：

    decode alone       = 28.357 s
    decode colocated   = 29.651 s
    decode slowdown    = 1.046x

    T2V alone          = 25.089 s
    T2V colocated      = 25.925 s
    T2V slowdown       = 1.033x

    throughput gain    = 1.706x

另一个以 VAE decode 为目标阶段的 warm 日志也给出相同方向：

    profiling_results/colocation_stage_sweep/eager_warm_sm136_vae_decode.log

    decode slowdown = 0.974x
    T2V slowdown    = 1.022x

这里小于 1 的 slowdown 应理解为单轮噪声，不是共置加速。

这组结果说明：

- T2V 冷启动与早期异常有关：第一次 generator 为 36.256 s，warm measured
  为 25.925 s；
- 第一次并发时 decode 也从 44.887 s 降到 warm measured 的 29.651 s，
  因而不只是 T2V 自身计时变短；
- 完成 warmup 和 call_once 修复后，原生 eager 在稳态下也能良好重合；
- 若要定量分解 cold T2V 对 launch gap 的贡献，仍需做同一二进制、同一分区下
  cold/warm × alone/colocated 的四格实验。现有数据只能说明相关，不能把全部
  差异因果归于 T2V 冷启动。

### 1.1.3 V3/V4：引入 static KV 和 CUDA Graph

优化后的 full-SM decode 从原生 eager 的约 25.8 s 降至约 6.1 s。这个
收益不是 CUDA Graph 单独贡献，而是 dynamic KV → static KV、
imaginaire_attention → flash_attn_with_kvcache，以及 Python 逐 token
launch → CUDA Graph replay 三者的合并效果。

在正确绑定 GC stream 后，34:136 分区的阶段共置结果为：

    profiling_results/colocation_stage_sweep/gc_graph_fixed_stream_g136_diffusion_sampling.log

    decode alone                  = 10.545 s
    decode colocated              = 11.505 s
    decode slowdown               = 1.091x
    T2V alone                     = 25.137 s
    T2V colocated                 = 25.787 s
    T2V slowdown                  = 1.026x
    diffusion sampling slowdown   = 1.035x

因此 CUDA Graph 大幅缩短了 decode，但没有让 T2V 的共置干扰明显恶化：
T2V 仍只变慢约 2%--3%。同一 34:136 分区下，因为 decode 从约 28 s
缩短至约 10.5 s，它不再覆盖完整 T2V，所以吞吐收益数字不能与 eager
的 1.706x 直接横向比较；需要重新调小 Reasoner 分区进行时间配平。

当前 10:160 配平实验为：

    decode alone       = 29.731 s
    decode colocated   = 34.115 s
    decode slowdown    = 1.147x
    T2V alone          = 22.253 s
    T2V colocated      = 22.816 s
    T2V slowdown       = 1.025x
    throughput gain    = 1.524x

这说明在仅给 decode 10 SM、让它覆盖完整 T2V 的激进配置下，主要代价落在
decode 一侧；T2V 仍只受到约 2.5% 的影响。

### 1.1.4 Scaling 平台期如何变化

原生 eager 的历史图中，Reasoner decode 在很低 SM ratio 后就接近 full-SM
性能，表现出很长的平台区。这与逐 token host launch、dynamic KV 和较低
GPU 并行度占比较大相符。

当前 static-KV + graph 曲线为：

    10 SM   29.689 s   Tfull/Tsm = 0.206
    34 SM   10.574 s   Tfull/Tsm = 0.579
    66 SM    7.059 s   Tfull/Tsm = 0.867
    98 SM    6.324 s   Tfull/Tsm = 0.967
    170 SM   6.117 s   Tfull/Tsm = 1.000

也就是说优化后 decode 对 SM 数量更敏感，约到 98 SM 才进入接近 full-SM
的平台，平台在横轴上的长度明显缩短。应将它解释为：

- graph/static KV 消除了大量与 SM 数量无关的 host/KV 管理开销；
- 剩余 GPU 工作在总时间中的占比提高，因此 SM scaling 更明显；
- 这不是性能退化：任意相同 SM 点的绝对 decode 时间仍显著缩短。

需要注意，原生 eager 与 graph 路径的 attention/KV 实现不同，因此该曲线
变化不能全部归因于 launch overhead。要严格拆分，需要后文所述的
static-KV eager 中间基线。

## 2. Cosmos3 原版 Reasoner decode

核心代码位于：

```text
/workspace/cosmos-framework/cosmos_framework/model/generator/mot/unified_mot.py
```

Reasoner generation 分为两个阶段：

```text
prompt / multimodal input
          │
          ▼
      Reasoner prefill
          │  生成第一枚 next token，并建立每层 KV
          ▼
 autoregressive decode loop
          │  每轮输入一个 token
          ▼
       generated tokens
```

### 2.1 原版 KV cache

`ReasonerKVCache` 每层保存：

```text
keys[layer]   [B, T, H_kv, D]
values[layer] [B, T, H_kv, D]
```

原版 dynamic cache 在每一层、每一个新 token 上执行：

```python
cached_k = torch.cat([cached_k, k], dim=1)
cached_v = torch.cat([cached_v, v], dim=1)
```

然后把完整的 `k_full/v_full` 交给 `imaginaire_attention`。

### 2.2 imaginaire_attention 不是朴素 attention

`imaginaire_attention` 是 Cosmos/Imaginaire4 的统一 frontend，而不是单个 kernel。它根据 GPU 架构、shape、dtype、causal/varlen、GQA 等条件自动选择 fused backend：

```text
cuDNN SDPA
FlashAttention 2
FlashAttention 3
FlashAttention 4
NATTEN
```

在当前 RTX 5090（SM120）上，默认优先级是：

```text
cuDNN → FlashAttention 2 → NATTEN
```

因此原版 attention kernel 本身已经是优化实现。原版 decode 的主要额外问题在 attention 之前：dynamic KV 每步改变地址、重复分配并复制整个历史 KV。

### 2.3 原版设计的合理性

dynamic KV 是通用性和正确性优先的实现：

- 每请求独立 cache，天然支持并发隔离；
- 不需要提前知道最终生成长度；
- EOS early-exit 自然；
- prompt 长度和 batch shape 可以动态变化；
- attention backend 可以统一调度和 fallback；
- 代码简单，适合作为 reference path。

代价是：

- 每层、每 token 的 `torch.cat`；
- allocator 和 KV 复制开销；
- tensor 地址变化，不能直接进行完整 decode-step CUDA Graph capture；
- Python 每 token 提交大量 kernel，可能出现 launch gap。

## 3. 当前 decode 优化

优化只作用于 Reasoner decode。以下路径保持不变：

- multimodal/text prefill；
- 第一枚 token 的 logits 和选择；
- T2V diffusion/generator pathway；
- VAE；
- 模型权重。

从第二枚生成 token 开始，优化路径使用 static KV 和 CUDA Graph。

### 3.1 Static KV cache

prefill 完成后，把 dynamic KV 转为固定容量：

```text
capacity = prefill_seq_len + max_new_tokens - 1
```

每层预分配：

```text
[B, capacity, H_kv, D]
```

同时维护 device-side `cache_seqlens`。decode attention 改为：

```python
flash_attn_with_kvcache(
    q,
    key_cache,
    value_cache,
    k=k,
    v=v,
    cache_seqlens=cache.cache_seqlens,
)
```

它会原地写入新 K/V，并只读取 `cache_seqlens` 指定的有效前缀。

### 3.2 CUDA Graph 捕获内容

一张 graph 捕获一枚 token 的完整 recurrent step：

```text
static_input token
    → reasoner transformer forward
    → LM head
    → argmax
    → EOS / finished bookkeeping
    → 更新 static_input
    → cache_seqlens += 1
    → position_ids += 1
```

后续 token 通过 `graph.replay()` 执行，并把结果从固定的 `static_input` 写入预分配的 `output_buf`。

### 3.3 Persistent graph runtime

只缓存 `CUDAGraph` 是错误的，因为 graph 绑定捕获时的 tensor 地址。当前 `_PersistentReasonerGraph` 同时持有：

- graph executable；
- 固定地址的每层 KV buffer；
- `static_input`；
- `static_position_ids`；
- `finished`；
- pad/EOS tensor；
- initial sequence length 和 capacity。

缓存 key 包含：

- model 和 causal LM identity；
- device；
- CUDA/GC stream pointer；
- batch/token shape；
- prefill length；
- capacity；
- position shape；
- EOS ids。

每个新请求仍先执行正常 prefill。进入 persistent runtime 后，将该请求的新 KV 前缀复制到固定 KV：

```text
new prefill KV → persistent static KV [:prefill_len]
```

然后重置所有 recurrent 状态，再 replay。旧请求超出当前 `cache_seqlens` 的数据不会被 attention 读取。

### 3.4 为什么这不只是“加了 CUDA Graph”

优化前后不是同一个 kernel 序列：

```text
原版 eager
  dynamic torch.cat KV
  + imaginaire_attention（通常 cuDNN/FA2）
  + Python 逐 token launch

当前 graph 路径
  static KV
  + flash_attn_with_kvcache
  + fixed-address recurrent state
  + graph replay
```

所以旧 eager scaling 曲线不能直接预测新 graph 路径。当前 fixed-length/full-SM 示例中：

```text
eager full-SM decode：约 25.8 s
graph full-SM decode：约 6.1 s
```

这个差异包含 static KV、KV 原地写入、attention backend 和 graph replay 的共同收益。要分离每一项，需要额外实现“static KV + eager token loop”中间基线。

## 4. CUDA Graph 与 Green Context 的关键修复

### 4.1 曾经的错误

最初代码位于 GC stream 的外层 context，但捕获写成：

```python
with torch.cuda.graph(graph, capture_error_mode="thread_local"):
    ...
```

PyTorch 在没有显式传入 `stream=` 时，不会沿用外层 current stream，而会切换到进程级默认 capture side stream。该 stream 属于 primary context。

结果是：看起来在 Reasoner GC 中调用 graph，实际 kernel 使用整张 GPU。NCU 证据：

```text
Uses Green Context = 0
# SMs = 170
Context 1
```

因此在该修复之前得到的所有 CUDA-Graph GC scaling、共置 slowdown 和 throughput 结论全部作废。非 graph eager GC 结果不受此问题影响。

### 4.2 当前正确实现

捕获必须显式指定当前 GC stream：

```python
capture_stream = torch.cuda.current_stream()
with torch.cuda.graph(
    graph,
    stream=capture_stream,
    capture_error_mode="thread_local",
):
    ...
```

验收 NCU 报告必须满足：

```text
Uses Green Context = 1
# SMs = 实际 Reasoner 分区大小
```

任何显示 170 SM / GC=0 的 report 都不能作为 GC 结果。

### 4.3 为什么使用 thread_local capture mode

Reasoner capture 期间，另一个 CPU thread 可能向 T2V stream 提交 CUDA 工作。CUDA 默认 global capture error mode 会因为其他线程的 CUDA activity 令 capture invalidated。

当前使用：

```python
capture_error_mode="thread_local"
```

同时 persistent 模式会在正式并发前先执行一次串行 Reasoner graph prime，避免 capture 与 T2V 并发。后续 warmup、alone、colocated 只 replay。

## 5. 正确性与适用边界

### 5.1 已验证范围

当前固定输入、batch=1、greedy、1350-token fixed-length 测试中，eager、graph/full-SM 以及所有 graph/GC SM 点的生成 token SHA-256 一致：

```text
fd71c838c5fc98cc0632c94949034e13b19fde636ed48c3fc459ec382da00021
```

验证通过环境变量启用：

```bash
export COSMOS_REASONER_VALIDATE_TOKENS=1
```

hash 在 decode timer 结束后计算，因此 D2H copy 不计入 decode CUDA event。

### 5.2 数值语义

static KV 和 dynamic KV 目标数学语义相同，但 attention backend 与浮点归约顺序不同。一般只会产生微小 logits 差异；当两个候选 token 非常接近时，greedy argmax 仍可能分叉。因此单个 prompt 的 hash 一致是必要证据，但不是对任意输入的形式化证明。

### 5.3 当前 graph 路径限制

当前只支持：

```text
single process
greedy decode
do_sample=False
repetition_penalty=1.0
presence_penalty=0.0
```

不满足条件时会显式报错。

### 5.4 并发安全

原版 KV 是 per-request。当前 persistent graph 对同一个 cache key 复用一套 static state，没有并发 slot 或锁。

因此当前实现适用于：

- profiling runner 中顺序到达的 Reasoner 请求；
- 一个 stream 同时只有一个 decode request。

它不能直接作为多请求并发 serving runtime。正式 serving 需要：

- 每个并发 slot 独立 graph/KV buffer；或
- graph pool；或
- scheduler 保证 runtime 独占。

### 5.5 常驻显存

module-level graph cache目前没有淘汰机制。每个新 shape、stream、prompt length 或 capacity 都可能保留一套 graph 和 static KV，直到进程退出。固定 workload profiling 可接受；动态 serving 会造成常驻显存增长。

## 6. Fixed-length 与真实 EOS early-exit

Static KV 本身不要求 fixed-length。以下开关只用于固定工作量控制实验：

```bash
export COSMOS_REASONER_FIXED_LENGTH_DECODE=1
```

它会在 EOS 后继续执行到 `max_new_tokens`，后续 token 使用 pad，因此改变：

- 实际计算量；
- 输出长度；
- KV 长度；
- latency；
- 与 T2V 的配平关系。

正式 scaling 和共置默认不设置该变量，保留真实 EOS early-exit：

```bash
unset COSMOS_REASONER_FIXED_LENGTH_DECODE
```

当前 non-fixed graph 每 token 检查 `bool(finished.all())`，包含 GPU→CPU 同步。这是当前 runtime 的真实成本，但未来 serving runtime 可以用批处理调度、device-side completion 或降低检查频率继续优化。

历史目录：

```text
profiling_results/reasoner_graph_sm_scaling/
```

是 fixed-length 历史数据。新主结果必须写入：

```text
profiling_results/reasoner_graph_sm_scaling_early_exit/
```

固定长度控制实验写入：

```text
profiling_results/reasoner_graph_sm_scaling_fixed_length/
```

## 7. Green Context 分区实现

实现文件：

```text
/workspace/cosmos-framework/profiling_tools/green_context.py
```

`GreenContextPair` 使用 CUDA Driver API：

```text
cuDeviceGetDevResource
cuDevSmResourceSplitByCount
cuDevResourceGenerateDesc
cuGreenCtxCreate
cuGreenCtxStreamCreate
```

关键点是只对完整 SM resource 做一次 split：

```text
available resource
    ├── first_resource
    └── returned remainder resource
```

然后分别从这两个 resource 创建 context/stream。这样两个 stream 的 SM 集合来自同一次 split，保证不重叠。不能创建两个独立 `GreenContextStream(requested_count)` 来代替，因为两个独立请求不保证拿到互斥 SM 集合。

### 7.1 为什么 generator 先分

`ColocationResources` 将 generator 设为 first resource：

```python
GreenContextPair(generator_sm_count, ignore_sm_coscheduling=False)
```

Reasoner 得到 driver 返回的 remainder。这样 generator 的容量是硬目标，适合保护 T2V latency。

### 7.2 requested 与 actual

CUDA 会按硬件 co-scheduling 粒度量化请求。RTX 5090 170-SM 实测示例：

```text
requested generator 162 → actual generator 168 + reasoner 2
requested generator 154 → actual generator 160 + reasoner 10
requested generator 146 → actual generator 152 + reasoner 18
requested generator 130 → actual generator 136 + reasoner 34
requested generator   2 → actual generator   8 + reasoner 162
```

所有分析和画图必须使用 actual SM，不能使用命令中的 requested 值或日志文件名。

## 8. runtime_region：模型代码与 runtime 解耦

实现文件：

```text
/workspace/cosmos-framework/cosmos_framework/runtime/regions.py
```

模型内部只写：

```python
with runtime_region("reasoner_decode"):
    ... decode ...
```

默认没有安装 factory 时，它等价于 `nullcontext()`，普通推理行为不变。

runner 在线程局部的 `ContextVar` 中安装 `DecodeRegionController.factory`。这样同一个模型函数无需知道 GC、barrier、NVTX 或计时细节。

ContextVar 很重要：两个 CPU worker 同时调用同一个模型时，Reasoner worker 的 region factory 不会泄漏到 generator worker。

## 9. DecodeRegionController 如何把 decode 放入 GC

进入 `reasoner_decode` 时：

1. 获取 prefill 所在的 original stream；
2. 在 original stream 记录 `prefill_done` event；
3. Reasoner GC stream `wait_event(prefill_done)`；
4. 切换到 `torch.cuda.stream(reasoner_gc_stream)`；
5. 必要时在 CPU barrier 等待 T2V；
6. 在 GC stream 记录 start event；
7. 执行 decode/graph replay；
8. 记录 end event；
9. original stream 等待 decode end，恢复后续依赖。

依赖关系是：

```text
original stream: prefill ── record(prefill_done) ───────── wait(decode_done)
                                  │                             ▲
reasoner GC:                      wait ── decode ── record ─────┘
```

这里没有在 prefill 与 decode 之间做全设备同步；只建立跨 stream event dependency。

## 10. 共置 runner 结构

实现文件：

```text
/workspace/cosmos-framework/profiling_tools/colocate_decode_t2v.py
```

模型和输入只加载一次：

```text
LoadedWorkloads
├── one shared Cosmos pipeline/model
├── reasoner sample args
└── generator sample args
```

每轮通过 `create_batches()` 重新构造 request data，避免复用已被推理过程修改的输入 tensor。

### 10.1 两个 CPU worker

`ThreadPoolExecutor(max_workers=2)` 启动：

```text
reasoner_worker                  generator_worker
---------------                  ----------------
fresh Reasoner request           fresh T2V request
normal prefill                   generator GC stream
runtime_region intercept         full/stage barrier
reasoner GC stream               generate_batch
decode/graph replay
```

两个 worker 共享模型权重，但在不同 GC stream 上提交 CUDA 工作。

### 10.2 full generator 共置

当：

```bash
COSMOS_COLOCATE_GENERATOR_REGION=full
```

使用三方 barrier：

- Reasoner worker 到达 decode region；
- Generator worker 准备开始完整 T2V；
- main thread。

barrier 释放后，decode 与完整 generator 同时开始。

### 10.3 component/stage 共置

当目标是：

```text
diffusion_sampling
vae_decode
```

runner 临时包装：

- `UniPCSampler.__call__` / `EDMSampler.__call__` / `FixedStepSampler.__call__`；或
- `OmniMoTModel.decode`。

generator 在目标组件入口与 Reasoner decode 使用两方 barrier。非目标阶段照常运行，只有指定阶段与 decode 对齐。

wrapper 在 context 退出时恢复原方法，不永久修改 class。

## 11. 为什么之前看起来近似串行

历史上已经排查过多个原因：

1. 最初 pipeline 数据依赖设计错误：同一个 request 的 Reasoner 和其依赖的 generator 串行，正确 pipeline 应理解为 request `i+1` 的 Reasoner 与 request `i` 的 generator 重叠；
2. 两个 Python thread 并不保证底层 kernel launch 真正并发；
3. flash-attention 内部一次性初始化/`call_once` 曾导致跨线程串行；
4. `CUDA_DEVICE_MAX_CONNECTIONS` 从 1 调到 8 有改善，8 到 32 收益很小；
5. CUDA Graph 的默认 side stream 曾逃出 GC，导致看似很快但分区不真实。

当前容器中的 flash-attention 使用过 `call_once` 相关修复，补丁曾保存在：

```text
/tmp/flash-attention-cosmos/call_once.diff
```

这是环境依赖，不应假设新容器天然包含。重新构建环境时必须确认该补丁是否已经持久化到镜像或安装包；否则并发行为可能回退。文档不把它声明为本仓库已管理的源码修改。

## 12. Warmup、alone、colocated 顺序

persistent graph 模式的典型顺序：

```text
1. Reasoner-alone graph prime（capture，不计入结果）
2. colocated warmup
3. partitioned Reasoner alone
4. partitioned T2V alone
5. measured colocation
```

graph prime 必须串行执行，避免 capture 期间另一个线程提交 T2V CUDA 工作。

alone 必须使用和 colocated 完全相同的 GC 分区。不能用 full-SM alone 计算 interference。

## 13. 计时口径

### 13.1 GPU event 时间

分别在所属 stream 上记录：

```text
T_decode_partition_alone
T_t2v_partition_alone
T_decode_colocated
T_t2v_colocated
```

定义：

```text
decode_slowdown = T_decode_colocated / T_decode_partition_alone
t2v_slowdown    = T_t2v_colocated / T_t2v_partition_alone
```

### 13.2 串行基线

runner 报告两个口径：

```text
serial_gpu_sum
  = decode partitioned-alone CUDA event
  + T2V partitioned-alone CUDA event

serial_host_wall
  = Reasoner-alone decode region 的实测 host wall
  + full T2V-alone request 的实测 host wall

serial_end_to_end_wall_diagnostic
  = 两个 workload 从 Reasoner prefill 前开始顺序执行的旧 wall-clock
  = 包含 prefill，仅作诊断，不用于吞吐计算
```

前者便于分析纯 GPU 工作。serial_host_wall 与 full-overlap 的 colocated
host_wall 使用对称边界，二者都只覆盖 decode + full T2V，并包含对应区间内
的 CPU/launch/synchronization 开销。

### 13.3 共置收益

full-generator 模式下：

```text
colocated_makespan = max(T_decode_colocated, T_t2v_colocated)
```

报告：

```text
throughput_gain_gpu_sum
  = serial_gpu_sum / colocated_makespan

throughput_gain_host_wall
  = serial_host_wall / colocated host wall

这里的两个 wall 都由 time.perf_counter 实测，不是由两个 GPU duration
取 max 得到。只有 colocated_makespan 使用
max(T_decode_colocated, T_t2v_colocated)。
```

stage 模式中 decode 可能在 generator 中途才开始，不能简单用两个 duration 的 max 重建全局时间轴；应优先使用 host wall 或通过 Nsys timestamp 计算真实 union interval。

当前 stage-mode `wall_start` 在 worker submit 之后记录，可能漏掉非常小的一段线程启动时间。正式端到端论文数据前应把 wall 起点移到 submit 之前，或完全以 Nsys/CUDA timestamp 重建时间轴。full 模式因三方 barrier 存在，不受该竞态影响。

## 14. NVTX 与 Nsys/NCU

关键 NVTX range：

```text
reasoner_decode_graph_prime
reasoner_decode_alone
reasoner_decode_balanced_measured_warm
reasoner_decode_full_sm_graph_prime
reasoner_decode_full_sm_alone
reasoner_decode_gc_graph_prime
reasoner_decode_gc_alone

NSYS_REASONER_ALONE
NSYS_GENERATOR_ALONE
NSYS_BALANCED_MEASURED
```

不同阶段使用不同 range，防止 NCU 的 `--launch-count 1` 抓到 prefill 或首次 materialization，而不是 measured replay。

NCU 必须使用 2025.3.1 或更高版本，并检查：

```text
Uses Green Context
# SMs
Context / Stream
NVTX Push/Pop Stack
```

一次错误的 full-SM graph GEMV 报告显示：

```text
No Eligible                    84.57%
L1TEX scoreboard stall        41.57%
L2 hit rate                   12.36%
Memory throughput             943.86 GB/s
Max bandwidth                 53.72%
```

它说明该 GEMV 有明显 memory-dependency latency，但因为 `GC=0/#SMs=170`，不能用于解释 34-SM GC 干扰。

## 15. Scaling 结果与画图

fixed-length 历史 sweep 的 actual Reasoner SM 是：

```text
2, 10, 18, 26, 34, ..., 154, 162, 170
```

不是 requested 的 `8,16,...`。示例时间：

```text
actual 2 SM     139.078 s
actual 10 SM     29.672 s
actual 18 SM     17.641 s
actual 34 SM     10.537 s
actual 82 SM      6.555 s
actual 130 SM     5.957 s
actual 162 SM     5.976 s
full 170 SM       6.087 s
```

130–170 SM 的小幅非单调变化属于单次测量噪声/频率波动范围，不能解释为受限 SM 真正加速。正式结果应多轮重复并报告均值和置信区间。

当前 speedup 定义遵循项目已有约定：

```text
Speedup = T_full / T_restricted
```

画图脚本：

```text
/workspace/cosmos-framework/profiling_tools/analysis/plot_reasoner_graph_sm_scaling.py
```

fixed-length 历史图：

```text
profiling_results/reasoner_graph_sm_scaling/plots/
```

其中 ratio 图包含 `(0,0)` 到 `(1,1)` 的理想线。

## 16. 运行命令

### 16.1 默认：真实 EOS early-exit scaling

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate

./profiling_tools/scripts/run_reasoner_graph_sm_scaling.sh
```

输出：

```text
profiling_results/reasoner_graph_sm_scaling_early_exit/
```

### 16.2 Fixed-length 控制实验

```bash
FIXED_LENGTH_DECODE=1 \
  ./profiling_tools/scripts/run_reasoner_graph_sm_scaling.sh
```

输出：

```text
profiling_results/reasoner_graph_sm_scaling_fixed_length/
```

### 16.3 完整 T2V 共置

以下示例请求 generator=154；当前硬件量化后预期为 generator=160、reasoner=10：

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate

export LD_LIBRARY_PATH=''
export CUDA_DEVICE_MAX_CONNECTIONS=8
export COSMOS_REASONER_CUDA_GRAPH=1
export COSMOS_REASONER_CUDA_GRAPH_PERSISTENT=1
export COSMOS_REASONER_VALIDATE_TOKENS=1
unset COSMOS_REASONER_FIXED_LENGTH_DECODE

export COSMOS_GENERATOR_SM_COUNT=154
export COSMOS_REASONER_REPEATS=1
export COSMOS_WARMUP_RUNS=1
export COSMOS_COLOCATE_GENERATOR_REGION=full

unset COSMOS_FULL_SM_DECODE_ONLY
unset COSMOS_GC_DECODE_ONLY
unset COSMOS_NSYS_CAPTURE

python -m profiling_tools.runners.colocate_decode_t2v 2>&1 \
  | tee profiling_results/decode_t2v_balanced/graph_g160_r10_full_overlap_early_exit.log
```

### 16.4 只测 GC decode

```bash
export COSMOS_GC_DECODE_ONLY=1
unset COSMOS_FULL_SM_DECODE_ONLY
python -m profiling_tools.runners.colocate_decode_t2v
```

### 16.5 只测 full-SM decode

CUDA Graph 不能在 legacy default stream 上 capture，因此 runner 创建普通非 default、未受 GC 限制的 primary-context stream：

```bash
export COSMOS_FULL_SM_DECODE_ONLY=1
unset COSMOS_GC_DECODE_ONLY
python -m profiling_tools.runners.colocate_decode_t2v
```

## 16.6 Eager/Graph × 分区影响矩阵

为了同时观察双向 slowdown，并避免只报告一个偶然的配平点，矩阵默认测试：

    implementation ∈ {eager, graph}
    actual partition ∈ {
      Reasoner 10 / Generator 160,
      Reasoner 18 / Generator 152,
      Reasoner 34 / Generator 136,
      Reasoner 50 / Generator 120
    }

运行：

    ./profiling_tools/scripts/run_decode_t2v_impact_matrix.sh

默认输出：

    profiling_results/decode_t2v_impact_matrix/

汇总：

    python -m profiling_tools.analysis.summarize_decode_t2v_impact_matrix

汇总 CSV：

    profiling_results/decode_t2v_impact_matrix/summary.csv

只补最小配平集合时，可以运行：

    IMPLEMENTATIONS=eager REQUESTED_GENERATOR_VALUES=114 RUN_TAG=balance01 \
      ./profiling_tools/scripts/run_decode_t2v_impact_matrix.sh

    IMPLEMENTATIONS=graph REQUESTED_GENERATOR_VALUES="154 146" RUN_TAG=balance01 \
      ./profiling_tools/scripts/run_decode_t2v_impact_matrix.sh

原因是 eager 的预计配平点是实际 50:120；graph 因 GC 粒度限制，
配平点位于 10:160 和 18:152 之间，需要用两个点夹逼。

每个单元都必须比较：

    balance_ratio_decode_over_t2v = decode_alone / t2v_alone
    decode_slowdown               = decode_colocated / decode_alone
    t2v_slowdown                  = t2v_colocated / t2v_alone
    throughput_gain_host_wall     = matched serial host wall / colocated host wall

其中 balance ratio 越接近 1，吞吐收益越能代表真正的时间配平共置。

## 16.7 Reasoner decode × T2I

单次 warm T2I 只有约 1.1 s，远短于一个 rich02 Reasoner decode。因此
T2I 实验把 generator workload 定义为同一 stream 上连续执行 N 个独立
fresh-batch T2I 请求，而不是通过修改单张图片的工作量来凑时间。

Runner 新增：

    COSMOS_GENERATOR_WORKLOAD=t2i
    COSMOS_GENERATOR_REPEATS=N

默认 T2I 输入：

    outputs/local_agentic_t2i/smoke01/inputs/t2i_upsampled.json

运行完整 eager/graph × 四分区配平矩阵：

    RUN_TAG=matrix01 ./profiling_tools/scripts/run_decode_t2i_impact_matrix.sh

快速验证预计最接近配平的两个单元：

    IMPLEMENTATIONS=eager REQUESTED_GENERATOR_VALUES=114 RUN_TAG=quick01 \
      ./profiling_tools/scripts/run_decode_t2i_impact_matrix.sh

    IMPLEMENTATIONS=graph REQUESTED_GENERATOR_VALUES=146 RUN_TAG=quick01 \
      ./profiling_tools/scripts/run_decode_t2i_impact_matrix.sh

汇总：

    python -m profiling_tools.analysis.summarize_decode_t2v_impact_matrix \
      --input-dir profiling_results/decode_t2i_impact_matrix \
      --output profiling_results/decode_t2i_impact_matrix/summary.csv

矩阵中的 generator duration 是 N 个 T2I 的总时间，generator slowdown
也是整段请求序列的 slowdown。单请求吞吐可再用 N / duration 计算。

### 16.7.1 T2I 正式矩阵结果

正式 matrix01 的 8 个单元均完成，Reasoner 输出 token hash 一致。结果：

| Decode | Actual R/G | T2I repeats | Alone decode/T2I (s) | Slowdown decode/T2I | Host throughput gain |
|---|---:|---:|---:|---:|---:|
| eager | 10/160 | 31 | 33.998 / 34.059 | 1.812 / 1.444 | 1.105x |
| eager | 18/152 | 27 | 30.432 / 29.615 | 1.829 / 1.487 | 1.078x |
| eager | 34/136 | 27 | 29.098 / 29.909 | 1.882 / 1.481 | 1.077x |
| eager | 50/120 | 25 | 28.922 / 28.454 | 1.808 / 1.444 | 1.097x |
| graph | 10/160 | 27 | 29.752 / 29.553 | 1.071 / 1.026 | 1.861x |
| graph | 18/152 | 16 | 17.711 / 17.863 | 1.078 / 1.017 | 1.863x |
| graph | 34/136 | 10 | 10.576 / 11.432 | 1.092 / 1.040 | 1.851x |
| graph | 50/120 | 7 | 8.368 / 7.979 | 1.092 / 1.039 | 1.789x |

正式 CSV：

    profiling_results/decode_t2i_impact_matrix/summary_matrix01.csv

结论：

- 所有单元基本配平，alone 时间最大偏差约 8%，多数在 3% 内；
- eager 的双向干扰在四种分区下都很大且近似不随 SM 比例变化；
- graph 下 T2I 序列仅变慢约 1.7%--4.0%，decode 仅变慢约 7%--9%；
- graph 的配平吞吐收益为 1.79--1.86x，接近理想 2x；
- eager 只有 1.08--1.11x，说明简单增加或减少任一侧 SM 无法解决主要干扰。

这种分区不敏感、实现敏感的差异强烈支持 host/driver launch 提交竞争
是 eager 路径主要瓶颈之一，而不是单纯的 SM 算力竞争。该解释仍属于基于
矩阵的推断；严格归因需要用 Nsys 比较 eager 与 graph 的 kernel launch gap、
API duration、stream overlap 和 CPU thread timeline。

## 17. 结果接受条件

一组数据进入最终分析前必须同时满足：

1. 日志记录 requested 和 actual SM；
2. NCU 抽查确认 `Uses Green Context=1` 和正确的 `#SMs`；
3. prime 不计入 measured；
4. alone 与 colocated 使用相同分区；
5. Reasoner alone/colocated 生成 token 数相同；
6. token hash 相同；
7. T2V 输出 shape/dtype/finite 检查通过；
8. 没有 OOM、capture invalidated 或 barrier broken；
9. 明确标注 fixed-length 或 early-exit；
10. 不混用旧的 primary-stream graph 结果；
11. 至少多轮重复后才解释 1%–3% 的差异；
12. 日志和绘图使用 actual SM。

## 18. 已知未完成工作

### 18.1 Static-KV eager 中间基线

需要实现：

```text
static KV + flash_attn_with_kvcache + Python eager token loop
```

用于分离：

- dynamic → static KV 的收益；
- eager → CUDA Graph 的纯收益。

### 18.2 Early-exit 主实验

fixed-length sweep 已完成，但主结论应重新运行：

- non-fixed full-SM；
- non-fixed SM scaling；
- non-fixed partitioned-alone；
- non-fixed full T2V colocation；
- non-fixed component colocation。

### 18.3 多轮统计

当前大部分点只有一个 measured request。正式数据需要至少 3–5 轮，报告：

- mean；
- median；
- standard deviation 或置信区间；
- GPU clocks / power 状态；
- token count。

### 18.4 Serving 化

当前 persistent runtime 不是通用 serving runtime。仍需：

- graph/KV slot pool；
- concurrency ownership；
- shape bucket；
- cache eviction；
- continuous batching/ragged prompt；
- sampling/penalty 支持；
- 无逐 token CPU sync 的 completion 检测；
- 异常恢复和 graph invalidation。

### 18.5 Runtime 测量细化

需要修正 stage-mode wall 起点，并将每轮 request 的绝对 CUDA/Nsys timestamp 导出，以严格计算：

- overlap interval；
- fill/drain；
- component union；
- pipeline steady-state throughput。

## 19. 文件索引

```text
模型 decode/static KV/CUDA Graph
  /workspace/cosmos-framework/cosmos_framework/model/generator/mot/unified_mot.py

runtime region 接口
  /workspace/cosmos-framework/cosmos_framework/runtime/regions.py

CUDA Driver Green Context 封装
  /workspace/cosmos-framework/profiling_tools/green_context.py

共置 runner
  /workspace/cosmos-framework/profiling_tools/colocate_decode_t2v.py

decode scaling sweep
  /workspace/cosmos-framework/profiling_tools/scripts/run_reasoner_graph_sm_scaling.sh

decode scaling 汇总/画图
  /workspace/cosmos-framework/profiling_tools/analysis/plot_reasoner_graph_sm_scaling.py

Nsys/NCU 操作
  /workspace/cosmos-framework/profiling_tools/docs/nsys_ncu.md

较早的 runtime 设计记录
  /workspace/cosmos-framework/profiling_tools/docs/archive/decode_t2v_colocation_design.md
```

## 20. 最重要的结论

1. Cosmos 原版 attention 并不朴素；它使用优化 backend dispatcher。主要 decode 浪费来自 dynamic KV 和逐 token orchestration。
2. 当前优化不是“只加 graph”，而是 static KV、KV-cache attention 和 graph replay 的组合。
3. CUDA Graph 必须显式 capture 到 GC stream；省略 `stream=` 会静默逃到 170-SM primary stream。
4. 共置 interference 必须相对同分区 alone 计算，不能相对 full-SM。
5. requested SM 不等于 actual SM，所有图表必须使用 driver 返回值。
6. fixed-length 仅作控制实验；真实结论默认使用 EOS early-exit。
7. 当前 persistent graph 已在固定输入上通过 token hash 验证，但不是并发安全的通用 serving 实现。
8. 任何没有 GC/SM 归因、token 正确性和 warmup 隔离的数据，都不应进入最终结论。
