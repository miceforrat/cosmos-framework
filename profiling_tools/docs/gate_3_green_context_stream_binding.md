# Gate 3：vLLM Reasoner 与 vLLM-Omni Generator 的 Green Context 绑定

本文记录单进程双模型进入 Green Context（GC）阶段后的实现、测量证据、当前失败原因，以及后续采用源码修改或 monkey patch 时需要触及的代码边界。本文不覆盖 Gate 1/2 文档，也不把尚未由 Nsys 验证的 stream 请求称为 GC 绑定成功。

## 1. 实验目标与运行栈

目标是在一个 PID、一个 CUDA primary context 中同时常驻：

- vLLM Reasoner：执行 prompt upsampler，自回归 decode；
- vLLM-Omni Generator：执行 `832x480`、121 帧、35 diffusion steps 的 T2V；
- Generator 使用主分区，Reasoner 使用剩余分区；
- 当前请求为 Generator 136 SM、Reasoner 34 SM，总计 170 SM。

入口：

```text
/workspace/cosmos-framework/profiling_tools/runners/concurrent_reasoner_generator.py
```

GC Driver API 封装：

```text
/workspace/cosmos-framework/profiling_tools/core/green_context.py
```

`GreenContextPair` 对同一份 SM resource 做一次 split，再分别创建两个 Green Context 和两个 stream。这一分配方式能够保证两个资源集合互不重叠；不能用两个彼此独立的单 GC allocation 替代。

## 2. 当前 runner 做了什么

runner 当前执行以下操作：

1. 创建 vLLM-Omni Generator；
2. 创建 `GreenContextPair(136)`；
3. 把 first stream 包装成 Generator `torch.cuda.ExternalStream`；
4. 把 remainder stream 包装成 Reasoner `torch.cuda.ExternalStream`；
5. 创建 Reasoner 时暂时把 PyTorch current stream 设置为 Reasoner GC stream；
6. 调用两个高级 API 时，分别在对应的 `torch.cuda.stream(...)` context 中运行；
7. 用两个 host thread 同时发起 Reasoner 和 Generator 请求；
8. 记录 alone、concurrent host wall time，并用 Nsys 验证 GPU placement。

这个实现能够表达“希望后端使用哪个 stream”，但不能强制后端内部 worker、model runner 或 CUDA Graph 使用调用者的 current stream。

## 3. Warm Nsys 实验

报告：

```text
profiling_results/gate_3_green_context/nsys/eager_eager_g136.nsys-rep
profiling_results/gate_3_green_context/nsys/graph_eager_g136.nsys-rep
```

对应 JSON：

```text
profiling_results/gate_3_green_context/nsys/eager_eager_g136.json
profiling_results/gate_3_green_context/nsys/graph_eager_g136.json
```

两个实验均先执行一次 serial warmup，再执行一次 alone 和一次 concurrent。固定输出长度分别为 eager 1200 tokens、graph 2600 tokens。因此下面是 warm workload，但一次重复只用于定位，不作为正式统计结果。

### 3.1 Host 计时

| Reasoner | Reasoner alone | Generator alone | Concurrent wall | Reasoner slowdown | Generator slowdown | Host-wall throughput gain |
|---|---:|---:|---:|---:|---:|---:|
| eager | 29.550 s | 18.816 s | 43.560 s | 1.474x | 1.014x | 1.110x |
| graph | 9.154 s | 18.601 s | 26.410 s | 2.885x | 1.059x | 1.051x |

吞吐定义为：

```text
(full request A alone host wall + full request B alone host wall)
---------------------------------------------------------------
                  concurrent host wall
```

这里的较低收益不能直接解释为“Green Context 没有效果”，因为 Nsys 证明 workload 没有正确进入两个 GC。

### 3.2 Nsys 中创建出的 CUDA contexts

两个报告都包含：

```text
context 1: primary context
context 2: Green Context, 136 SM
context 3: Green Context,  34 SM
```

因此 SM resource split 和 GC creation 本身成功。失败发生在 kernel submission 的 stream/context 归属。

### 3.3 Eager placement

主要 kernel 分布：

```text
Generator: greenContextId=0, streamId=7
Reasoner:  greenContextId=3, streamId=34
GC 2:     zero kernels
```

结论：Reasoner eager kernel 正确进入 34-SM GC；Generator 完全绕过 136-SM GC，仍从普通 primary context stream 提交，因此可以使用全卡 SM。

在两个 concurrent host NVTX range 的交集内：

```text
window                    19.076 s
GC 0 union busy           18.014 s
GC 3 union busy            2.092 s
GC0/GC3 kernel overlap     1.747 s
```

这个 overlap 只能说明普通-context Generator 与 GC Reasoner 有并发，不能称为 136:34 空间隔离。

### 3.4 Graph placement

主要 kernel 分布：

```text
Generator:            greenContextId=0, streamId=7
Reasoner graph nodes: greenContextId=0, streamId=221
Reasoner peripheral:  greenContextId=3, streamId=34
GC 2:                 zero kernels
```

在 concurrent host 区间内，GC 3 只有约 0.687 s kernel busy，而普通 context 约 19.224 s busy。Reasoner Graph 主体没有保留 34-SM GC 归属。

因此 `reasoner_graph_gc_binding_requested=true` 只表示 runner 发出了绑定请求；它不等于 placement 已验证。JSON 中 `green_context_kernel_placement_verified=false` 是有意保留的事实边界。

## 4. 为什么外层 stream context 不生效

### 4.1 Generator

外层调用是：

```python
with torch.cuda.stream(generator_gc_stream):
    generator.generate(...)
```

但 `Omni.generate()` 会进入 scheduler/executor/worker。真正调用 Cosmos3 pipeline 的位置是持久化 vLLM-Omni 源码中的：

```text
.runtime/vllm_omni_tf517/vllm-omni/vllm_omni/diffusion/worker/diffusion_worker.py
    DiffusionWorker.execute_model()

.runtime/vllm_omni_tf517/vllm-omni/vllm_omni/diffusion/worker/diffusion_model_runner.py
    DiffusionModelRunner._execute_request_list()
        pipeline.run_stage(batch) / pipeline.forward(batch)
```

当前使用 uniproc executor，但 worker 调用仍由内部 runtime 发起。CUDA current stream 是线程局部状态；高级 API 调用线程上的 `torch.cuda.stream(...)` 不会自动传播到内部执行线程。因此 pipeline 最终使用普通 stream 7。

### 4.2 Reasoner eager

Reasoner eager 当前能够进入 GC 3，说明当前单进程配置下，外层 wrapping 对 eager model execution 路径有效。但这只是当前 vLLM 版本和执行模式下的实测事实，仍应由每次 Nsys placement check 验证，而不能视为稳定 API 契约。

### 4.3 Reasoner CUDA Graph

vLLM 的 graph capture 位于：

```text
.runtime/vllm_omni_tf517/venv/lib/python3.13/site-packages/vllm/compilation/cuda_graph.py
    CUDAGraphWrapper.__call__()
```

它显式执行：

```python
with torch.cuda.graph(
    cudagraph,
    pool=self.graph_pool,
    stream=current_stream(),
):
    ...
```

这里的 `current_stream()` 在 vLLM 真正执行 capture 的内部线程上求值，而不是在创建 `LLM(...)` 的调用线程上求值。只在外部 `create_reasoner()` 前执行 `torch.cuda.set_stream(reasoner_gc_stream)`，不足以控制内部 capture stream。结果是 graph 捕获到普通 stream 221，并在 GC 0 中 replay。

这和此前 Cosmos 基模 prototype 不同：prototype 直接控制 token loop，并把 GC stream 显式传给 `torch.cuda.graph(..., stream=current_gc_stream)`；没有 serving runtime 的线程和 stream 所有权边界。

## 5. 需要修改多少代码

### 5.1 Generator：当前 workload 的最小原型

当前单请求、非 stepwise workload 的最低侵入点是：

```text
DiffusionWorker.execute_model()
```

在真正的 worker 执行线程内，用 Generator GC ExternalStream 包住原方法：

```python
with torch.cuda.stream(generator_gc_stream):
    return original_execute_model(...)
```

这比只包 `pipeline.forward()` 更完整，因为 request preparation、conditioning、diffusion sampling、VAE decode 和输出准备中可能都包含 CUDA 工作。

原型预计只需一个 monkey patch wrapper，加一个“如何取得 Generator stream pointer”的进程内注册接口。修改规模约几十行。

局限：只 patch `execute_model()` 仅覆盖当前 single-request 路径。完整支持还需覆盖：

```text
execute_model_batch()
execute_stepwise()
profile_run()/初始化 warmup（若要求初始化工作也受分区）
内部额外创建的 copy/offload/decode streams
```

当前没有使用 layerwise offload；若未来打开，普通 copy stream 不一定适合强制进入 compute GC，需要单独定义数据搬运与 compute 的同步契约。

### 5.2 Reasoner eager：最小原型

当前 eager 已由 Nsys 验证进入 GC 3，因此暂时不需要修改 vLLM 源码。应增加启动自检，而不是先扩大 patch 面。

### 5.3 Reasoner graph：最小原型

Reasoner graph 需要同时满足：

1. capture 在 Reasoner GC stream 上；
2. replay 使用捕获后的同一 GC graph；
3. eager fallback、input staging 和相关 events 不引入普通-stream 全局依赖。

最小实验性 monkey patch 有两个部分：

- 在 vLLM model-runner 的实际执行线程入口设置 Reasoner GC ExternalStream；
- patch `vllm.compilation.cuda_graph.current_stream`，让 capture 取得该线程已经安装的 GC stream。

不能只替换 `CUDAGraph.replay()` 外面的 stream context，因为 graph node 的资源归属由 capture 决定。

直接全局把 `current_stream()` 固定成一个 GC stream 风险较高：它可能影响 vision encoder、profiling capture、offloader 或其他 model runner。monkey patch 必须由线程局部/context variable 控制，只在 Reasoner model execution/capture region 内返回 GC stream，其他调用保持原行为。

### 5.4 可维护的源码接口

长期方案不应依赖全局 monkey patch。建议给两个框架各加一个可选 runtime stream provider：

```python
set_external_compute_stream(stream_or_provider)
```

Generator 在 `DiffusionWorker.execute_*` 的执行线程内进入该 stream；Reasoner model runner 在 eager forward、graph capture 和 replay 使用同一 provider。默认值为 `None`，保证上游行为不变。

预计改造面：

| 部分 | 原型 monkey patch | 可维护实现 |
|---|---:|---:|
| Generator single request | 1 个方法 wrapper | worker 配置 + 3 个 execute 入口 |
| Reasoner eager | 暂不修改 | model-runner stream provider |
| Reasoner graph | model-runner wrapper + capture stream hook | capture/replay 统一 stream provider |
| 验证 | Nsys 手工查询 | 启动 probe + Nsys 回归 |

因此并不需要修改 Cosmos3 模型层或各个 Transformer block。修改集中在两个 serving runtime 的 worker/model-runner 边界，但 graph 路径必须比 eager 更谨慎。

## 6. Monkey patch 还是改 fork

建议顺序：

1. 先做 profiling-only monkey patch，验证 Generator GC 2 和 Reasoner Graph GC 3 都出现主要 kernel；
2. 重跑一轮 warm alone/concurrent，确认性能是否回到基模趋势；
3. 只有 placement 与性能都成立后，再把接口落入持久化的 vLLM/vLLM-Omni fork；
4. 不直接修改 `.runtime/.../site-packages`，因为重新建环境后不可追踪；
5. monkey patch 放在仓库 `profiling_tools/core/`，并记录所适配的 vLLM/vLLM-Omni commit/version。

Monkey patch 的意义是验证最小改造点，不是最终部署方案。

## 7. 强制验收条件

修复后的 Gate 3 必须同时满足：

```text
Generator major kernels: greenContextId = GC(136 SM)
Reasoner major kernels:  greenContextId = GC(34 SM)
ordinary GC 0:           只允许可解释的少量辅助工作
```

并记录：

- 每个 GC 的 kernel count 与 union busy time；
- concurrent NVTX 交集内两个 GC 的 kernel overlap；
- graph node 的 `greenContextId`；
- alone/concurrent host wall；
- 固定 token workload 是否一致；
- Generator cache、Reasoner eager/graph、warmup/repeat 配置。

在上述 placement 条件满足前，`spatial_sharing_enabled=true` 只能表示配置意图，不能写成“空间切分已验证”。
