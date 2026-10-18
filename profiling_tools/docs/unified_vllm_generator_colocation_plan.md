# Cosmos3 Reasoner（vLLM）与 Generator 统一调度共置方案

## 1. 文档目的

本文档定义下一阶段的工程方案：保留 Cosmos3 Reasoner 和 Generator 各自的模型后端，但将 GPU 执行交给一个统一调度层，以实现：

- Reasoner 使用 vLLM 的模型实现、Paged KV、attention backend 和 CUDA Graph 能力；
- Generator 继续使用 Cosmos Framework 中已经验证的完整生成路径；
- 在同一 Python 进程和同一 CUDA context 内控制两条 CUDA stream；
- 对 Reasoner decode 和 Generator 分别施加 Green Context（GC）SM 分区；
- 流水化重叠请求 `i+1` 的 Reasoner 与请求 `i` 的 Generator；
- 保留严格的正确性、独占性能、共置干扰和端到端吞吐验证。

本方案不是把两个完整服务器简单放入同一个容器，也不是立即重写 vLLM 或 Generator。第一阶段的目标是确认两个后端能在同一进程稳定共存，然后逐步下沉执行控制粒度。

## 2. 已确认事实与尚未确认的问题

### 2.1 已确认事实

1. Cosmos3-Reasoner NIM 1.7.0 使用 custom vLLM backend，而 Cosmos3-Generator NIM 是另一条独立的 serving/runtime 路径。
2. 仓库中的 Cosmos3 vLLM adapter 位于：

   ```text
   /data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/packages/vllm-cosmos3/
   ```

3. `Cosmos3ReasonerForConditionalGeneration` 继承 vLLM 的 `Qwen3VLForConditionalGeneration`，仅覆写初始化和权重加载，没有覆写 `forward`、attention、KV cache 或 decode loop：

   ```text
   /data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/packages/vllm-cosmos3/vllm_cosmos3/model.py
   ```

4. 因而在开源插件路径中，Reasoner 的模型执行、attention 和 KV 管理由 vLLM 提供。Cosmos3 adapter 主要负责过滤 Generator 权重并完成权重名称映射。
5. vLLM 的 KV 管理是预分配 block pool 加 block table 的 Paged KV，不等同于我们此前实现的单个连续 static-KV tensor，但两者都避免每个 token 重新扩展整段 KV tensor。
6. NIM 默认没有通过 `NIM_DISABLE_CUDA_GRAPH` 关闭 CUDA Graph，但“没有关闭”不等于具体请求一定执行 full CUDA Graph replay。

### 2.2 仍需运行时验证

- 当前目标 vLLM/NIM 的精确 vLLM 版本及 NVIDIA 私有或上游补丁；
- Cosmos3 Reasoner 实际选择的 attention backend；
- 最终 `cudagraph_mode`，以及是否因 backend 或输入形态发生降级；
- decode token 是否持续通过 `cudaGraphLaunch` replay；
- vLLM 是否启动额外 worker 进程或建立独立 CUDA context；
- vLLM graph capture/replay 是否能稳定绑定到指定 stream 和 Green Context；
- Reasoner 与 Generator 同时加载后的显存峰值是否可接受；
- Generator 内部是否仍存在默认 stream、host synchronization 或其他阻塞点。

因此本文档把 Paged KV 和 CUDA Graph 视为需要实测确认的 backend 能力，而不是仅凭“集成 vLLM”认定已经生效。

## 3. 目标部署结构

```text
单个自定义容器
└── 单个 Python 主进程 / 单个 CUDA primary context
    ├── UnifiedScheduler
    │   ├── 请求依赖与跨请求流水线
    │   ├── Reasoner / Generator worker thread
    │   ├── CUDA stream 与 CUDA event
    │   ├── Green Context 分区
    │   ├── NVTX、GPU event 与 host wall 计时
    │   └── 结果记录和异常传播
    │
    ├── ReasonerBackend
    │   ├── Cosmos3 Reasoner adapter
    │   ├── vLLM Qwen3-VL model runner
    │   ├── Paged KV block pool / block table
    │   ├── attention backend
    │   └── eager 或 CUDA Graph decode
    │
    └── GeneratorBackend
        ├── condition_prepare
        ├── diffusion_sampling
        ├── vae_decode
        └── Cosmos Framework 模型与输入状态
```

容器只提供统一依赖环境。真正决定共置质量的是同一进程、stream 隔离、GC 生效位置、graph capture/replay 位置以及统一调度，而不是容器数量。

## 4. 请求依赖与正确的流水线关系

同一请求的 Generator 依赖 Reasoner/upsampler 输出，因此不能将同一请求的两阶段直接并发。共置应跨请求进行：

```text
时间 ───────────────────────────────────────────────>

请求 i:      Reasoner(i) ─── Generator(i) ───────────
请求 i+1:                    Reasoner(i+1) ─── Generator(i+1)
                             <---- overlap ---->
```

稳态时，共置对象是：

```text
Reasoner decode(request i+1) × Generator(request i)
```

首个请求存在 pipeline fill，最后一个请求存在 pipeline drain。吞吐实验必须分别报告：

- 单个请求 latency；
- 稳态 stage interval；
- N 个请求实测 host wall；
- 去除 fill/drain 后的稳态吞吐（仅作辅助值）。

## 5. 后端接口

接口的第一版应保持小而稳定，避免把 vLLM 或 Cosmos Framework 的内部对象扩散到统一调度层。

```python
class ReasonerBackend:
    def load(self) -> None: ...
    def prepare_request(self, request): ...
    def prefill(self, request_state): ...
    def decode_step(self, request_state): ...
    def is_finished(self, request_state) -> bool: ...
    def finalize(self, request_state): ...


class GeneratorBackend:
    def load(self) -> None: ...
    def prepare_condition(self, request): ...
    def diffusion_sampling(self, condition_state): ...
    def vae_decode(self, latent_state): ...
    def finalize(self, generator_state): ...


class UnifiedScheduler:
    def run_serial(self, requests): ...
    def run_pipeline(self, requests): ...
    def run_colocated(self, reasoner_state, generator_state): ...
```

第一版可允许 `ReasonerBackend` 内部使用较粗的 `engine.step()`。但最终 GC 共置版本需要在 vLLM GPU model runner 的执行入口附近获得控制，至少能确保一次 decode GPU forward 在指定 stream/GC 中执行。

## 6. vLLM 的接入层次

### 6.1 不采用的第一选择：直接嵌入 `vllm serve`

`vllm serve` 适合独立服务，但它会带入完整 HTTP 层、调度器、worker/executor 和自己的设备控制。即使与 Generator 位于同一容器，也通常仍是两个进程或两个相对封闭的 runtime，难以保证：

- 共用同一 CUDA context；
- kernel 在指定 stream 上 launch；
- Green Context 在真正的 GPU launch 线程生效；
- graph 在目标 stream 上 capture 和 replay。

因此独立 NIM 或 `vllm serve` 只作为官方性能基线，不作为第一版 GC 共置骨架。

### 6.2 第一阶段：in-process vLLM engine

先在当前 Python 进程创建 vLLM engine，完成以下验证：

1. Cosmos3 Reasoner 正确加载和生成；
2. TP=1 时没有额外 GPU worker 进程；
3. vLLM 与 Generator 能在同一进程先后加载；
4. Paged KV block pool 确实建立；
5. CUDA Graph 模式、capture size 和 attention backend 可观测；
6. 两个后端的总显存峰值可接受。

这一步先追求“同进程串行正确”，不加入并发或 GC。

### 6.3 第二阶段：在 GPU model runner 执行入口增加薄 hook

若 `engine.step()` 隐藏了 stream 或执行线程，需要在 vLLM GPU model runner 的模型执行入口增加薄 hook：

```python
with runtime_region("reasoner_decode"):
    output = model_runner.execute_model(...)
```

该 hook 只负责：

- 进入 Reasoner Green Context；
- 选择 Reasoner CUDA stream；
- 添加 NVTX 和 CUDA event；
- 执行已有 eager forward 或 graph replay。

不应在这一层重写：

- Qwen3-VL attention；
- Paged KV block allocator；
- block table；
- sampling；
- vLLM CUDA Graph dispatcher。

## 7. Runtime region、stream 与 Green Context

现有 runtime region 位于：

```text
/data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/cosmos_framework/runtime/regions.py
```

统一调度层持有 GPU 资源：

```python
@dataclass
class RuntimeResources:
    reasoner_stream: torch.cuda.Stream
    generator_stream: torch.cuda.Stream
    reasoner_gc: object
    generator_gc: object
```

预期执行结构：

```python
def run_reasoner_decode(...):
    with reasoner_gc:
        with torch.cuda.stream(reasoner_stream):
            reasoner.decode_until_finished(...)


def run_generator(...):
    with generator_gc:
        with torch.cuda.stream(generator_stream):
            generator.run(...)
```

具体实现必须满足：

1. GC 在真正 launch kernel 的线程中进入，而不是只在提交任务的主线程进入；
2. CUDA Graph 在 Reasoner 目标 stream 上 capture，并在兼容的 stream/context 条件下 replay；
3. Generator 不隐式切回 legacy default stream；
4. 两条路径之间只通过必要 CUDA event 表达数据依赖；
5. 不在共置区间调用全设备 `torch.cuda.synchronize()`；
6. `.item()`、阻塞 D2H copy 和 Python barrier 不应出现在每 token 热路径。

## 8. 线程和进程模型

目标是：

```text
一个 GPU 主进程 + 两个 host worker thread + 两条 CUDA stream
```

理由：

- 同一进程更容易共享 CUDA primary context；
- Green Context 和 stream 可由统一调度层建立；
- 避免 MPS 只能提供进程级配额的问题；
- 避免跨进程 event、IPC tensor 和重复显存池。

如果 vLLM 无法在目标配置下避免独立 worker 进程，应先停止 GC 集成，记录原因，再评估：

- 使用 vLLM in-process executor；
- 下沉到 model runner；
- 或将双进程版本降级为 MPS/自然并发基线。

不能把双进程自然并发的结果称为 Green Context 共置结果。

## 9. 显存规划

两个后端同时驻留时，至少包括：

- Reasoner 权重；
- vLLM KV block pool；
- vLLM CUDA Graph private pool/capture buffers；
- Generator 权重；
- diffusion latent 和临时 workspace；
- VAE decode 峰值中间张量；
- PyTorch caching allocator 保留内存。

第一阶段必须记录以下时点的 `allocated`、`reserved` 和设备空闲显存：

1. 空进程；
2. 只加载 Reasoner；
3. Reasoner 完成 graph capture；
4. 再加载 Generator；
5. Generator warmup 后；
6. 两者并发时的峰值。

初期应显式限制 vLLM KV cache 容量，而不是让 vLLM 使用默认比例占满剩余显存。具体参数需等目标 vLLM 版本确定后固定，并写入实验元数据。

## 10. CUDA Graph 验证要求

不能只从配置项判断 graph 是否生效。需要三层证据：

### 10.1 配置和日志

记录：

- vLLM 版本；
- `enforce_eager`；
- `cudagraph_mode`；
- capture sizes；
- attention backend；
- graph downgrade/fallback 警告。

### 10.2 Nsys 时间线

decode 稳态区间应观察到：

- 重复的 `cudaGraphLaunch`；
- graph replay 后的 kernel 序列；
- Reasoner 与 Generator 位于不同 stream；
- 两组 kernel 在时间线上真实重叠；
- 不存在贯穿共置区间的默认 stream wait 或 device synchronization。

### 10.3 行为对照

至少保留：

```text
vLLM eager
vLLM graph
当前 Cosmos static-KV eager
当前 Cosmos static-KV graph
```

这里只比较相同 prompt、相同生成 token 数、相同精度和相同采样设置。否则不能把差异归因于 KV 或 CUDA Graph。

## 11. 正确性验证

集成后不能只比较运行时间。至少验证：

- tokenizer 输入 token IDs；
- prefill 后首 token logits；
- 固定 seed/greedy 下的生成 token 序列；
- eager 与 graph 的 token 一致性；
- 独占与共置时的 token 一致性；
- Reasoner 输出能被 Generator 正确解析；
- Generator 输出 shape、dtype、NaN/Inf；
- 固定输入下的生成结果或 latent 误差。

Paged KV 与原 Cosmos static-KV 的物理布局不同，不能要求内部 tensor 逐元素同构；应在模型可观察边界比较 logits、tokens 和最终输出。

## 12. 性能实验矩阵

### 12.1 独占基线

分别测：

- Reasoner full SM eager；
- Reasoner full SM graph；
- Generator full SM；
- Reasoner restricted SM；
- Generator restricted SM。

每项区分 cold、warmup 和 measured repeats。正式结论使用 warm measured 数据，同时保留 cold 作为部署启动成本。

### 12.2 并发基线

按顺序增加控制能力：

1. 同进程、串行；
2. 同进程、双线程、双 stream、不加 GC；
3. 同进程、双线程、双 stream、加 GC；
4. GC 下 eager Reasoner；
5. GC 下 graph Reasoner。

### 12.3 双向影响矩阵

对每个 SM 分区记录：

```text
decode_alone
decode_colocated
generator_alone
generator_colocated
decode_slowdown
generator_slowdown
serial_host_wall
colocated_host_wall
throughput_gain
```

配平需要区分两种目标：

- Reasoner 覆盖 Generator 全程；
- Generator 覆盖 Reasoner 全程。

不能只测一组“看起来接近”的比例后同时推断双向影响。

## 13. 分阶段实施与验收标准

### 阶段 0：冻结现有环境

目标：获得可恢复的 pre-vLLM 环境和代码快照。

容器镜像建议：

```bash
docker commit --pause=true \
  cosmos3-clean \
  cosmos3-colocation-base:20261018

docker tag \
  cosmos3-colocation-base:20261018 \
  cosmos3-colocation-base:pre-vllm
```

注意：`docker commit` 不保存 bind mount 内容。仓库代码必须单独通过 Git 保存，至少检查：

```bash
cd /data/cjh/cosmos3/cosmos3-workspace/cosmos-framework
git status --short --untracked-files=all
git diff --stat
```

验收：可以从基线镜像重新启动，并使用同一挂载仓库复现当前 Cosmos 实验环境。

### 阶段 1：Reasoner-only vLLM 验证

目标：只加载 Reasoner，固定输入，跑通 eager 和 graph。

验收：

- 输出正确；
- Paged KV 有运行时证据；
- graph replay 有 Nsys 证据；
- 记录版本、attention backend、显存和绝对时间。

### 阶段 2：同进程双后端串行

目标：一个 Python 进程同时加载 ReasonerBackend 和 GeneratorBackend，并串行完成 agentic T2V/T2I。

验收：

- 没有额外 GPU worker 进程；
- 两个模型均可运行；
- 输出正确；
- 显存稳定；
- 串行 host wall 与分阶段 GPU 时间可解释。

### 阶段 3：双 stream 并发，不加 GC

目标：验证 CPU 调度和 CUDA stream 不阻止真实 overlap。

验收：Nsys 显示不同 stream 上存在 GPU kernel 重叠，且没有意外的全局同步。

### 阶段 4：加入 Green Context

目标：分别给 Reasoner 和 Generator 分配 SM，并保持 graph/stream 正确。

验收：

- 实际 SM 数符合分区量化结果；
- graph replay 仍生效；
- 正确性不变；
- 独占 scaling 与共置 slowdown 可重复。

### 阶段 5：稳态流水线

目标：运行至少多个连续请求，使 `Reasoner(i+1)` 与 `Generator(i)` 重叠。

验收：

- 数据依赖正确；
- measured 请求中大部分处于稳态 overlap；
- 报告真实 N-request host wall；
- 吞吐收益相对同后端串行基线计算。

## 14. 失败回退路径

### 14.1 vLLM 强制多进程

先尝试 in-process executor 或直接使用 model runner。如果仍不可避免，则该版本只作为双进程基线，不继续声称可以进行同 context GC 控制。

### 14.2 依赖冲突

不要直接破坏 `pre-vllm` 镜像。所有安装在派生容器中进行；记录精确 wheel、CUDA、PyTorch、FlashAttention 和 vLLM 版本，成功后转成 Dockerfile。

### 14.3 显存不足

依次考虑：

1. 限制 vLLM KV block pool；
2. 减少 graph capture sizes；
3. 只加载所需 Reasoner tower；
4. 使用较小模型或较低精度完成机制验证；
5. 最后才考虑双进程或双 GPU。

### 14.4 CUDA Graph 无法绑定目标 stream/GC

保留 vLLM eager + Paged KV 路径作为中间基线。先完成正确共置，再单独解决 graph capture/replay，不同时修改 attention、KV 和调度。

### 14.5 Generator 内部同步阻止重叠

通过 Nsys 找到同步调用或默认 stream 依赖，只对明确阻塞点加 hook。不要先大范围重写 Generator。

## 15. 工程产物规划

建议逐步新增，而不是立即一次完成：

```text
profiling_tools/unified_runtime/
├── __init__.py
├── interfaces.py
├── resources.py
├── scheduler.py
├── reasoner_vllm_backend.py
├── cosmos_generator_backend.py
├── metrics.py
└── correctness.py

profiling_tools/runners/run_unified_agentic_pipeline.py
profiling_tools/runners/profile_unified_colocation.py
profiling_tools/analysis/summarize_unified_colocation.py
```

运行结果建议放到：

```text
profiling_results/unified_runtime/
├── manifests/
├── correctness/
├── reasoner_only/
├── serial/
├── concurrent_no_gc/
├── concurrent_gc/
├── impact_matrix/
└── nsys/
```

每次运行的 manifest 至少包含：

- Git commit 和 dirty 状态；
- 容器镜像 ID；
- Python、PyTorch、CUDA、vLLM 版本；
- GPU 型号和驱动；
- 模型 checkpoint；
- attention backend；
- graph mode 和 capture sizes；
- KV block 配置；
- requested/actual SM；
- seed、prompt、token 数和 Generator 参数；
- warmup/repeat 设置。

## 16. 当前下一步

下一步只做阶段 0 和阶段 1：

1. 保存 `cosmos3-clean` 为 `pre-vllm` 基线镜像；
2. 保存挂载仓库当前代码状态；
3. 从基线镜像启动派生开发容器；
4. 确认目标 CUDA/PyTorch 对应的 vLLM 安装版本；
5. 先运行 Reasoner-only in-process 验证；
6. 用日志和 Nsys 确认 Paged KV、attention backend 和 CUDA Graph 的实际路径。

在阶段 1 完成前，不修改 Generator 共置 runner，也不开始完整影响矩阵测试。
