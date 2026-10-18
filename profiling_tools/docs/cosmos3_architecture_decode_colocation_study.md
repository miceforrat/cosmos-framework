# Cosmos 3 架构、Reasoner Decode 优化与生成任务共置实验

本文档重新组织 Cosmos 3 架构、agentic 工作流、Reasoner decode 改造、SM scaling 和共置实验。它是一份独立主线说明，不替代下列按时间记录的调试文档：

- `profiling_tools/docs/cosmos3_reasoner_decode_colocation.md`
- `profiling_tools/docs/archive/decode_t2v_colocation_design.md`
- `profiling_tools/docs/cold_warm_sm_scaling.md`
- `profiling_tools/docs/nsys_ncu.md`

全文区分三类陈述：

- **官方事实**：来自 Cosmos 3 技术报告、官方模型文档或上游框架文档；
- **代码事实**：由当前工作树中的实现直接确认；
- **实验观察**：由仓库内保留的 CSV、日志和图片得到；
- **解释/推断**：与观察一致，但尚未由单变量实验或 profiler 指标完成因果证明。

除非特别说明，仓库相对路径均以 `/workspace/cosmos-framework` 为根；宿主机对应根目录是：

```text
/data/cjh/cosmos3/cosmos3-workspace/cosmos-framework
```

## 1. Cosmos 3 的架构设计与模型路径复用

### 1.1 统一 Mixture-of-Transformers

Cosmos 3 官方将模型描述为统一的 Mixture-of-Transformers（MoT）：同一模型同时容纳：

- 自回归 Reasoner：对文本/多模态理解 token 做 causal attention，并进行 next-token prediction；
- diffusion Generator：对图像、视频、音频或动作 token 做双向/full attention，并执行生成。

官方模型文档称两种模式共享 transformer weights、multimodal attention layers 和统一的 3D mRoPE。技术报告进一步说明，同一次 MoT 前向中存在两种不同的可见性要求：Reasoner query 只看 causal Reasoner 上下文；Generator query 可以看本样本完整的 Reasoner 与 Generator token 上下文。

来源：

- [Cosmos 3 官方模型参考](https://github.com/NVIDIA/cosmos/blob/main/docs/reference/models.md)
- [Cosmos 3 技术报告](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- [Cosmos 3 官方项目页](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)

### 1.2 “共享模型”不等于“每一组矩阵参数完全相同”

当前代码中的 Reasoner 和 Generator 复用同一个层堆栈与运行框架，但每个 MoT layer 内同时存在 pathway-specific 参数。以 attention 为例：

```text
Reasoner / understanding pathway
  q_proj, k_proj, v_proj, o_proj

Generator pathway
  q_proj_moe_gen, k_proj_moe_gen, v_proj_moe_gen, o_proj_moe_gen
```

MLP 和 normalization 也存在 generation pathway 版本，例如 `mlp_moe_gen`。因此更准确的说法是：

1. 两条路径属于同一个 checkpoint、同一组 MoT blocks 和统一 token/position 表示；
2. 它们在每层联合组织并发生跨路径 attention；
3. 但不是所有 projection、MLP 和 norm 参数都由两条路径逐元素共用。

对应实现：

```text
cosmos_framework/model/generator/mot/unified_mot.py
cosmos_framework/model/generator/mot/attention.py
cosmos_framework/model/generator/mot/causal_attention.py
```

### 1.3 四类任务如何复用路径

本项目早期 profile 的四类任务是 T2I、T2V、Action Policy 和 Video-to-Text/Video Reasoner。它们复用的是同一个 Cosmos 3 模型框架，但进入的主路径不同：

| 任务 | 主要前向路径 | 主要可观测组件 |
|---|---|---|
| T2I | Generator | `condition_prepare`、`diffusion_sampling`、`vae_decode` |
| T2V | Generator | `condition_prepare`、`diffusion_sampling`、`vae_decode` |
| Action | Generator/action head | `condition_prepare`、`diffusion_sampling`、`vae_decode`、`action_postprocess` |
| Video-to-Text | vision encoder + Reasoner | `vision_encoder`、`reasoner_prefill`、`reasoner_decode` |

需要特别避免一种误解：Generator 在 diffusion sampling 中使用 Reasoner token 作为条件，不意味着它必须在每个 diffusion layer/step 重新执行一次完整的自回归 Reasoner decode。生成路径会使用文本上下文及其层内表示/KV，但 Generator query 和 Reasoner query 使用各自的 projection/MLP 路径。是否重算、缓存到什么粒度，还受具体 inference memory/cache 路径控制。

### 1.4 对共置实验的意义

路径复用带来两点：

- 权重可以在一个常驻 Cosmos 模型实例内复用，无需为 LLM 和 Generator 各复制一个完整 checkpoint；
- 运行时状态不能简单混用。Reasoner KV、Generator latent、scheduler state、CUDA stream 和 Green Context 都必须按请求/阶段隔离。

当前共置 runner 使用同一进程、同一模型实例、两个 CPU worker 和两个不重叠 GC stream。这与两个独立 serving 进程各自加载模型不同，也不是同一请求内有数据依赖的 `Reasoner -> Generator` 串行执行。

## 2. 选用的 agentic 场景

### 2.1 Prompt upsampling + T2I/T2V

最短串行链路是：

```text
短用户 prompt
  -> Reasoner/LLM prompt upsampler
  -> 结构化长 prompt
  -> T2I 或 T2V Generator
```

本地脚本：

```text
profiling_tools/runners/run_local_agentic_t2i.py
profiling_tools/runners/run_local_agentic_t2v.py
```

本地实现用 Cosmos Reasoner 代替官方 demo 默认依赖的远程/闭源 LLM。它适合研究如下 pipeline overlap：当第 `i` 个请求进入 Generator 时，第 `i+1` 个请求的 prompt upsampling 可以在另一个 GC 中执行。

### 2.2 T2I refinement

官方 `Cosmos3-Super-Text2Image` agentic demo 并不只是一次 prompt expansion。其主结构包含：

```text
prompt upsampler / rewriter
  -> image generator（每轮可生成多个候选）
  -> VLM quality judge / critic（看生成图像）
  -> 根据反馈重写 prompt
  -> 下一轮生成，直到 early stop 或达到 max_iterations
```

官方脚本和入口：

- [官方 agentic upsampling `run.py`](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image/blob/main/agentic_upsampling/run.py)

因此严格意义上的 T2I refinement 是**视觉闭环**：critic/judge 必须看到已生成图像，反馈再影响下一轮 prompt 或候选选择。

### 2.3 T2V refinement

T2V 可采用同样的闭环抽象：

```text
prompt -> upsampler -> T2V -> video-aware critic -> rewrite -> T2V
```

但当前本地 `run_local_agentic_t2v.py --with-critic` 中的 critic 是 caption refiner：它读取 upsampler 输出的 `temporal_caption`，并没有读取尚未生成的视频。它可以用于增加 Reasoner workload 或研究 prompt rewrite，但不能被称为经过视频质量反馈的完整 T2V refinement。

这个区分会影响实验解释：

- prompt-only critic 可以与 Generator 前后形成文本 pipeline；
- visual critic 还包含视频/图像编码和多模态 prefill，负载结构、显存占用和 scaling 都不同；
- 当前影响矩阵只研究 Reasoner decode 与 Generator 的并发，不代表完整 critic/refinement loop 的端到端收益。

## 3. Cosmos 3 原始 attention 设计及其灵活性

### 3.1 技术报告中的 two-way flat attention

技术报告第 5.2.2 节给出的动机是：同一次 MoT forward 内必须同时表达两种 attention mask：

- Reasoner：只在 Reasoner tokens 上做 causal attention；
- Generator：对本样本的 `[Reasoner, Generator]` tokens 做 bidirectional attention。

报告指出，直接用 FlexAttention 等通用算子表达这种异构 mask 虽然正确，但 mask 结构对 kernel 不透明，会做相当于 padding 的无效工作，降低 tensor-core 利用率并增加 memory-bandwidth 压力。

官方方案把计算拆成两个 variable-length kernel launch：

1. Reasoner queries 对 Reasoner K/V 的 causal varlen SDPA；
2. Generator queries 对同一 sample 内 `[Reasoner, Generator]` K/V 的 bidirectional varlen attention。

为避免不同 sample 串扰，K/V 按 sample 交错组织为：

```text
[R0, G0, R1, G1, ..., Rn, Gn]
```

报告称该方法在 Cosmos3-Nano 的训练端到端吞吐上相对 FlexAttention baseline 提升 22%。这是**训练/联合 MoT forward 的官方结果**，不能直接当作 standalone AR decode 的加速数字。

### 3.2 联合 MoT attention 与 standalone Reasoner decode 必须分开

当前 `generate_reasoner_text` 的 standalone decode 使用 `MoTAttention.reasoner_forward()`：

```text
one-token query
  -> Reasoner q/k/v projections
  -> append/update Reasoner KV cache
  -> attention over cached Reasoner K/V
```

原始 eager 路径中，动态 cache 每层每 token 执行：

```python
cached_k = torch.cat([cached_k, k], dim=1)
cached_v = torch.cat([cached_v, v], dim=1)
```

然后调用 `imaginaire_attention`。`imaginaire_attention` 是按硬件、dtype、shape、causal/varlen 和 GQA 条件选择 fused backend 的 dispatcher，不是朴素 Python attention。

所以需要区分：

- 技术报告 two-way flat attention：解决联合 Reasoner+Generator forward 中的异构 mask；
- standalone Reasoner decode：解决不断增长的单 token AR KV cache；
- 本项目的 `flash_attn_with_kvcache`：专门服务固定地址 KV 上的 one-token decode。

### 3.3 原始设计提供的灵活性

动态 KV + dispatcher 路径的优势是：

- prompt 长度、batch shape、生成长度可以动态变化；
- EOS early exit 自然；
- 支持 sampling、top-k/top-p、repetition/presence penalty；
- 每个请求拥有独立 cache，对并发请求更容易隔离；
- attention backend 可以按平台/shape 选择或 fallback；
- 更容易扩展分布式、FSDP 和不同模型变体。

代价是：

- `torch.cat` 随序列增长反复分配并复制历史 K/V；
- tensor 地址和 shape 每 token 改变，不适合直接捕获为一个固定 CUDA Graph；
- Python 每 token 提交大量 kernel，短 kernel 之间可能出现 host launch gap。

这不是“官方 attention 没有优化”，而是 reference/general path 优先保留动态语义与平台适配能力。

## 4. Reasoner decode 优化及其局限

### 4.1 当前改造包含三个同时发生的变化

当前 graph decode 不是只加了一层 CUDA Graph。它同时改变：

1. **KV 管理**：动态 `torch.cat` 改为预分配 fixed-capacity KV；
2. **attention backend**：`imaginaire_attention` 改为 `flash_attn_with_kvcache`；
3. **提交方式**：Python 逐 token eager launch 改为 one-token CUDA Graph replay。

核心实现：

```text
cosmos_framework/model/generator/mot/unified_mot.py
```

重要对象/函数：

```text
ReasonerKVCache.make_static()
MoTAttention.reasoner_forward()
_PersistentReasonerGraph
_reasoner_cuda_graph_decode()
_impl_generate_reasoner_text()
```

这意味着 eager 与 graph 的绝对时间差不能全部归因于 CUDA Graph。当前已经加入中间基线：

```text
static KV + flash_attn_with_kvcache + eager token loop
```

该模式由 `COSMOS_REASONER_STATIC_KV_EAGER=1` 启用，代码标签为 `static_eager`。它与 graph 路径使用相同的 fixed-capacity KV 和 `flash_attn_with_kvcache`，但保留 Python one-token loop，不执行 capture/replay。因此三点对照应解释为：

```text
eager -> static_eager：KV 管理与 attention backend 的合并影响
static_eager -> graph：在相同 KV/backend 下，CUDA Graph capture/replay 的影响
eager -> graph：三项改造的端到端合并影响
```

同一输入、1350 个生成 token、相同 token SHA-256、相同 full-SM runner 的正式结果为：

| 实现 | Decode（s） | 相对上一阶段 speedup | 相对原始 eager speedup |
|---|---:|---:|---:|
| 原始 eager | 25.786 | 1.000x | 1.000x |
| static_eager | 18.838 | 1.369x | 1.369x |
| CUDA Graph | 6.117 | 3.080x | 4.216x |

按端点时间计算：

- KV 管理与 attention backend 合并改造使 25.786 s 降至 18.838 s，降时 26.9%；
- 在相同 static KV/backend 上加入 CUDA Graph，使 18.838 s 降至 6.117 s，继续降时 67.5%；
- 完整改造相对原始 eager 降时 76.3%。

这里的百分比是分阶段 wall/GPU-region 时间差，不应当相加；两阶段还可能改变 kernel shape、调度和 cache 行为，因此也不等价于可线性相加的独立贡献率。

### 4.2 当前路径到底何时“退化/回到”原始 attention

当前实现没有 vLLM 那样的自动 capability dispatcher。实际行为是：

| 条件 | 实际行为 |
|---|---|
| `COSMOS_REASONER_CUDA_GRAPH` 未设为 `1` | 走原始 dynamic KV + `imaginaire_attention` eager 路径 |
| CUDA Graph 开启，greedy、无 penalty、单进程 | 走 static KV + `flash_attn_with_kvcache` + graph |
| CUDA Graph 开启，同时 `do_sample=True` | 直接 `ValueError`，不会静默 fallback |
| CUDA Graph 开启，同时 repetition/presence penalty 非默认 | 直接 `ValueError`，不会静默 fallback |
| CUDA Graph 开启，同时 distributed 已初始化 | 直接 `ValueError`，不会静默 fallback |
| static KV 收到多 token incremental input | 直接报错；仅支持 `T == 1` decode |
| static capacity 不足或请求 shape 与 persistent graph 不符 | 直接报错或需要另一份 graph；不会复用不兼容 graph |

因此，“优化在某些 decode 情况下退化成原始 attention”对当前代码并不准确。更准确的是：

- 关闭优化时显式使用原始路径；
- 开启优化后，不支持的语义多数被拒绝；
- 目前没有生产级自动 fallback/graph bucketing/slot allocator。

### 4.3 等价性边界与风险

已经验证的范围是当前固定 workload 下的 greedy single-process decode。non-fixed-length graph sweep 的各个 SM 点得到相同 token SHA-256：

```text
fd71c838c5fc98cc0632c94949034e13b19fde636ed48c3fc459ec382da00021
```

正确性日志：

```text
profiling_results/correctness/eager_full_sm.log
profiling_results/correctness/graph_full_sm.log
```

但单一 token hash 不能证明普遍等价。风险包括：

1. `imaginaire_attention` 与 `flash_attn_with_kvcache` 的 reduction/order 不同，通常只能期待数值接近；greedy argmax 在 logit 很接近时可能翻转；
2. 当前 graph 只覆盖 greedy、无 penalty；sampling RNG、top-k/top-p 和 penalties 尚未捕获；
3. 当前 prototype 只支持单进程，不能外推到 FSDP/TP/多 rank；
4. KV capacity、batch、prompt 长度、dtype、device、stream 和 EOS 配置进入 graph key；shape/语义变化需要新的 graph；
5. persistent graph 持有固定地址和一份静态 KV 状态，当前没有 serving 所需的 request slot pool、并发锁、淘汰策略和内存上限管理；
6. EOS 检查发生在 replay 外。non-fixed-length 每 token 检查 `finished.all()`，仍可能产生 device-to-host sync；
7. graph 内 `cache_seqlens.add_(1)` 与 host 侧 `static_seq_len` bookkeeping 必须严格一致，off-by-one 会让 KV 写入/读取错位而不一定立即崩溃；
8. `flash_attn_with_kvcache` 的 GQA、RoPE position、causal=False one-token 语义必须持续由测试覆盖；one-token query 位于最右端，因此不需要 causal mask，但这个结论不能外推到多-token incremental decode。

### 4.4 建议补充的等价性测试矩阵

后续至少覆盖：

- 多种 prompt 长度和 `max_new_tokens`；
- 有 EOS 与无 EOS、早停与达到 token cap；
- batch size 1 与大于 1；
- 文本 prompt、图像输入、视频输入对应的 mRoPE；
- logits/hidden-state 逐 token 容差对比，而不只是最终 token hash；
- graph 连续复用多请求，确认不会残留前一请求 KV；
- capacity 边界和 graph-key 不匹配；
- 若未来支持 sampling，固定 seed 下 RNG 状态与分布测试；
- 若未来支持多 rank，collective 次序与全 rank EOS 一致性。

## 5. Decode CUDA Graph 改造及其与 LLM serving 的关系

### 5.1 本项目的捕获与 replay

当前实现先执行普通 prefill，得到第一枚 token 和每层 KV；随后：

1. 把动态 KV 复制到 `prompt_len + max_new_tokens - 1` 容量的固定 buffer；
2. 建立固定地址的 `static_input`、position ids、finished/EOS 状态；
3. eager warmup 两次以 materialize lazy kernel/allocator 状态；
4. 捕获一个完整 one-token decode step：所有 layer forward、LM head、argmax、EOS/pad、下一 token 回写、KV sequence length 递增；
5. Python 循环只负责 `graph.replay()` 和把静态输出复制进 output buffer；
6. persistent 模式按 model、device、GC stream、shape、capacity、EOS 等字段缓存 graph，并在新请求前复制 fresh prefill KV。

PyTorch 官方文档指出，CUDA Graph replay 重用相同 kernel 和参数，尤其要求相同内存地址、静态 shape/控制流；它的主要收益是用一次 `cudaGraphLaunch` 跳过 Python、C++ 和 CUDA driver 的逐 kernel dispatch 开销。这与本项目的设计目标一致。

来源：

- [PyTorch CUDA Graphs / CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)

### 5.2 必须绑定 Green Context stream

早期实现没有向 `torch.cuda.graph()` 显式传入当前 GC stream，capture 被 PyTorch 切换到 process-global side stream，最终 graph 实际运行在 primary context。那一阶段的 graph GC scaling/共置结论无效。

当前修复是：

```python
capture_stream = torch.cuda.current_stream()
with torch.cuda.graph(
    graph,
    stream=capture_stream,
    capture_error_mode="thread_local",
):
    ...
```

graph cache key 也包含 `stream_ptr`，防止在一个 GC stream 捕获后错误地在另一个 partition 上复用。

### 5.3 是否借鉴了 LLM serving

原则上是同一类 LLM decode 优化，但当前实现是为 Cosmos profiling 写的最小 prototype，并非直接接入 vLLM/SGLang runner。

共同点：

- pure decode 每次 query length 为 1，shape 比 prefill 更稳定；
- 预分配 KV/输入输出 buffer，保持地址稳定；
- 先 warmup/capture，再 replay；
- 用 batch/shape/运行模式作为 graph dispatch key；
- 不兼容的 attention/backend/shape 需要 eager fallback 或另一份 graph。

vLLM 的官方设计把 uniform decode 单独识别，提供 `FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE` 等模式，并由 dispatcher 根据 batch descriptor 和 attention backend capability 在 full graph、piecewise graph 与 eager 之间选择。vLLM 还为不同 capture sizes 准备多份 graph。

来源：

- [vLLM CUDA Graphs 设计](https://docs.vllm.ai/en/latest/design/cuda_graphs/)

当前 Cosmos prototype 缺少的 serving 能力包括：

- 连续 batching 和不同 active request 数的 padding/bucketing；
- paged KV/slot allocator 与请求生命周期管理；
- 多 graph 内存预算和淘汰；
- attention backend capability dispatcher；
- sampling、structured output/speculative decode；
- 多请求并发安全与多 rank 支持；
- full/piecewise/eager 自动降级。

因此可以说“借鉴了 LLM serving 中 static-buffer + one-token replay 的通用方法”，不能说“已经具有 vLLM 等价的 decode runtime”。

### 5.4 CUDA Graph 改造改变了 scaling 含义

原始 eager 时间中，host launch、动态 KV 分配/复制等不随 SM 数量近似线性缩放的部分占比较大，因此很早进入平台。graph/static-KV 将这些部分显著压低后，剩余 GPU 计算占比上升，decode 对 SM 数量更敏感，平台期反而缩短。

这是 Amdahl 定律意义上的负载重构，不是同一 workload 的纯调度变化。比较 eager 与 graph scaling 时，必须同时给出绝对时间，不能只比较 normalized speedup。

## 6. T2V、T2I 与 LLM 的绝对时间和 SM scaling

### 6.1 测量口径

下表的 Generator 数字是组件 CUDA event 时间之和；Reasoner decode 数字来自对应 decode region。它们不是模型加载、输入处理、文件写出等端到端 host wall time。

`speedup` 定义为：

```text
speedup(SM) = T(full SM) / T(restricted SM)
```

因此小于 1 表示 restricted-SM 更慢。GC 请求 SM 与驱动实际给出的 SM 可能不一致，绘图必须使用 `actual_sm_count`。

### 6.2 代表性 full-SM/warm 绝对时间

当前保留数据中，Generator 的最高 GC 点为实际 168 SM，Reasoner graph 另有真实 170-SM baseline：

| workload | 实现/数据集 | 组件时间（s） | 组件和/Decode（s） |
|---|---|---|---:|
| T2I | warm measured, 168 SM | condition 0.024；diffusion 0.995；VAE 0.070 | 1.089 |
| T2V | warm measured, 168 SM | condition 1.555；diffusion 14.185；VAE 6.896 | 22.636 |
| agentic T2V generator | rich02, warm measured, 168 SM | condition 1.564；diffusion 13.805；VAE 6.889 | 22.258 |
| agentic upsampler LLM | eager, warm measured, 168 SM | prefill 0.021；decode 27.315 | decode 27.315 |
| matched upsampler LLM | eager, non-fixed, 170 SM | decode | 25.786 |
| matched upsampler LLM | static_eager, non-fixed, 170 SM | decode | 18.838 |
| agentic upsampler LLM | graph, non-fixed, 170 SM | decode | 6.117 |

原始数据：

```text
profiling_results/sm_scaling_phased_on.csv
profiling_results/agentic_t2v_rich02_scaling.csv
profiling_results/reasoner_graph_sm_scaling_early_exit/plots/reasoner_graph_sm_scaling.csv
profiling_results/correctness/eager_full_sm.log
profiling_results/static_eager_full_sm_decode.log
```

168-SM agentic 曲线中的 27.315 s 仅用于该 scaling 数据集自身。严格拆因采用后三个 matched full-SM 点：它们生成相同的 1350 tokens，并具有相同的 token SHA-256。

### 6.2.1 static_eager 中间基线复现命令

先跑一个 T2V 分区点进行正确性和运行时间快速检查：

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate

IMPLEMENTATIONS=static_eager \
REQUESTED_GENERATOR_VALUES=130 \
RUN_TAG=static_eager_smoke01 \
./profiling_tools/scripts/run_decode_t2v_impact_matrix.sh
```

确认日志中存在 `decode_implementation=static_eager`、token validation 通过且时间合理后，补跑 full-SM decode 和两个完整矩阵：

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate
export LD_LIBRARY_PATH=
export CUDA_DEVICE_MAX_CONNECTIONS=8
unset COSMOS_REASONER_CUDA_GRAPH
unset COSMOS_REASONER_CUDA_GRAPH_PERSISTENT

# static_eager full-SM decode；这项只需跑一次。
COSMOS_REASONER_STATIC_KV_EAGER=1 \
COSMOS_FULL_SM_DECODE_ONLY=1 \
COSMOS_REASONER_REPEATS=1 \
COSMOS_WARMUP_RUNS=1 \
python -m profiling_tools.runners.colocate_decode_t2v \
  2>&1 | tee profiling_results/static_eager_full_sm_decode.log

# T2V × static_eager decode：四个既有分区。
IMPLEMENTATIONS=static_eager \
RUN_TAG=static_eager_matrix01 \
./profiling_tools/scripts/run_decode_t2v_impact_matrix.sh

# T2I × static_eager decode：每个分区先以单次 T2I 自动校准 repeats，
# 再按四舍五入后的 repeats 跑正式配平矩阵。
IMPLEMENTATIONS=static_eager \
RUN_TAG=static_eager_matrix01 \
./profiling_tools/scripts/run_decode_t2i_impact_matrix.sh
```

整合新旧三种实现；汇总器会自动忽略 T2I 的 `_calibration_` 日志：

```bash
python -m profiling_tools.analysis.summarize_decode_t2v_impact_matrix \
  --input-dir profiling_results/decode_t2v_impact_matrix \
  --output profiling_results/decode_t2v_impact_matrix/summary_all_implementations.csv

python -m profiling_tools.analysis.summarize_decode_t2v_impact_matrix \
  --input-dir profiling_results/decode_t2i_impact_matrix \
  --output profiling_results/decode_t2i_impact_matrix/summary_all_implementations.csv
```

T2I 自动校准产生一次额外的 repeat=1 完整运行，这是为了避免用 eager/graph 的旧 repeat 数错误配平 `static_eager`。正式矩阵结果和 calibration 日志分开保存。

### 6.3 T2I/T2V eager Generator 曲线

四任务统一数据与图：

```text
profiling_results/sm_scaling_phased_on.csv
profiling_results/plots_phased_on/speedup_vs_sm_count_measured.png
profiling_results/plots_phased_on/speedup_vs_sm_ratio_measured.png
profiling_results/plots_phased_on/speedup_vs_sm_count_cold.png
profiling_results/plots_phased_on/speedup_vs_sm_ratio_cold.png
profiling_results/plots_phased_on/component_time_breakdown_measured.png
profiling_results/plots_phased_on/component_time_breakdown_cold.png
```

原始早期图仍保留，但测量轮数/phase 口径不如 phased-on 数据完整：

```text
profiling_results/sm_scaling.csv
profiling_results/plots/speedup_vs_sm_count.png
profiling_results/plots/speedup_vs_sm_ratio.png
profiling_results/plots/component_time_breakdown.png
```

agentic rich02 T2V 数据与图：

```text
profiling_results/agentic_t2v_rich02_scaling.csv
profiling_results/plots_agentic_t2v_rich02/local_agentic_t2v_component_time_breakdown_measured.png
profiling_results/plots_agentic_t2v_rich02/local_agentic_t2v_speedup_vs_sm_count_measured.png
profiling_results/plots_agentic_t2v_rich02/local_agentic_t2v_speedup_vs_sm_ratio_measured.png
```

观察：

- T2V 的三个主要组件均随 SM 减少显著变慢；diffusion sampling 占比最大；
- T2I 总时间短得多，因此影响矩阵中要重复运行多次 T2I 才能与一次 decode 配平；
- T2I/T2V Generator 的曲线来自 eager Generator 路径，不要与 Reasoner 的 eager/graph 标签混淆。

### 6.4 LLM eager 曲线

当前 agentic rich02 图的左图给出 Reasoner prefill/decode eager scaling，右图给出同一 workflow 的 T2V Generator scaling：

```text
profiling_results/plots_agentic_t2v_rich02/local_agentic_t2v_speedup_vs_sm_ratio_measured.png
profiling_results/plots_agentic_t2v_rich02/local_agentic_t2v_speedup_vs_sm_count_measured.png
```

对应 CSV：

```text
profiling_results/agentic_t2v_rich02_scaling.csv
```

eager decode 在较低 SM 后很快接近 full-SM performance。这个现象与较大的 host/KV 管理固定开销一致，但不能仅凭 scaling 曲线证明具体比例由 launch、KV copy 或 memory bandwidth 中哪一项造成。

### 6.5 LLM graph 曲线

当前应优先使用 non-fixed-length/early-exit 目录：

```text
profiling_results/reasoner_graph_sm_scaling_early_exit/plots/reasoner_graph_sm_scaling.csv
profiling_results/reasoner_graph_sm_scaling_early_exit/plots/reasoner_graph_decode_time_vs_sm.png
profiling_results/reasoner_graph_sm_scaling_early_exit/plots/reasoner_graph_speedup_vs_sm_ratio.png
profiling_results/reasoner_graph_sm_scaling_early_exit/full_sm.log
profiling_results/reasoner_graph_sm_scaling_early_exit/requested_reasoner_sm*.log
```

关键点：

| actual Reasoner SM | decode（s） | `Tfull/Tsm` |
|---:|---:|---:|
| 2 | 138.870 | 0.044 |
| 10 | 29.689 | 0.206 |
| 18 | 17.671 | 0.346 |
| 34 | 10.574 | 0.579 |
| 66 | 7.059 | 0.867 |
| 98 | 6.324 | 0.967 |
| 130 | 5.984 | 1.022 |
| 162 | 6.010 | 1.018 |
| 170 | 6.117 | 1.000 |

130--162 SM 略快于单次 170-SM baseline 的 1.0 属于约 2% 的测量波动/基线差异，不能解释为限制 SM 本身必然加速。

历史 fixed-length graph 数据仍保留用于追溯：

```text
profiling_results/reasoner_graph_sm_scaling/plots/reasoner_graph_sm_scaling.csv
profiling_results/reasoner_graph_sm_scaling/plots/reasoner_graph_decode_time_vs_sm.png
profiling_results/reasoner_graph_sm_scaling/plots/reasoner_graph_speedup_vs_sm_ratio.png
profiling_results/reasoner_graph_sm_scaling/plots/reasoner_graph_scaling_vs_sm_ratio.png
```

后续实验应优先 non-fixed-length，除非目的明确是 attribution-only 固定步数测试。

### 6.6 LLM static_eager 曲线

`static_eager` 使用与 graph 相同的 static KV 和 `flash_attn_with_kvcache`，但保留 Python one-token eager loop。它的 sweep 将 Generator（GC first partition）按 `8,16,...,168` SM 请求，Reasoner 使用驱动返回的余数；绘图横轴始终取 `actual_reasoner_sm`，不使用名义余数：

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate

bash profiling_tools/scripts/run_reasoner_static_eager_sm_scaling.sh

python -m profiling_tools.analysis.plot_reasoner_graph_sm_scaling \
  --implementation static_eager \
  --input-dir profiling_results/reasoner_static_eager_sm_scaling \
  --output-dir profiling_results/reasoner_static_eager_sm_scaling/plots
```

已生成产物：

```text
profiling_results/reasoner_static_eager_sm_scaling/full_sm.log
profiling_results/reasoner_static_eager_sm_scaling/requested_reasoner_sm*.log
profiling_results/reasoner_static_eager_sm_scaling/plots/reasoner_static_eager_sm_scaling.csv
profiling_results/reasoner_static_eager_sm_scaling/plots/reasoner_static_eager_decode_time_vs_sm.png
profiling_results/reasoner_static_eager_sm_scaling/plots/reasoner_static_eager_speedup_vs_sm_ratio.png
```

本次 non-fixed-length sweep 的关键点为：

| actual Reasoner SM | decode（s） | `Tfull/Tsm` |
|---:|---:|---:|
| 2 | 141.322 | 0.132 |
| 10 | 32.272 | 0.576 |
| 18 | 20.677 | 0.900 |
| 34 | 21.592 | 0.862 |
| 50 | 19.198 | 0.969 |
| 82 | 18.788 | 0.990 |
| 170 | 18.603 | 1.000 |

所有点的生成 token SHA-256 均为：

```text
fd71c838c5fc98cc0632c94949034e13b19fde636ed48c3fc459ec382da00021
```

曲线表明 static_eager 在约 50 SM 已达到 full-SM 性能的 96.9%，18 SM 时也达到 90.0%；除极低 SM 外，它仍保留很长的平台区。相比之下，graph 路径约到 98 SM 才达到 96.7%。结合 matched full-SM 时间 `18.838/18.603 s`（两轮 static_eager）与 graph 的 `6.117 s`，这支持如下解释：static KV/backend 已降低一部分 KV 管理成本，但 Python/driver 逐 token 提交仍占据大量不随 SM 数量缩放的时间；CUDA Graph 移除这部分开销后，剩余 GPU 工作占比提高，因此曲线对 SM 数量明显更敏感。

34 SM 点的 21.592 s 比 18 SM 的 20.677 s 更慢，且 26--42 SM 有非单调波动。这些差异应视为单轮测量噪声/运行状态波动，不能解释为增加 SM 会稳定降低性能；若需要对平台边界做置信区间，应对每个点增加 repeats。

## 7. T2V × LLM 与 T2I × LLM 影响矩阵

### 7.1 指标与实验结构

每个矩阵点都先在相同 GC partition 上分别测：

```text
decode_alone
generator_alone
```

再让两者通过 barrier 同时进入各自 GC stream，测：

```text
decode_colocated
generator_colocated
colocated_makespan = max(两侧 GPU region duration)
```

指标：

```text
decode_slowdown    = decode_colocated / decode_alone
generator_slowdown = generator_colocated / generator_alone
serial_gpu_sum     = decode_alone + generator_alone
throughput_gain    = serial_gpu_sum / colocated_makespan
```

`host_wall` 由 `time.perf_counter()` 实测，不是取两侧时间的 max。`serial_host_wall` 采用 decode region host wall 加完整 generator host wall；包含 prefill 的 `serial_end_to_end_wall_diagnostic` 只用于诊断，不能作为同口径吞吐基线。

runner 和汇总脚本：

```text
profiling_tools/colocate_decode_t2v.py
profiling_tools/scripts/run_decode_t2v_impact_matrix.sh
profiling_tools/scripts/run_decode_t2i_impact_matrix.sh
profiling_tools/analysis/summarize_decode_t2v_impact_matrix.py
```

### 7.2 T2V × LLM

正式汇总：

```text
profiling_results/decode_t2v_impact_matrix/summary.csv
profiling_results/decode_t2v_impact_matrix/summary_all_implementations.csv
profiling_results/decode_t2v_impact_matrix/*_matrix01.log
```

| Decode | actual R:G SM | alone：Decode/T2V（s） | slowdown：Decode/T2V | makespan（s） | throughput gain |
|---|---:|---:|---:|---:|---:|
| eager | 10:160 | 33.970 / 22.248 | 1.144 / 1.031 | 38.865 | 1.446x |
| eager | 18:152 | 30.447 / 23.063 | 1.055 / 1.034 | 32.132 | 1.665x |
| eager | 34:136 | 29.433 / 25.169 | 1.062 / 1.033 | 31.245 | 1.747x |
| eager | 50:120 | 29.068 / 27.365 | 1.053 / 1.035 | 30.618 | 1.843x |
| graph | 10:160 | 29.727 / 22.355 | 1.148 / 1.024 | 34.140 | 1.525x |
| graph | 18:152 | 17.765 / 23.098 | 1.204 / 1.029 | 23.760 | 1.720x |
| graph | 34:136 | 10.603 / 25.254 | 1.170 / 1.016 | 25.655 | 1.398x |
| graph | 50:120 | 8.395 / 27.461 | 1.207 / 1.013 | 27.826 | 1.289x |
| static_eager | 10:160 | 32.343 / 22.249 | 1.153 / 1.028 | 37.279 | 1.464x |
| static_eager | 18:152 | 21.992 / 23.096 | 1.191 / 1.032 | 26.192 | 1.721x |
| static_eager | 34:136 | 20.662 / 25.260 | 1.134 / 1.029 | 26.002 | 1.766x |
| static_eager | 50:120 | 20.204 / 27.397 | 1.070 / 1.023 | 28.041 | 1.698x |

可以直接确认：

- T2V slowdown 在全部点仅约 1.3%--3.5%；
- eager decode slowdown 为约 5.3%--14.4%；graph decode 为约 14.8%--20.7%，主要干扰落在 decode 一侧；
- eager 50:120 最接近平衡，吞吐收益最高 1.843x；
- graph 在同一 partition 下 decode 已显著缩短，所以 34:136 和 50:120 不再配平，吞吐收益下降不能解释为 graph 共置能力更差；
- graph 18:152 的 1.720x 是当前表内 graph 较好的折中，但仍不是严格双向配平的全局最优搜索。
- static_eager 在 18:152 最接近平衡；其 1.721x 与 graph 同分区相同，但两者的绝对构成不同；
- 在 34-SM Reasoner 点，decode-alone 从原始 eager 的 29.433 s 降至 static_eager 的 20.662 s，再降至 graph 的 10.603 s。这证明 KV/backend 改造和 graph replay 都贡献了显著降时；
- static_eager 的 T2V slowdown 仍只有 2.3%--3.2%，decode slowdown 为 7.0%--19.1%。

关于原因只能谨慎表述：T2V 对共置不敏感而 decode 更敏感，与 decode 的小 kernel/访存/调度特征和 T2V 的大 kernel 占用模式相容；仅凭 slowdown 非对称不能证明一定是 DRAM bandwidth。严格归因还需要同区间 Nsys kernel overlap、SM active、DRAM/L2 throughput、eligible warps 和 launch gap 的联合证据。

### 7.3 T2I × LLM

T2I 单次约 1.1 s，无法与一次长 decode 配平。因此矩阵让 Generator worker 串行执行 `N` 次 T2I，使 `N × T2I` 接近该分区的 decode-alone 时间。它测量的是“一次 decode 与一批连续 T2I 的 pipeline overlap”，不是单请求 latency。

正式汇总：

```text
profiling_results/decode_t2i_impact_matrix/summary_matrix01.csv
profiling_results/decode_t2i_impact_matrix/summary_all_implementations.csv
profiling_results/decode_t2i_impact_matrix/*_matrix01.log
```

| Decode | actual R:G SM | T2I repeats | alone：Decode/T2I batch（s） | slowdown：Decode/T2I | makespan（s） | throughput gain |
|---|---:|---:|---:|---:|---:|---:|
| eager | 10:160 | 31 | 33.998 / 34.059 | 1.812 / 1.444 | 61.595 | 1.105x |
| eager | 18:152 | 27 | 30.432 / 29.615 | 1.829 / 1.487 | 55.675 | 1.078x |
| eager | 34:136 | 27 | 29.098 / 29.909 | 1.882 / 1.481 | 54.770 | 1.077x |
| eager | 50:120 | 25 | 28.922 / 28.454 | 1.808 / 1.444 | 52.282 | 1.097x |
| graph | 10:160 | 27 | 29.752 / 29.553 | 1.071 / 1.026 | 31.860 | 1.861x |
| graph | 18:152 | 16 | 17.711 / 17.863 | 1.078 / 1.017 | 19.093 | 1.863x |
| graph | 34:136 | 10 | 10.576 / 11.432 | 1.092 / 1.040 | 11.892 | 1.851x |
| graph | 50:120 | 7 | 8.368 / 7.979 | 1.092 / 1.039 | 9.137 | 1.789x |
| static_eager | 10:160 | 29 | 32.322 / 31.790 | 1.629 / 1.326 | 52.642 | 1.218x |
| static_eager | 18:152 | 19 | 21.453 / 20.976 | 1.816 / 1.341 | 38.960 | 1.089x |
| static_eager | 34:136 | 18 | 20.302 / 20.345 | 1.843 / 1.341 | 37.422 | 1.086x |
| static_eager | 50:120 | 17 | 20.156 / 19.355 | 1.821 / 1.347 | 36.706 | 1.076x |

可以直接确认：

- eager 下两侧都受到严重干扰：decode 约 1.81--1.88x，T2I batch 约 1.44--1.49x；吞吐收益仅 1.08--1.11x；
- graph 下 decode slowdown 降至 1.07--1.09x，T2I batch 降至 1.02--1.04x；吞吐收益达到 1.79--1.86x；
- static_eager 虽降低了 decode-alone 绝对时间，但 T2I 共置干扰仍接近原始 eager：decode slowdown 为 1.63--1.84x，T2I slowdown 为 1.33--1.35x，吞吐收益仅 1.08--1.22x；
- 因而 T2I 矩阵把两个因素分开了：static KV/backend 主要改善 standalone decode 绝对时间，而在相同 KV/backend 上加入 CUDA Graph replay，才消除了这组短任务共置中的大部分干扰；
- 这种差异强烈支持“eager 两个 workload 的 host/driver launch 竞争是重要因素”的解释，因为 CUDA Graph 降低逐 kernel 提交开销后矩阵显著改善；
- 原先“graph 同时改变 KV/backend，因而不能拆因”的限制已由 static_eager 基线消除：`static_eager -> graph` 保持 static KV、`flash_attn_with_kvcache`、输入、token 数和输出 hash 一致，主要实现变量就是逐 token eager launch 与 CUDA Graph replay；
- 但“CUDA Graph 是关键变量”仍不等于已经把底层机制全部定量归因于 CPU launch competition。Graph replay 同时改变 host submission、CUDA driver/front-end 调度和 kernel 到达时序；若要继续区分这些机制，仍应对 static_eager 与 graph 的相同共置区间做 Nsys 对照。这个 Nsys 对照是微观机制分析，而不再是建立中间性能基线的前置条件。

### 7.4 两个矩阵不能互相直接替代

T2V 是一次较长、连续的 Generator 请求；T2I 矩阵是一批短请求串行执行。二者对 host launch、allocator、cache locality 和请求边界的压力不同。因此：

- T2V 矩阵更接近单个长生成任务与一次 decode 的 overlap；
- T2I 矩阵更接近高频短生成队列与一次 decode 的 throughput overlap；
- T2I graph 的显著收益不能直接外推到 T2V；
- 矩阵中的 throughput gain 是同分区串行基线相对共置 makespan，不是相对“两项都独占 full GPU”的系统吞吐。

## 附录 A：结果可信度与历史数据使用规则

以下结果不应进入主结论：

1. 同一请求中 `Reasoner -> Generator` 存在数据依赖的早期伪共置；
2. FlashAttention `call_once`/首次编译/allocator 尚未 warm 的 cold 结果，除非研究目标就是 cold start；
3. 未显式绑定 GC capture stream 的早期 CUDA Graph 结果；
4. 只记录 stage-triggered overlap、但没有完整同起点 barrier 的吞吐数字；
5. fixed-length 与 non-fixed-length 混合比较但未标注 token 数的结果。

仍可用于历史诊断的日志：

```text
profiling_results/colocation_stage_sweep/eager_warm_sm136_diffusion_sampling.log
profiling_results/colocation_stage_sweep/eager_warm_sm136_vae_decode.log
profiling_results/colocation_stage_sweep/gc_graph_fixed_stream_g136_diffusion_sampling.log
profiling_results/max_connections/connections_1.log
profiling_results/max_connections/connections_8.log
profiling_results/max_connections/connections_32.log
```

## 附录 B：代码入口索引

```text
# Cosmos MoT / Reasoner
cosmos_framework/model/generator/mot/unified_mot.py
cosmos_framework/model/generator/mot/attention.py
cosmos_framework/model/generator/mot/causal_attention.py

# runtime region 与 Green Context
cosmos_framework/runtime/regions.py
profiling_tools/profiling_tools/green_context.py

# agentic 串行工作流
profiling_tools/runners/run_local_agentic_t2i.py
profiling_tools/runners/run_local_agentic_t2v.py
profiling_tools/serial_agentic_workflows.py
profiling_tools/docs/serial_agentic_workflows.md

# 共置与影响矩阵
profiling_tools/colocate_decode_t2v.py
profiling_tools/scripts/run_decode_t2v_impact_matrix.sh
profiling_tools/scripts/run_decode_t2i_impact_matrix.sh
profiling_tools/analysis/summarize_decode_t2v_impact_matrix.py

# scaling 与绘图
profiling_tools/scripts/run_reasoner_graph_sm_scaling.sh
profiling_tools/analysis/plot_reasoner_graph_sm_scaling.py
profiling_tools/analysis/plot_local_agentic_sm_scaling.py
profiling_tools/analysis/plot_local_agentic_t2v_full.py
profiling_tools/analysis/plot_phased_sm_scaling.py
```

## 附录 C：外部一手资料

- [Cosmos 3 技术报告](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- [Cosmos 3 官方模型参考](https://github.com/NVIDIA/cosmos/blob/main/docs/reference/models.md)
- [Cosmos Framework 官方仓库](https://github.com/NVIDIA/cosmos-framework)
- [Cosmos3-Super-Text2Image agentic upsampling](https://huggingface.co/nvidia/Cosmos3-Super-Text2Image/blob/main/agentic_upsampling/run.py)
- [PyTorch CUDA Graphs 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [vLLM CUDA Graphs 设计](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
