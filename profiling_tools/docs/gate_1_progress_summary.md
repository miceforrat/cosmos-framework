# Cosmos3 Edge 单进程双模型：当前进度总览

更新时间：2026-10-21。

本文记录进入并发与 Green Context 实验前已经稳定复现的环境、负载、上游优化和单进程双模型实现。
Reasoner 输出的 schema 内容偏差已记录，但按当前决定暂不修复。

## 1. 当前目标与完成度

当前目标是让 Cosmos3-Edge 的两条模型路径在同一进程、同一 GPU CUDA primary context 中共同驻留：

```text
官方 prompt-upsample 请求
        |
        v
vLLM Edge Reasoner
        |
        | 当前仅在实验语义上代表前一阶段；输出尚未接入 Generator
        v
vLLM-Omni Edge Generator --> 832x480 T2V
```

已完成：

- 统一 Reasoner 与 Generator 的依赖环境；
- 两套引擎分别跑通 Edge；
- 两套引擎在同一 PID、同一 CUDA context 中共同驻留；
- reasoner-first 和 generator-first 两种构造顺序均跑通；
- Generator-first 下 Reasoner CUDA Graph 跑通；
- Reasoner eager/graph 与 Generator eager/compile 的 2x2 稳态对照；
- SeaCache、固定 KV 预算、warmup/repeat 和显存记录。

尚未完成：

- Reasoner 输出到 Generator 输入的真实数据依赖；
- 两阶段同时执行；
- Green Context 分区、stream 调度和 overlap；
- 共置后的双向 slowdown、makespan 与吞吐收益；
- 多卡或任何并行度大于 1 的共享 parallel-state 支持。

## 2. 持久化运行环境

主仓库 `.venv` 不是本实验环境。统一环境固定在：

```text
/workspace/cosmos-framework/.runtime/vllm_omni_tf517/venv
/workspace/cosmos-framework/.runtime/vllm_omni_tf517/vllm-omni
```

版本组合：

```text
Python        3.13
PyTorch       CUDA 13.0 build
vLLM          0.31.0
vLLM-Omni     commit 03c334818305d3b1666da01cd0f67d48a4f3e7a2
Transformers  5.17.0
GPU           NVIDIA GeForce RTX 5090, 32607 MiB
```

启动方式：

```bash
cd /workspace/cosmos-framework
source .runtime/vllm_omni_tf517/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=''
```

环境由 `profiling_tools/scripts/setup_unified_vllm_omni_env.sh` 创建。上游该 Omni commit 声明
`transformers>=5.13,<5.15`，但 Edge 原生支持需要 Transformers 5.17。本实验只在隔离的 Omni
源码副本中把包元数据约束调整为 `>=5.17,<5.18`；没有修改 Cosmos 模型实现，也没有污染主 `.venv`。
Gate -1 已分别验证 Edge Reasoner 和 Edge Omni Generator 能在该组合运行，但该兼容结论不能外推到
vLLM-Omni 的所有模型和功能。

## 3. 当前固定负载

### 3.1 Reasoner / prompt upsampler

输入使用 NVIDIA Hugging Face `Cosmos3-Super-Text2Image` 的官方 agentic prompt-upsampling JSON
模板，本地固定快照为 `profiling_tools/core/official_upsampler_workload.py`。用户请求固定为：

```text
A red robot repairs a bicycle in a bright workshop while the camera slowly moves around them.
```

参数：

```text
model                     nvidia/Cosmos3-Edge
max_num_seqs              1
max_model_len             6144
max output tokens         2600
sampling temperature      0.0
prefix caching            off
KV allocation             1 GiB
measured KV capacity      9360 tokens
6144-token concurrency    1.52x
warmup / measured         1 / 3
```

实际自然停止长度：eager 2248 tokens，graph 2266 tokens。输出已能被 JSON parser 解析，但模型将
`aspect_ratio` 放入 `resolution` 内并遗漏顶层字段；该内容问题暂不修复，不影响本阶段性能负载。

### 3.2 Generator / T2V

```text
model                 nvidia/Cosmos3-Edge
output                 video
resolution             832 x 480 (W x H)
frames                 121
inference steps        35
CFG / guidance scale   6.0
seed                    0
cache                   SeaCache on
warmup / measured      1 / 3
```

当前 Generator prompt 从既有 `t2v_upsampled.json` 读取，文件不存在时使用固定 fallback prompt。
它没有读取当前进程内刚生成的 Reasoner 文本。因此当前测量是“符合部署形态的两套真实负载串行执行”，
不是数据依赖已经贯通的端到端 agentic pipeline。

## 4. 使用的上游/原框架优化

### 4.1 Reasoner：vLLM

- Paged KV cache；实验用 `kv_cache_memory_bytes=1 GiB` 固定实际 KV 池。
- `gpu_memory_utilization=0.70` 只用于 vLLM 启动准入检查，不代表分配 70% KV。
- vLLM 原生 Cosmos3-Edge model/processor 与 FlashAttention backend。
- graph 模式设置 `enforce_eager=False` 和 `FULL_AND_PIECEWISE`：同时启用 vLLM compile 与 CUDA Graph。
- 日志已经观察到 AOT compile artifact 加载、PIECEWISE/FULL capture 和 graph capture 完成。

### 4.2 Generator：vLLM-Omni / Cosmos3 diffusion

- `enforce_eager=False` 时进入 Omni diffusion 的 lazy regional `torch.compile`；本轮未把 Cosmos3
  diffusion 误写成 CUDA Graph，当前明确验证的是 compile。
- SeaCache 参数与原 Cosmos Framework cache-on 基线对齐：

```text
sea_threshold               0.25
sea_residual_order          1
sea_max_consecutive_cached  2
sea_power_exp               3.0
```

- Wan VAE lossless fast path 由上游默认启用。
- 当前没有启用 layerwise offload、VAE tiling/slicing 或 prompt embedding cache；这些不是本轮延迟
  基线的一部分。

SeaCache 会改变 diffusion 的计算复用策略，属于近似推理优化。它在全部 2x2 实验中保持开启，
因此不会污染 eager/compile 这一变量对照，但该基线不等同于 cache-off 精确推理。

## 5. 单进程双模型如何实现

入口是 `profiling_tools/runners/single_process_dual_engine.py`，兼容层是
`profiling_tools/core/omni_shared_parallel.py`。

### 5.1 禁止 vLLM EngineCore 多进程

runner 在导入 vLLM 前设置：

```text
VLLM_ENABLE_V1_MULTIPROCESSING=0
VLLM_USE_DEEP_GEMM=0
```

Reasoner EngineCore 因而留在主 Python PID 内。每个阶段之后通过 `ps`、`nvidia-smi` 和 CUDA Driver API
记录子进程、GPU PID、显存以及主线程 current CUDA context。

### 5.2 共享进程级 distributed / parallel state

vLLM 与 vLLM-Omni 都把 TP/DP/PP coordinator 放在 vLLM 进程全局变量中，直接各自初始化会冲突。
当前兼容层只支持：

```text
world size = 1
TP = DP = PP = SP = CFG = 1
无 EP
无 HSDP
```

Generator-first 是当前主路径：

1. Omni 先创建默认 process group 及 TP/DP/PP；
2. Reasoner 借用 Omni 已创建的 TP coordinator；
3. vLLM 0.31 额外需要的 PCP/DCP/ETP 槽位，在 rank 1 下临时别名到 TP；
4. 不创建第二套冲突的默认 process group。

Reasoner-first 也支持：

1. vLLM Reasoner 先拥有 TP/DP/PP；
2. 兼容层拦截 Omni diffusion worker 的 parallel initialization；
3. Omni 借用 vLLM groups，只创建自己的 diffusion DP/CFG/PP/SP 单 rank groups；
4. Omni shutdown 只销毁自己创建的 groups。

这不是通用分布式适配；任一并行度大于 1 都直接拒绝，以避免静默错误。

### 5.3 CUDA Graph coordinator 适配

Omni 的单 rank `GroupCoordinator` 缺少 vLLM 0.31 capture 所需的 `graph_capture(context)`。
兼容层只对 world-size 1 coordinator 补充该 context manager：等待当前 stream 后进入 vLLM 提供的
共享 capture stream。rank 1 没有 NCCL/custom-all-reduce capture，因此不伪造通信 capture。
该适配使 Generator-first 下的 Reasoner FULL/PIECEWISE capture 可以完成。

### 5.4 所有权与关闭顺序

Generator-first 时 process-wide distributed environment 属于 Omni。正确关闭顺序为：

```text
清除 Reasoner 借用的 PCP/DCP/ETP aliases
    -> generator.close()
    -> Reasoner EngineCore shutdown
    -> Python GC
```

这避免 Reasoner 先销毁 Omni 仍引用的 process group，以及随后发生 double destroy。

## 6. 最新 2x2 稳态结果

统一使用 2600-token cap、SeaCache on、两边各 1 次 warmup 和 3 次 measured。表内为中位数：

| Reasoner | Generator | Reasoner | Generator | 串行稳态总时间 |
|---|---|---:|---:|---:|
| eager | eager | 39.413 s | 18.253 s | 57.666 s |
| graph | eager | 7.550 s | 18.354 s | 25.904 s |
| eager | compile | 38.469 s | 16.494 s | 54.963 s |
| graph | compile | 7.513 s | 16.572 s | 24.084 s |

观察：

- Reasoner graph 相对 eager 约 5.1--5.2x；约 57 tokens/s 提升到约 300--302 tokens/s。
- Generator compile 在两种 Reasoner 模式下都约 1.107x，即稳态时间降低约 9.7%。
- 全优化相对全 eager 的串行稳态时间加速为 2.39x。
- 三轮 measured 波动小；四组均正常 shutdown，全部同 PID、单 GPU PID、同 CUDA context 检查通过。
- 峰值 `nvidia-smi` 显存约 21.2--21.7 GiB，32 GiB 5090 上仍有约 10 GiB 余量。

结果文件位于：

```text
profiling_results/gate_1_single_process/reasoner_eager_generator_eager_tok2600.json
profiling_results/gate_1_single_process/reasoner_graph_generator_eager_tok2600.json
profiling_results/gate_1_single_process/reasoner_eager_generator_compile_tok2600.json
profiling_results/gate_1_single_process/reasoner_graph_generator_compile_tok2600.json
```

## 7. 当前可信结论与下一阶段入口

当前可以确认：Edge Reasoner 与 Edge Omni Generator 能在单张 RTX 5090 上，以同一 PID、同一 CUDA
primary context 和同一进程级 distributed environment 共同驻留；Reasoner CUDA Graph、Generator
regional compile 与 SeaCache 可在该形态下同时工作，串行稳态基线已经建立。

当前不能声称：

- 两模型已经并发；
- 已经产生 GPU kernel overlap；
- Green Context 已隔离两条路径；
- 2.39x 是共置收益（它只是优化后的串行时间相对全 eager 串行时间）；
- Reasoner eager 与 graph 逐 token 数值等价。两种模式输出语义合理但 hash 不同。

下一阶段应在不改变上述固定负载与优化组合的前提下，先把两个 `generate()` 调用抽象为可由调度器启动的
独立 runtime region，再加入两个 CUDA stream 与 Green Context，最后分别测 alone、colocated、GPU
makespan、host wall time 和双向 slowdown。
