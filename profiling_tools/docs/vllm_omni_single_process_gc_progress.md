# Cosmos3 Edge：vLLM + vLLM-Omni 单进程部署与 Green Context 共置进度

更新时间：2026-10-23

本文从工程实现角度整理目前从统一环境、单进程双引擎、无空间切分并发，到 Green Context（GC）共置的完整进度。重点不是只列入口文件，而是解释为什么必须做这些兼容与运行时改造、关键代码如何工作，以及各阶段目前测到了什么。

本文中的“Reasoner”指 vLLM 0.31.0 加载的 `nvidia/Cosmos3-Edge` 自回归路径；“Generator”指 vLLM-Omni 加载的 Cosmos3 Edge diffusion/VAE 路径。

---

## 1. 阶段划分与当前状态

```text
Gate -1   统一依赖环境：同一 Python 环境分别跑通 Reasoner 与 Generator
Gate 0.5  显存预算：确认 32 GiB 可以同时容纳两套引擎和代表性 T2V
Gate 1    同一 PID / 同一 CUDA primary context 中共同驻留并串行运行
Gate 2    不做 SM 空间切分，两条普通 CUDA stream 并发
Gate 3    创建互不重叠的 GC 分区，并让 eager/graph 真正进入各自分区
```

当前已经完成：

- vLLM 与 vLLM-Omni 的统一隔离环境；
- Edge Reasoner 与 Edge Generator 同 PID、同 GPU PID、同 CUDA primary context 常驻；
- Generator-first 和 Reasoner-first 两种初始化顺序；
- 单 rank parallel-state 共享与正确关闭；
- vLLM Reasoner eager、AOT compile + CUDA Graph；
- Omni Generator eager、regional `torch.compile`、SeaCache、Wan VAE lossless fast path；
- Gate 2 普通 stream 双线程并发及 2×2 性能消融；
- Gate 3 两个互斥 GC 的创建；
- Omni worker、输出打包、输出后处理的 GC stream 注入；
- vLLM eager model runner 的 stream 重绑；
- vLLM graph 在 Reasoner GC 上重新 capture；
- full-SM、partition-alone、colocated 三套计时；
- full Generator × Reasoner 和 VAE decode × Reasoner 的干扰测试；
- `cuda-sw` Nsys 下的 GC kernel placement 与 stream timeline 检查。

当前尚未完成：

- 把 profiling-only monkey patch 落成 vLLM/vLLM-Omni 的稳定 runtime API；
- Omni regional compile 的更多形状/输入正确性与稳定性覆盖；
- 生产式跨请求流水线，即 `Generator(i) × Reasoner(i+1)`；
- VAE fast path on/off 与 Reasoner eager/graph 的 2×2 干扰消融；
- 多 rank、TP/PP/DP 大于 1 的 parallel-state 共享；
- 对所有 graph、输入形状和动态 batch 的正确性覆盖。

---

## 2. Gate -1 / Gate 0.5：先统一环境并确认显存可行

### 2.1 为什么不能直接使用主 `.venv`

实验固定版本为：

```text
Python        3.13
PyTorch       CUDA 13.0 build
vLLM          0.31.0
vLLM-Omni     commit 03c334818305d3b1666da01cd0f67d48a4f3e7a2
Transformers  5.17.0
GPU           RTX 5090, 170 SM, 32607 MiB
```

该 vLLM-Omni commit 的包元数据要求 Transformers `<5.15`，但 Cosmos3 Edge 的原生 Transformers 配置在 5.17。我们没有污染 Cosmos 主 `.venv`，而是在：

```text
/workspace/cosmos-framework/.runtime/vllm_omni_tf517/
```

创建隔离环境，并只在隔离的 Omni 源码副本中将依赖约束调整为 `>=5.17,<5.18`。环境脚本的核心逻辑是：

```bash
omni_commit=03c334818305d3b1666da01cd0f67d48a4f3e7a2
vllm_version=0.31.0
transformers_version=5.17.0

git clone https://github.com/vllm-project/vllm-omni.git "$source_dir"
git -C "$source_dir" checkout --detach "$omni_commit"

uv venv --python 3.13 "$venv_dir"
uv pip install \
  --python "$venv_dir/bin/python" \
  --torch-backend cu130 \
  --editable "$source_dir" \
  "vllm==$vllm_version" \
  "transformers==$transformers_version"
```

创建后同时检查：

```python
config = AutoConfig.from_pretrained("nvidia/Cosmos3-Edge", local_files_only=True)
assert type(config).__name__ == "Cosmos3EdgeConfig"
assert "Cosmos3EdgeForConditionalGeneration" in ModelRegistry.get_supported_archs()
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
    Cosmos3OmniDiffusersPipeline,
)
```

这一步证明“同一依赖组合分别可用”，尚不证明两个 Engine 可以共同构造。

### 2.2 Gate 0.5 显存预算

首先分别测服务 ready 驻留和请求峰值，避免 Reasoner 按默认 `gpu_memory_utilization` 吃掉绝大多数空闲显存。Reasoner 的 KV pool 显式固定为 1 GiB。

| workload | ready median | request peak | peak above ready | request |
|---|---:|---:|---:|---:|
| Reasoner eager，1 GiB KV，128-token decode | 10,048 MiB | 10,050 MiB | 2 MiB | 2.157 s |
| Generator，256×256，2-step T2I | 9,846 MiB | 9,862 MiB | 16 MiB | 0.515 s |
| Generator，832×480，121 帧，35-step T2V | 9,847 MiB | 15,755 MiB | 5,908 MiB | 34.456 s |

Reasoner ready 与代表性 T2V 峰值相加约 25,803 MiB，仍剩约 6.16 GiB。这个结果允许进入 Gate 1，但它不是并发峰值的替代品。

相关实现和结果：

```text
profiling_tools/scripts/setup_unified_vllm_omni_env.sh
profiling_tools/runners/measure_gate_0_5_memory.py
profiling_tools/docs/gate_minus1_unified_environment.md
profiling_tools/docs/gate_0_5_memory_budget.md
profiling_results/gate_0_5_memory/
```

---

## 3. Gate 1：如何在一个进程里同时部署 vLLM 和 vLLM-Omni

### 3.1 目标进程结构

```text
一个 Python PID / 一个 CUDA primary context
├── vLLM offline LLM：Cosmos3 Edge Reasoner
├── vLLM-Omni Omni：Cosmos3 Edge Generator
├── 一套进程级 torch.distributed / vLLM parallel state
└── 两套模型权重、KV pool、diffusion/VAE runtime
```

这里不能简单地在同一个 Python 文件中依次调用两个构造函数。vLLM 与 vLLM-Omni 都会读写 vLLM 的进程级 parallel-state 单例；若两边各自初始化和销毁，第二个 Engine 会遇到重复初始化、缺失 coordinator 或 double destroy。

#### 3.1.1 能放进同一调度进程的核心前提：两边都有离线 Engine API

这个方案之所以可以成立，首先不是因为“关掉了 vLLM 多进程”，而是 **vLLM 和 vLLM-Omni 都提供了不需要启动 HTTP/API Server 的离线 Engine 入口**。因此，我们可以在自己的 Python runner 中直接构造两个长期存活的 Engine 对象，并由同一个上层调度器决定何时调用 Reasoner、何时调用 Generator：

```python
from vllm import LLM
from vllm_omni import Omni

reasoner = LLM(
    model="nvidia/Cosmos3-Edge",
    max_num_seqs=1,
    kv_cache_memory_bytes=1 << 30,
    ...,
)

generator = Omni(
    model="nvidia/Cosmos3-Edge",
    model_config={"guardrails": False},
    max_num_seqs=1,
    ...,
)

reasoner_outputs = reasoner.chat(messages, sampling_params)
generator_outputs = generator.generate(inputs, sampling_params)
```

其中：

- `vllm.LLM` 是 vLLM 的同步离线推理入口，我们用它承载 Cosmos3 Edge Reasoner；
- `vllm_omni.Omni` 是 vLLM-Omni 的同步离线入口，官方仓库也在 `examples/offline_inference/` 中以这种方式构造 Engine；我们用它承载 Cosmos3 Edge diffusion/VAE Generator；
- “离线”表示 Engine 可由 Python 调用方直接构造和调用，而不是必须通过两个 `vllm serve` 服务进程之间进行 RPC；
- 这使我们能够在 Engine 外层自定义串行、双线程并发、NVTX region、CUDA stream 和 Green Context 调度；
- 但离线 API 只解决“可内嵌调用”，不自动解决两个 Engine 对进程级 distributed/parallel state、CUDA context、stream 和资源所有权的冲突。后续 3.2–3.7 的改造正是在解决这些冲突。

因此，完整的因果链是：

```text
两个框架均提供离线 Engine API
    → 可以被同一个 Python runner 持有和调度
    → 关闭 vLLM EngineCore 多进程，使 Reasoner 实际执行留在当前 PID
    → 兼容两边的 parallel state、初始化顺序与关闭所有权
    → 两套权重才能在同一 PID / CUDA primary context 中稳定共存
    → 上层 runner 才能对两个 Engine 实施 stream/GC 共置调度
```

### 3.2 禁用 vLLM EngineCore 多进程

runner 在导入 vLLM 前设置：

```python
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_DEEP_GEMM"] = "0"
```

这一步是对上述“双离线 Engine”前提的进一步约束：离线 `LLM` 对象本身仍可以把 EngineCore 放到子进程。禁用 V1 multiprocessing 后，Reasoner EngineCore 才真正留在当前 PID，从而可以与内嵌的 Omni Engine 共享 CUDA primary context、Python runtime，并被我们的上层 runner 绑定到指定 stream 和 GC。每个阶段都记录：

```python
{
    "pid": os.getpid(),
    "children": ps_children,
    "gpu_processes": nvidia_smi_compute_processes,
    "cuda_context": current_cuda_driver_context,
}
```

最终检查包括：同一 host PID、GPU 上只有自己的 PID、主线程 CUDA context 始终相同。

### 3.3 固定 Reasoner KV 预算，而不是占满空闲显存

Reasoner 的构造代码为：

```python
return LLM(
    model="nvidia/Cosmos3-Edge",
    tensor_parallel_size=1,
    max_num_seqs=1,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    kv_cache_memory_bytes=1 << 30,
    enable_prefix_caching=False,
    disable_log_stats=True,
    **execution_kwargs,
)
```

`kv_cache_memory_bytes=1<<30` 才是实际 KV pool 上限。`gpu_memory_utilization=0.70` 只是 vLLM 0.31 在显式 KV 参数生效前仍执行的启动准入阈值。Generator-first 时若保留默认 0.92，Reasoner 会因为“启动时空闲显存不足 92%”提前失败，即使实际只需要 1 GiB KV。

### 3.4 Generator-first：Reasoner 借用 Omni 的 vLLM groups

当前主要使用 Generator-first：

1. Omni 先初始化默认 process group 和 vLLM TP/DP/PP；
2. Reasoner 使用现有 TP/DP/PP；
3. vLLM 0.31 还需要 PCP/DCP/ETP，单 rank 下临时别名到 TP；
4. 不创建第二套冲突的 process group。

关键代码：

```python
def install_reasoner_context_groups_from_omni() -> None:
    from vllm.distributed import parallel_state as vllm_ps

    if vllm_ps._TP is None or vllm_ps._PP is None:
        raise RuntimeError("Omni must initialize vLLM TP/PP before the Reasoner")

    _install_single_rank_graph_capture_compat()
    for name in ("_PCP", "_DCP", "_ETP"):
        current = getattr(vllm_ps, name)
        if current is not None and current is not vllm_ps._TP:
            raise RuntimeError(f"cannot replace existing vLLM {name} group")
        setattr(vllm_ps, name, vllm_ps._TP)
```

这个别名只在 `world_size=1` 下成立。它不是多卡 PCP/DCP/ETP 的实现。

### 3.5 Reasoner-first：Omni 借用 vLLM groups

反向顺序下，Reasoner 先创建 vLLM TP/DP/PP。我们替换 Omni diffusion worker 绑定的初始化函数：保留 vLLM groups，只创建 Omni 自己的单 rank DP/CFG/PP/SP。

```python
def initialize_shared(...):
    sizes = {
        "data_parallel_size": data_parallel_size or 1,
        "cfg_parallel_size": cfg_parallel_size,
        "sequence_parallel_size": sequence_parallel_size or 1,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        ...
    }
    if torch.distributed.get_world_size() != 1 or any(v != 1 for v in sizes.values()):
        raise RuntimeError("shared parallel compatibility is restricted to world_size=1")
    if enable_expert_parallel or use_hsdp:
        raise RuntimeError("shared parallel compatibility does not support EP or HSDP")

    singleton = [[0]]
    omni_ps._DP = omni_ps.init_model_parallel_group(singleton, ...)
    omni_ps._CFG = omni_ps.init_model_parallel_group(singleton, ...)
    omni_ps._PP = omni_ps.init_model_parallel_group(singleton, ...)
    omni_ps._SP = omni_ps.init_model_parallel_group(singleton, ...)
```

关闭时只销毁 Omni 自己创建的 `_SP/_CFG/_PP/_DP`，不碰借来的 vLLM TP/DP/PP 和默认 process group。

### 3.6 补齐 CUDA Graph coordinator 契约

Omni 的单 rank coordinator 比 vLLM 0.31 的接口旧，缺少 `graph_capture(context)`。Generator-first 创建 graph Reasoner 时，vLLM capture 会直接失败。

我们只对单 rank coordinator 补充 capture-stream context：

```python
@contextmanager
def graph_capture(group, graph_capture_context=None):
    if group.world_size != 1:
        raise RuntimeError("restricted to world_size=1")
    if graph_capture_context is None:
        graph_capture_context = vllm_ps.GraphCaptureContext(torch.cuda.Stream())

    stream = graph_capture_context.stream
    current = torch.cuda.current_stream()
    if current != stream:
        stream.wait_stream(current)
    with torch.cuda.stream(stream):
        yield graph_capture_context
```

rank 1 没有 NCCL/custom-all-reduce capture，因此这里只补 stream 契约，不伪造通信 capture。

### 3.7 所有权与关闭顺序

Generator-first 时，process-wide distributed environment 由 Omni 创建并拥有。Reasoner 的标准 shutdown 会调用全局 `cleanup_dist_env_and_memory()`，提前销毁 Omni 仍在使用的 groups。

当前做法是在“重建 Reasoner”时临时只关闭 Reasoner worker/model/KV/graph，不销毁共享 groups：

```python
original_cleanup = engine_core_module.cleanup_dist_env_and_memory

def cleanup_memory_without_distributed_teardown(*args, **kwargs):
    gc.collect()
    torch.cuda.empty_cache()

engine_core_module.cleanup_dist_env_and_memory = cleanup_memory_without_distributed_teardown
try:
    reasoner.llm_engine.engine_core.shutdown()
finally:
    engine_core_module.cleanup_dist_env_and_memory = original_cleanup
```

整个进程最终关闭顺序是：

```text
清除 Reasoner 借用的 PCP/DCP/ETP aliases
→ generator.close()，由所有者关闭 distributed environment
→ Reasoner EngineCore shutdown
→ Python GC / CUDA allocator 清理
```

### 3.8 Gate 1 串行性能

固定工作量：官方 prompt upsampler、`max_model_len=6144`、自然停止不超过 2600 tokens；Generator 为 832×480、121 帧、35 steps、SeaCache on。每边 1 次 warmup、3 次 measured，中位数如下：

| Reasoner | Generator | Reasoner | Generator | 串行总时间 |
|---|---|---:|---:|---:|
| eager | eager | 39.413 s | 18.253 s | 57.666 s |
| eager | compile | 38.469 s | 16.494 s | 54.963 s |
| graph | eager | 7.550 s | 18.354 s | 25.904 s |
| graph | compile | 7.513 s | 16.572 s | 24.084 s |

注意：这里的“全优化相对全 eager 2.39×”是串行部署优化收益，不是共置吞吐收益。

结果：

```text
profiling_results/gate_1_single_process/*_tok2600.json
```

---

## 4. Gate 2：不带 GC 的单进程并发，以及新增优化

Gate 2 不限制 SM。两套引擎仍在同一个 PID/context 中，但由两个 host thread 向两条普通 CUDA stream 提交。该阶段的目的，是建立“自然并发/time multiplexing”基线。

### 4.1 两线程和同步起点

```python
start_barrier = threading.Barrier(3)

reasoner_future = executor.submit(reasoner_worker)
generator_future = executor.submit(generator_worker)
start_barrier.wait(timeout=60.0)   # 主线程同时释放两个 worker

wall_start = time.perf_counter()
reasoner_result = reasoner_future.result()
generator_result = generator_future.result()
wall_end = time.perf_counter()
```

worker 内部调用：

```python
with torch.cuda.stream(reasoner_stream), torch.cuda.nvtx.range("gate2.reasoner..."):
    run_reasoner(...)

with torch.cuda.stream(generator_stream), torch.cuda.nvtx.range("gate2.generator..."):
    run_generator(...)
```

host interval 重合不等于 GPU kernel 重合，所以结果 JSON 中 `gpu_kernel_overlap_verified` 默认仍为 false；GPU overlap 由对应 Nsys timeline 单独验证。

### 4.2 为使性能对照有效而新增的固定负载

早期 eager/graph 输出长度不同，无法比较 slowdown。benchmark 模式加入：

```python
SamplingParams(
    temperature=0.0,
    seed=0,
    max_tokens=max_tokens,
    min_tokens=max_tokens if fixed_tokens else 0,
    ignore_eos=fixed_tokens,
)
```

并在 warmup、alone、concurrent 后强制核对 token 数：

```python
if fixed_tokens and result["completion_tokens"] != max_tokens:
    raise RuntimeError(...)

all_counts = full_sm + warmup + alone + concurrent
if len(set(all_counts)) != 1:
    raise RuntimeError("Reasoner token workload mismatch")
```

这不是生产生成策略，而是性能消融所需的工作量控制。

### 4.3 同时驻留时启用的后端优化

Reasoner：

- Paged KV，固定 1 GiB pool；
- FlashAttention v2；
- eager 或 vLLM AOT `torch.compile` + `FULL_AND_PIECEWISE` CUDA Graph；
- graph 模式保留 vLLM 自己的 stream/event replay 契约，不再用外层普通 stream 强包 replay；
- BF16，无量化，prefix cache off。

Generator：

- `generator-mode=eager|compile`；compile 表示 Omni diffusion regional `torch.compile`，不是 Cosmos3 DiT CUDA Graph；
- SeaCache：`threshold=0.25`、`residual_order=1`、`max_consecutive_cached=2`、`power_exp=3.0`；
- Wan VAE decoder/encoder lossless fast path；
- VAE 使用 cuDNN convolution 和 fused Triton padding/norm/residual/upsample kernels；
- 当前未启用 layerwise offload、VAE tiling/slicing、prompt embedding cache。

这些后端性能优化并不是“让两个 Python 对象能共存”的必要条件；真正保证共同运行的是 Gate 1 的 parallel-state、显存预算、单进程 EngineCore 和关闭顺序。它们的作用是在双模型部署形态下把各自稳态开销降下来。

### 4.4 Gate 2 最终 2×2 结果

结果为 2 次 warmup、3 次 alone、3 次 concurrent 的中位数。eager Reasoner 固定 1200 tokens，graph Reasoner 固定 2600 tokens；因此只能在相同 Reasoner 模式内比较 Generator eager/compile，不应跨行比较绝对时间。

| Reasoner | Generator | R alone | G alone | concurrent wall | R slowdown | G slowdown | full-SM serial / concurrent |
|---|---|---:|---:|---:|---:|---:|---:|
| eager 1200 | eager | 20.026 s | 18.228 s | 35.251 s | 1.760× | 1.027× | 1.085× |
| eager 1200 | compile | 22.140 s | 16.471 s | 33.861 s | 1.530× | 1.025× | 1.140× |
| graph 2600 | eager | 8.718 s | 18.230 s | 26.791 s | 3.073× | 1.193× | 1.006× |
| graph 2600 | compile | 8.732 s | 16.552 s | 24.990 s | 2.862× | 1.191× | 1.012× |

吞吐分子是同一进程里实测的 full-SM 串行 host wall：

```python
throughput_gain = (
    reasoner_full_sm_alone_host_wall
    + generator_full_sm_alone_host_wall
) / concurrent_host_wall
```

不是 colocated 时间之和，也不是直接取两边 colocated 时间的 `max()`。

Gate 2 说明：普通 CUDA 调度可以产生 host/GPU overlap，但没有空间隔离时，Reasoner 尤其是 graph 路径受到严重干扰；graph 两组几乎没有吞吐收益。这正是进入 Gate 3 的原因。

结果和 Nsys：

```text
profiling_results/gate_2_no_spatial/final_2x2/
profiling_results/gate_2_no_spatial/nsys_2x2_warm/
```

---

## 5. Gate 3：引入 GC 后，eager 和 graph 分别做了什么

### 5.1 两个 GC 必须来自同一次资源 split

不能创建两个彼此独立的单 GC，因为那不能保证 SM 集合互斥。当前实现从设备完整 SM resource 做一次 split，first 给 Generator，remainder 给 Reasoner：

```python
self._cuda.cuDevSmResourceSplitByCount(
    byref(first_resource),
    byref(group_count),
    byref(available),
    byref(second_resource),
    flags,
    requested_first_sm_count,
)

for resource in (first_resource, second_resource):
    self._cuda.cuDevResourceGenerateDesc(...)
    self._cuda.cuGreenCtxCreate(...)
    self._cuda.cuGreenCtxStreamCreate(...)
```

然后把 Driver stream 包装成 PyTorch ExternalStream：

```python
gc_owner = GreenContextPair(generator_sm_count, device=0)
generator_stream = torch.cuda.ExternalStream(gc_owner.first_stream_ptr, device=0)
reasoner_stream = torch.cuda.ExternalStream(gc_owner.second_stream_ptr, device=0)
```

RTX 5090 上请求数量会按硬件 granularity 调整，例如请求 Reasoner 18 SM 实际得到 24；文档和 JSON 始终记录 requested/actual。

### 5.2 初版为什么失败：外层 stream context 不会穿透 serving runtime

最初只做：

```python
with torch.cuda.stream(generator_gc_stream):
    generator.generate(...)
```

Nsys 显示 Reasoner eager 能进入 Reasoner GC，但 Omni Generator 仍在 primary-context ordinary stream；Reasoner graph 的主体 graph nodes 也在普通 context。原因是：

- CUDA current stream 是线程局部状态；
- `Omni.generate()` 进入 scheduler/executor/worker 后，真正 launch kernel 的不是外层调用位置；
- CUDA Graph 的资源归属由 capture 时使用的 stream/context 决定，replay 外面再包 stream 已经太晚。

因此 Gate 3 的工作重点不是模型层 forward，而是 serving runtime 的 worker/model-runner/capture 边界。

### 5.3 Generator eager/compile：在真正 worker 内注入 GC stream

我们 patch `DiffusionWorker.execute_model()`，让完整 pipeline 在 worker 的执行线程内进入 Generator GC：

```python
def execute_model_on_external_stream(self, *args, **kwargs):
    patch.calls += 1
    active_stream = patch.stream
    with torch.cuda.stream(active_stream), torch.cuda.nvtx.range(
        "gate3.omni_worker.generator_gc"
    ):
        return original(self, *args, **kwargs)
```

只包 `execute_model()` 仍不完整。输出共享内存打包/D2H 和 `DiffusionEngine.postprocess_output()` 会在 pipeline 返回后执行，早期 Nsys 发现这些 per-frame mul/clamp/add kernel 逃回普通 stream。因此又补了两个边界：

```python
def pack_output_on_external_stream(output, d2h_stream=None, **kwargs):
    if d2h_stream is not None and d2h_stream.cuda_stream != active_stream.cuda_stream:
        d2h_stream.synchronize()
    with torch.cuda.stream(active_stream):
        return original_pack(output, d2h_stream=active_stream, **kwargs)


def postprocess_output_on_external_stream(self, *args, **kwargs):
    with torch.cuda.stream(active_stream):
        return original_postprocess(self, *args, **kwargs)
```

这三处分别覆盖：模型执行、output pack/D2H、视频输出后处理。

对 Cosmos3 Omni 而言，`generator-mode=compile` 是对 `transformer.language_model` 和 `transformer` 的 repeated blocks 进行 regional `torch.compile(dynamic=True)`，不是持久 diffusion CUDA Graph。因此生成的 Inductor kernel 仍在每次调用时的 current stream 上提交，可以复用上述 worker-level stream patch：

```text
Omni offline Engine
  → DiffusionWorker.execute_model()
  → torch.cuda.stream(generator_gc_stream)
  → regional-compiled block callable
  → Inductor/Triton/cuBLAS/cuDNN kernels on Generator GC stream
```

runner 之前将所有 `generator-mode=compile` 错误地当成“需要另一套 CUDA Graph capture”并在 Gate 3 直接拒绝。现在已移除这个 eager-only 限制，并让 compile 与 eager 共用 full-SM、partition-alone 和 colocated 的 stream 切换流程。

需要保留一个严格边界：“compile 请求跑通”不等于“编译 kernel 已证明留在 GC”。因此 JSON 会同时记录：

```python
"generator_compile_gc_binding_requested": True
"generator_compile_kernel_placement_verified": False
```

第二个字段在自动 Nsys/SQLite 验证接入前保守地保持 `False`。必须在 `cuda-sw` Nsys 中确认编译后的 DiT kernel 属于 Generator GC/stream，才能把该路径标记为完成。

该 placement 已用 136→120→136 SM 迁移实验完成验证：

| 区间 | actual SM | GC / stream | kernel calls | kernel sum |
|---|---:|---:|---:|---:|
| 136 SM before | 136 | GC 2 / stream 15 | 37,513 | 19.539 s |
| 120 SM after migration | 120 | GC 3 / stream 18 | 37,513 | 21.486 s |
| 136 SM after return | 136 | GC 2 / stream 15 | 37,513 | 19.641 s |

三个 measured NVTX 区间内没有出现第二条 ordinary/primary stream 的 kernel 逃逸；返回 136 SM 后 host time 与迁移前只相差 0.46%。这证明当前 Cosmos3 Omni regional compile 产物使用调用时 current stream，可在同一 Engine 实例中迁移 GC，不需要像 Reasoner CUDA Graph 那样销毁并重新 capture。

证据：

```text
profiling_results/gate_3_green_context/generator_compile_gc_stream_migration.json
profiling_results/gate_3_green_context/generator_compile_gc_stream_migration_placement.json
profiling_results/gate_3_green_context/nsys/generator_compile_gc_stream_migration.nsys-rep
```

### 5.4 Reasoner eager：重绑 model runner 的长期 stream

仅在 `generate()` 外层设置 stream 不能覆盖 vLLM runner 自己持有的 main/copy stream。当前直接找到 in-process V1 `GPUModelRunner` 并同步重绑：

```python
def bind_vllm_runner_stream(reasoner, stream):
    stream.synchronize()
    runner = get_vllm_gpu_model_runner(reasoner)
    runner.__dict__["main_stream"] = stream
    runner.output_copy_stream = stream
```

eager 可以在 full-SM GC baseline 与 partition GC 之间切换，因为没有持久 CUDA Graph 绑定到旧 stream。切换前同步旧 stream，避免尚未完成的工作跨越资源所有权变化。

### 5.5 Reasoner graph：不能迁移，只能销毁并在目标 GC 重新 capture

CUDA Graph 捕获后不能从 full-SM ordinary/GC stream“搬到”另一个 GC。当前 graph 流程是：

```text
1. 创建 full-SM Reasoner graph，测 full-SM warm/alone
2. 清除 vLLM graph registry
3. 关闭 Reasoner，但保留 Omni 拥有的共享 parallel state
4. 回收 Reasoner model/KV/graph 内存
5. 安装 GC capture-stream patch
6. 在 Reasoner GC current stream 下重新创建 Reasoner
7. 检查 capture patch 的调用计数非零
8. 再测 partition-alone 与 colocated
```

runner 中的核心重建逻辑：

```python
clear_vllm_cudagraphs()
shutdown_reasoner_preserving_shared_parallel_state(reasoner)
reasoner = None
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

reasoner_graph_stream_patch = install_vllm_graph_capture_stream_patch(reasoner_stream)
torch.cuda.set_stream(reasoner_stream)
try:
    reasoner = create_reasoner(
        borrow_omni_parallel_state=True,
        max_model_len=6144,
        execution_mode="graph",
        gpu_memory_utilization=0.55,
    )
finally:
    torch.cuda.set_stream(primary_stream)
```

这里把准入阈值降到 0.55，是因为 full-SM graph teardown 后 CUDA/runtime pool 仍可能驻留；实际 KV pool 仍固定为 1 GiB，没有缩小工作量。

### 5.6 graph capture 需要同时 patch 三层入口

vLLM 的 `graph_capture` 被若干模块按值 import，仅 patch 定义模块不会影响已绑定引用；同时，内部 graph manager 和 compiled graph wrapper 还会自行取得 capture stream。因此当前同时替换：

```python
for owner in (parallel_state, gpu_model_runner, cudagraph_utils):
    owner.graph_capture = graph_capture_on_external_stream

ModelCudaGraphManager._capture_stream = capture_stream_on_external_green_context
compilation_cuda_graph.current_stream = compiled_capture_stream_on_external_green_context
```

统一的 capture context 使用 Reasoner GC stream：

```python
@contextmanager
def graph_capture_on_external_stream(device):
    context = parallel_state.GraphCaptureContext(reasoner_gc_stream)
    with (
        parallel_state.get_tp_group().graph_capture(context),
        parallel_state.get_pp_group().graph_capture(context),
        parallel_state.get_dp_group().graph_capture(context),
    ):
        yield context
```

runner 还要求下列计数至少有一类非零，否则直接判为 graph 没有进入 patch：

```python
patch.calls
patch.manager_capture_stream_calls
patch.compiled_capture_stream_calls
```

### 5.7 两线程下的 vLLM IR priority 全局状态竞争

vLLM 0.31 的 `IrOp.set_priority()` 会暂时修改进程级 `_priority_impls`。两个模型线程嵌套进入/退出该 context 时可能覆盖并错误恢复对方的 priority，早期 eager 共置出现约 40% Reasoner slowdown，其中一部分由此造成。

profiling-only 修复把临时 override 移到 `ContextVar`：

```python
def _priority_context(op) -> ContextVar:
    context = getattr(op, "_cosmos_priority_context", None)
    if context is None:
        context = ContextVar(f"vllm_ir_priority_{op.name}_{id(op)}", default=None)
        setattr(op, "_cosmos_priority_context", context)
    return context

@contextmanager
def _set_context_priority(self, priority):
    filtered = self._filter_priority_impls(priority)
    token = _priority_context(self).set(filtered)
    try:
        yield
    finally:
        _priority_context(self).reset(token)
```

修复后单轮 136:34、1300-token eager 诊断中，Reasoner colocated slowdown 从约 `1.400×` 降到 `1.136×`。这是单轮归因结果，不应当作三轮正式中位数。

### 5.8 full-SM、partition-alone 与 colocated 必须全部实测

Gate 3 的 summary 同时记录：

```python
reasoner_sm_restriction_slowdown = partition_alone_R / full_sm_alone_R
generator_sm_restriction_slowdown = partition_alone_G / full_sm_alone_G
reasoner_slowdown = colocated_R / partition_alone_R
generator_slowdown = colocated_G / partition_alone_G
throughput_gain_host_wall = full_sm_serial_wall / colocated_wall
```

这将“SM 变少导致的 slowdown”和“共置竞争导致的额外 slowdown”分开。吞吐收益的分子必须是两条 full-SM 请求在同一进程中的实测串行 wall，而不是 partition-alone 之和。

---

## 6. Gate 3 当前性能与干扰结果

### 6.1 full Generator × Reasoner

#### Eager，Generator 136 SM / Reasoner 34 SM，1300 tokens

IR priority 隔离后的单轮诊断：

| 指标 | 数值 |
|---|---:|
| R full-SM alone | 23.802 s |
| G full-SM alone | 18.194 s |
| R partition-alone | 23.205 s |
| G partition-alone | 22.198 s |
| R colocated | 26.370 s |
| G colocated | 22.795 s |
| R colocation slowdown | 1.136× |
| G colocation slowdown | 1.027× |
| full-SM serial / colocated wall | 1.593× |

文件：

```text
profiling_results/gate_3_green_context/eager_eager_g136_tok1300_ir_context_quick.json
```

#### Graph，Generator 136 SM / Reasoner 34 SM，2600 tokens

三轮中位数：

| 指标 | 数值 |
|---|---:|
| R full-SM alone | 8.726 s |
| G full-SM alone | 18.250 s |
| R partition-alone | 9.163 s |
| G partition-alone | 22.232 s |
| R colocated | 15.500 s |
| G colocated | 23.545 s |
| R colocation slowdown | 1.692× |
| G colocation slowdown | 1.059× |
| full-SM serial / colocated wall | 1.145× |

文件：

```text
profiling_results/gate_3_green_context/graph_eager_g136_tok2600_ir_context_repeat3_v4.json
```

#### Graph，Generator 120 SM / Reasoner 50 SM，两次 2400-token Reasoner

为了接近一次 T2V 的长度，单次 Generator 对应两次 Reasoner：

| 指标 | 数值 |
|---|---:|
| R full-SM alone（两次） | 16.068 s |
| G full-SM alone | 18.285 s |
| R partition-alone | 19.658 s |
| G partition-alone | 24.144 s |
| R colocated | 27.156 s |
| G colocated | 25.613 s |
| R colocation slowdown | 1.381× |
| G colocation slowdown | 1.061× |
| full-SM serial / colocated wall | 1.265× |

文件：

```text
profiling_results/gate_3_green_context/graph_eager_r50_g120_tok2400x2_balanced_repeat3.json
```

这说明 graph 共置已经能运行且 Generator 影响较小，但 Reasoner 仍有明显额外膨胀；仅靠增加 Reasoner SM 并不能消除全部干扰。

### 6.2 VAE decode × Reasoner matched 实验

为了把 full T2V 中的干扰定位到组件，当前最新实验只让 Reasoner 与 VAE decode 同时开始。

严格匹配负载：

```text
同一个 native prompt JSON
固定生成 1300 tokens
同一个 832×480、121-frame、35-step T2V
Generator 120 SM / Reasoner 50 SM
Generator cache on
Reasoner graph
```

三轮普通测量的中位趋势：

| 实现 | Reasoner alone | colocated | R slowdown | VAE alone | colocated | VAE slowdown |
|---|---:|---:|---:|---:|---:|---:|
| native Cosmos graph | 约 8.08 s | 约 10.92 s | 1.353× | 约 7.63 s | 约 8.62 s | 1.129× |
| vLLM graph + Omni VAE | 约 4.90 s | 约 8.77 s | 1.788× | 约 6.29 s | 约 6.38 s | 1.014× |

Nsys 单轮与三轮趋势一致：

| 实现 | R alone→colocated | R slowdown | VAE slowdown |
|---|---:|---:|---:|
| native | 8.327→11.076 s | 1.330× | 1.129× |
| vLLM + Omni | 5.126→8.975 s | 1.751× | 1.013× |

更关键的是 Reasoner GC stream 的 kernel/gap 分解：

| 实现 | kernel sum alone→colocated | inter-kernel gaps alone→colocated | 最大 gap |
|---|---:|---:|---:|
| native | 7.678→10.272 s | 0.575→0.728 s | 3.64 ms |
| vLLM | 4.752→5.018 s | 0.364→3.949 s | 186.70 ms |

因此当前证据支持：

- native slowdown 主要表现为 kernel 自身执行时间增加，像常规共享 cache/带宽/执行资源竞争；
- vLLM 的 kernel sum 只增加约 5.6%，但 kernel 之间多出约 3.6 s 空洞；
- vLLM 的 `1.75~1.79×` slowdown 不能主要归因于 kernel body 的显存带宽膨胀；
- 更像 CUDA Graph node 调度、依赖推进或 VAE 连续提交造成 Reasoner graph 间歇性等待；
- Omni VAE 几乎不受影响，而 Reasoner 承担大部分等待，干扰高度不对称。

vLLM Reasoner kernel body 中增量最大的是 BF16 GEMV：约增加 216 ms；FlashAttention split-K 与 combine 合计只增加十余 ms，远不足以解释近 3.85 s wall-time 增量。

结果：

```text
profiling_results/vae_reasoner_baseline/repeats/
profiling_results/vae_reasoner_baseline/matched_prompt/
profiling_results/vae_reasoner_baseline/nsys/native_graph_r50_g120_tok1300_matched.nsys-rep
profiling_results/vae_reasoner_baseline/nsys/vllm_graph_r50_g120_tok1300_matched.nsys-rep
```

### 6.3 Nsys 版本和 GC 观察口径

容器内使用 Nsys 2025.6。对 GC + CUDA Graph 必须优先使用：

```bash
--trace=cuda-sw,nvtx,osrt --cuda-graph-trace=node
```

此前硬件/device graph trace 在该版本下可能把 graph nodes 显示到 primary context；`cuda-sw` 软件追踪才能正确观察 graph kernel 的 Green Context/stream 归属。性能 JSON 中的 `green_context_kernel_placement_verified` 仍保守保持 false，因为这个字段还没有自动读取 Nsys；placement 结论来自离线 Nsys/SQLite 查询。

---

## 7. vLLM 与 vLLM-Omni 各自启用了什么，为什么可能影响共置

### 7.1 vLLM Reasoner

已由日志确认：

- vLLM V1 Engine、V2 Model Runner；
- BF16，无量化；
- Paged KV，LBNHC layout；
- 1 GiB KV pool、9360-token capacity；
- FlashAttention v2；
- AOT `torch.compile` artifact；
- FULL + PIECEWISE CUDA Graph；
- fused RMSNorm/Triton kernels；
- `reshape_and_cache_flash_kernel` 等 Paged-KV 更新 kernel。

这些优化将 2600-token cap 下的 Reasoner 从约 38~39 s eager 降至约 7.5 s graph，但 graph 是更紧凑、依赖更固定的 kernel 链。最新 VAE matched Nsys 表明，共置时主要问题不是这些 kernel 全部变慢，而是 graph stream 上 node 之间出现大量等待。

### 7.2 vLLM-Omni Generator / VAE

已由源码和日志确认：

- Cosmos3 diffusion eager 或 regional `torch.compile`；
- SeaCache 复用 diffusion 中间结果；
- `DistributedAutoencoderKLWan`；
- Wan VAE decoder/encoder `lossless` fast path；
- `channels_last=False`，因此使用 bit-exact replacement 档位；
- cuDNN causal convolution；对不能保持 bitwise 等价的形状使用 pre-padded path；
- Triton fused kernels，例如：
  - `_rms_norm_scale_kernel`
  - `_add_bias_residual_kernel`
  - `_upsample2x_rows_kernel`
  - `_dup_up3d_add_rows_kernel`
  - `_cat_pad_pixels_kernel`
  - `_cat_pad_rows_kernel`

在 matched Nsys 中，Omni VAE 的主 stream kernel sum 从 5.647 s 变成 5.686 s，gap 只从 0.020 s 变成 0.027 s。它保持了非常连续的执行，而 vLLM graph stream 出现大量停顿。这使“Omni VAE fast path 的连续 kernel 链压制 graph node 推进”成为下一步需要直接消融的假设，但目前还不能写成已经证明的根因。

SeaCache 只影响 diffusion sampling，不直接解释 VAE-only overlap；Generator eager 也意味着本轮 VAE matched 实验没有启用 Omni diffusion compile。

---

## 8. 当前代码地图与工作量

| 文件 | 作用 |
|---|---|
| `profiling_tools/scripts/setup_unified_vllm_omni_env.sh` | 固定统一依赖环境 |
| `profiling_tools/core/omni_shared_parallel.py` | 单 rank parallel-state 借用、graph coordinator 兼容、关闭所有权 |
| `profiling_tools/runners/single_process_dual_engine.py` | Gate 1 双引擎构造、串行 warm/measured、工作负载 |
| `profiling_tools/core/green_context.py` | CUDA Driver GC resource split、context/stream 生命周期 |
| `profiling_tools/core/runtime_stream_patch.py` | Omni worker/output stream 注入、vLLM eager stream 重绑、graph capture stream patch |
| `profiling_tools/core/vllm_thread_context_patch.py` | 双线程 IrOp priority 的 ContextVar 隔离 |
| `profiling_tools/runners/concurrent_reasoner_generator.py` | Gate 2/3 full-SM、partition-alone、concurrent、VAE-only 测量 |
| `profiling_tools/runners/validate_reasoner_gc_stream_migration.py` | Reasoner eager stream/GC 迁移验证 |
| `profiling_tools/runners/validate_generator_gc_stream_migration.py` | Omni worker/output/postprocess GC 迁移验证 |

目前改造集中在 serving runtime 边界，没有重写 Cosmos3 Transformer block、attention 数学或 diffusion model。本阶段代码量的复杂性主要来自：

1. 两个框架共享进程级全局状态；
2. stream 是线程局部状态，外层 context 不会自动传播；
3. graph 的 GC 归属在 capture 时确定，不能在 replay 时迁移；
4. worker 返回后仍有输出 kernel，必须逐段排除“逃逸”；
5. full-SM baseline 与 partition graph 需要在一个进程生命周期里安全销毁和重建 Reasoner；
6. 计时必须分离 SM restriction、colocation interference 和 full-SM throughput reference。

### 8.1 技术代码工作量统计

这里的统计刻意不使用整个 `profiling_tools/` 的 diff，因为其中大量内容是运行入口、
warmup/repeat 控制、计时、Nsys 采集、结果分析和绘图，不能代表实际部署改造量。

统计口径如下：

- Cosmos 仓库以当前 `HEAD` 为基线；
- vLLM-Omni 以固定 commit `03c334818305d3b1666da01cd0f67d48a4f3e7a2` 为基线；
- 数字是 diff 的物理行数，包含必要注释和 docstring；
- 新增文件按完整物理行数计；
- 混合型 runner 只统计承担 Engine 构造职责的函数，不统计 CLI、工作负载运行、
  warmup、计时和结果落盘；
- 不统计文档、shell 命令、验证 runner、分析/绘图脚本、测试输入和 profiling 结果；
- `green_context.py` 是本轮 vLLM/Omni 集成前已经存在的 GC 基础设施，因而不计入本轮增量。

按 Gate 划分，当前增量为：

| 部分 | 技术实现 | 新增 | 删除 | 说明 |
|---|---|---:|---:|---|
| Gate -1：统一环境 | Omni `requirements/common.txt` | 1 | 1 | 仅放宽隔离源码副本的 Transformers 版本约束；不改变 Omni runtime |
| Gate 1：单进程双 Engine | `omni_shared_parallel.py` | 198 | 0 | parallel-state 借用、rank-1 graph coordinator 兼容、资源所有权和安全关闭 |
| Gate 1：Engine 构造层 | `create_reasoner()` + `create_generator()` | 75 | 0 | 只计双后端构造和关键配置；所在 runner 的其余 395 行全部排除 |
| Gate 2：普通 stream 并发 | 无独立 runtime patch | 0 | 0 | 两线程、barrier、固定 token 和计时均属于实验 harness，不计入技术改造；SeaCache/compile 配置已包含在上面的 Generator 构造函数中 |
| Gate 3：stream/graph 注入 | `runtime_stream_patch.py` | 287 | 0 | Omni worker/output stream 注入、vLLM eager stream 重绑、vLLM graph capture stream 注入与恢复 |
| Gate 3：线程局部 vLLM 状态 | `vllm_thread_context_patch.py` | 145 | 0 | 将并发冲突的 IR priority scoped override 改成 `ContextVar` 语义 |
| Gate 3：Cosmos runtime hook | `regions.py` | 22 | 1 | 传递线程局部 expected CUDA stream identity |
| Gate 3：Cosmos graph 校验 | `unified_mot.py` | 16 | 0 | capture 前检查实际 stream 与 runtime region 指定 stream 一致 |
| **合计** |  | **744** | **2** | 不含运行、测试、profiling 和文档代码 |

从“是否直接修改第三方框架”角度看，工作量还可以重新归纳为：

| 修改面 | 直接修改上游源码 | 仓库内新增适配/调度代码 | 结论 |
|---|---:|---:|---|
| vLLM 0.31.0 | 0 行 | 与 Omni 共用的适配层中实现 | 没有修改安装在环境中的 vLLM package；eager stream、graph capture 和 IR priority 都是可恢复 monkey patch |
| vLLM-Omni | 1 增 / 1 删 | 与 vLLM 共用的适配层中实现 | 上游源码只改依赖约束；parallel-state、worker/output stream 和 shutdown 兼容都在仓库 adapter 中 |
| Cosmos3 framework | 38 增 / 1 删 | — | 只增加 runtime stream identity 与 graph capture 校验，不修改 Transformer、attention 或 diffusion 数学 |
| 新增统一适配层 | — | 630 行 | `omni_shared_parallel.py` 198 行、`runtime_stream_patch.py` 287 行、`vllm_thread_context_patch.py` 145 行 |
| 双 Engine 构造 | — | 75 行 | 只计 Reasoner/Generator factory；后续应从 runner 抽成正式 runtime module |

因此，“修改了多少 vLLM/Omni 代码”需要区分两种含义：直接 patch 第三方源码几乎为零，
但为了让两个后端在同一进程、同一 CUDA context 和两个 GC 中正确协作，新增了 630 行
可撤销兼容层以及 75 行双 Engine 构造代码。当前实现仍是 profiling 阶段的 adapter；若进入
长期维护，应把这些 monkey patch 变成带版本检查的正式 runtime API，并为每个被替换符号
增加 fail-closed 校验和单元测试。

---

## 9. 下一步实验

首先做 VAE fast path 与 Reasoner execution mode 的 matched 2×2：

```text
Reasoner graph + VAE lossless fast path
Reasoner graph + VAE fast path off
Reasoner eager + VAE lossless fast path
Reasoner eager + VAE fast path off
```

四组保持：同一 prompt、固定 1300 tokens、同一 T2V、120:50 GC、相同 warmup/repeats。每组记录：

- Reasoner/VAE partition-alone 与 colocated 时间；
- Reasoner kernel sum、inter-kernel gap sum、最大 gap；
- VAE kernel sum 与 gap；
- graph launch/node timeline；
- 两个 GC 的 kernel placement。

如果关闭 VAE fast path 明显缩小 vLLM graph gaps，则继续定位是哪类 fused/continuous VAE kernel 造成调度不对称；如果 graph→eager 才消除 gaps，则重点检查 vLLM graph node 推进、event/依赖和 driver 调度，而不是优先做显存带宽 NCU。
