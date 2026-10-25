# Cosmos3 单进程共置：阅读讨论笔记

本文档记录技术报告阅读过程中提出的问题、讨论结论及其证据边界。它是讨论索引，不替代已有实验报告；尚未由代码、日志或 profiler 证实的判断会明确标为假设。

## 记录格式

每个问题按以下结构记录：

1. **问题**：阅读时产生的疑问。
2. **简短回答**：当前最直接的结论。
3. **机制解释**：为什么会出现该现象。
4. **证据**：对应代码、JSON、日志或 Nsys/NCU 结果。
5. **边界与未决项**：目前不能推出什么，以及是否还需要验证。

## 讨论记录

后续问题从这里开始按编号追加。

## 1. Gate 1：单进程证据、显存、并行组和实现工作量

### 1.1 两个 Engine 是否真的运行在同一个 PID

**问题。** Gate 1 文档声称 vLLM Reasoner 与 vLLM-Omni Generator 位于同一 PID，但没有列出直接证据。

**简短回答。** 验证过，而且两个初始化顺序均有 JSON 证据。最直接的判断不是 JSON 中固定写入的 \`same_host_pid: true\`，而是每个阶段的 \`nvidia-smi --query-compute-apps\` 只有 runner 自己的 PID，同时主线程 current CUDA context 在双 Engine 构造和两次推理后保持不变。

Reasoner-first 的成功记录：

\`\`\`text
文件：profiling_results/gate_1_single_process/reasoner_first_v2.json
runner PID：874379
after_reasoner GPU process：PID 874379，10040 MiB
after_both GPU process：PID 874379，15784 MiB
after_reasoner_inference：PID 874379，15786 MiB
after_generator_inference：PID 874379，15806 MiB
CUDA context：以上有效快照均为 0x1608f880
\`\`\`

Generator-first 的修正后记录：

\`\`\`text
文件：profiling_results/gate_1_single_process/generator_first_v4.json
runner PID：877806
after_generator GPU process：PID 877806，9840 MiB
after_both GPU process：PID 877806，17380 MiB
after_reasoner_inference：PID 877806，17382 MiB
after_generator_inference：PID 877806，17410 MiB
CUDA context：Reasoner 构造后的有效快照均为 0x265027a0
\`\`\`

后续 2600-token 的四组 Gate 1 结果也逐组满足：

\`\`\`json
{
  "same_host_pid": true,
  "only_own_gpu_pid_after_both": true,
  "same_main_thread_cuda_context": true,
  "reasoner_output_nonempty": true,
  "generator_output_nonempty": true
}
\`\`\`

例如全优化记录 \`reasoner_graph_generator_compile_tok2600.json\` 中，runner/GPU PID 均为 \`886693\`，双 Engine 构造后的 context 为 \`0x3e4df600\`。

Generator-first 快照里还会出现一个 Python 子进程，例如 PID \`878260\`。它的命令是 Python \`multiprocessing.resource_tracker\`，没有出现在 GPU compute-process 列表中，不是承载模型执行的 GPU worker。因此准确表述应是“只有主 PID 持有 GPU compute context”，而不是“进程树中绝对没有任何辅助子进程”。

对应采集代码：

\`\`\`python
def process_snapshot() -> dict:
    pid = os.getpid()
    # 枚举整个后代进程树
    ps = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,pgid=,stat=,comm=,args="], text=True
    ).splitlines()
    ...
    # 独立读取真正拥有 GPU compute context 的 PID
    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    return {
        "pid": pid,
        "children": rows,
        "gpu_processes": gpu_processes,
        "cuda_context": cuda_context(),
    }
\`\`\`

**证据边界。** \`cuCtxGetCurrent\` 记录的是采样时主线程的 current context；单凭它不能证明每一个库内部线程从未短暂创建其他 context。这里的结论还结合了单一 GPU PID，以及后续 Nsys 中同进程执行的事实。Gate 1 所需的“两个模型没有拆到两个 GPU worker PID”已经被充分验证。

### 1.2 Gate 0.5/Gate 1 的显存参数、阈值、时间和实测占用

#### 1.2.1 三个容易混淆的量

Reasoner 构造中的三个参数不是同一种预算：

\`\`\`python
LLM(
    max_num_seqs=1,
    max_model_len=6144,
    gpu_memory_utilization=0.70,
    kv_cache_memory_bytes=1 << 30,
    enable_prefix_caching=False,
)
\`\`\`

1. \`kv_cache_memory_bytes=1 GiB\` 是实际 KV pool 的显式预算。
2. \`gpu_memory_utilization=0.70\` 在本版本中仍参与启动前的显存准入检查；它不是“把 70% 显存分给 KV”。
3. \`max_model_len=6144\` 与 \`max_num_seqs=1\` 限定服务负载。日志中 1 GiB KV 对应约 9360-token capacity，因此对 6144-token 请求显示约 \`1.52x\` concurrency。

Generator-first 时，Omni 驻留后测得空闲约 \`23.59 / 31.36 GiB = 75.2%\`。默认阈值 \`0.92\` 要求约 \`28.85 GiB\` 空闲，因而 Reasoner 在真正按 1 GiB 创建 KV pool 前就被拒绝。Gate 1 将阈值降到 \`0.70\`，要求约 \`21.95 GiB\`，相对当时空闲量留出约 \`1.64 GiB\` 准入余量；实际 KV pool 仍然只有 1 GiB。

后来 Gate 3 在销毁 full-SM graph 并于 GC 上重建 Reasoner 时使用过 \`gpu_memory_utilization=0.55\`。这是重建阶段的准入阈值，不属于 Gate 1 基线，也没有改变 1 GiB KV pool；主文档应把这两个阶段分开写。

#### 1.2.2 Gate 0.5 单模型采样

采样周期为 0.1 秒，原始文件位于 \`profiling_results/gate_0_5_memory/\`：

| 负载 | Ready 中位显存 | 请求峰值 | 高于 Ready | 请求 wall time |
|---|---:|---:|---:|---:|
| Reasoner eager，1 GiB KV，128 tokens | 10048 MiB | 10050 MiB | 2 MiB | 2.157 s |
| Generator，256×256，2-step T2I | 9846 MiB | 9862 MiB | 16 MiB | 0.515 s |
| Generator，832×480×121，35-step T2V | 9847 MiB | 15755 MiB | 5908 MiB | 34.456 s |

对应证据：

\`\`\`text
profiling_results/gate_0_5_memory/reasoner.summary.json
profiling_results/gate_0_5_memory/generator.summary.json
profiling_results/gate_0_5_memory/generator_t2v.summary.json
profiling_results/gate_0_5_memory/*.memory.csv
\`\`\`

将 Reasoner ready 与代表性 T2V 单模型峰值机械相加为 \`25803 MiB\`，相对 32607 MiB 物理显存还剩约 6804 MiB。它只用于进入 Gate 1 前的可行性判断，不等同于双 Engine 同时驻留时的严格峰值。

#### 1.2.3 Gate 1 双 Engine 的时间与阶段显存

2600-token cap、832×480×121、35-step T2V、SeaCache on，1 次 warmup + 3 次 measured：

| Reasoner | Generator | Reasoner 中位时间 | Generator 中位时间 | 最后阶段 GPU 占用快照 |
|---|---|---:|---:|---:|
| eager | eager | 39.413 s | 18.253 s | 21294 MiB |
| eager | compile | 38.469 s | 16.494 s | 21294 MiB |
| graph | eager | 7.550 s | 18.354 s | 21704 MiB |
| graph | compile | 7.513 s | 16.572 s | 21576 MiB |

这些记录还给出了初始化阶段的差异：

\`\`\`text
Generator ready：约 9838--9840 MiB
双 Engine ready：eager Reasoner 约 17404 MiB
双 Engine ready：graph Reasoner 约 13996 MiB
Reasoner 推理后：约 14038 或 17406 MiB
Generator 推理后：约 21294--21704 MiB
\`\`\`

需要保持严谨：Gate 1 JSON 的这些值是阶段结束时的 \`nvidia-smi\` 快照，不是后台高频采样得到的瞬时峰值；文档中更适合称为“阶段占用快照/观测最大值”，而不是严格峰值。Gate 0.5 的 CSV 才是 0.1 秒周期的连续采样。

### 1.3 “借用并行组”在代码中具体是什么

“借用”不是复制 NCCL communicator，也不是创建同配置的新 process group。它指：第二个 Engine 直接读取第一个 Engine 已经写入的进程全局 coordinator 对象；对第二个 Engine 额外要求但缺失的槽位，则保存同一个 Python 对象引用。

#### 1.3.1 Generator-first

Omni 先初始化后，\`vllm.distributed.parallel_state._TP/_DP/_PP\` 已经存在。Reasoner 构造前执行：

\`\`\`python
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
\`\`\`

这里有两层复用：

- Reasoner 对 TP/DP/PP 不需要显式赋值，因为它本来就从 \`vllm_ps\` 全局槽读取 Omni 已创建的对象；
- rank 1 下缺失的 PCP/DCP/ETP 被显式设置为 \`_TP\` 的同一对象引用，即 \`vllm_ps._PCP is vllm_ps._TP\`，不是新建 group。

关闭前只清空借用者增加的别名：

\`\`\`python
def clear_reasoner_context_group_aliases() -> None:
    for name in ("_PCP", "_DCP", "_ETP"):
        if getattr(vllm_ps, name) is vllm_ps._TP:
            setattr(vllm_ps, name, None)
\`\`\`

#### 1.3.2 Reasoner-first

Reasoner 先创建 vLLM 全局 TP/DP/PP。Omni diffusion worker 原本会再次初始化整套进程级 parallel state；我们在导入后替换它已经绑定的函数名：

\`\`\`python
diffusion_worker.initialize_model_parallel = initialize_shared
diffusion_worker.destroy_distributed_env = destroy_shared
\`\`\`

替代初始化先要求 vLLM groups 已存在：

\`\`\`python
if any(group is None for group in (vllm_ps._TP, vllm_ps._DP, vllm_ps._PP)):
    raise RuntimeError("Reasoner vLLM groups must exist before shared Omni initialization")
\`\`\`

随后只创建 Omni diffusion 自己的单 rank 组：

\`\`\`python
singleton = [[0]]
omni_ps._DP = omni_ps.init_model_parallel_group(singleton, ..., "data")
omni_ps._CFG = omni_ps.init_model_parallel_group(singleton, ..., "classifier_free_guidance")
omni_ps._PP = omni_ps.init_model_parallel_group(singleton, ..., "pipeline")
...
omni_ps._SP = omni_ps.init_model_parallel_group(singleton, ..., "sequence", ...)
\`\`\`

因此这里的“借用”表现为：不重新初始化或销毁 vLLM 的进程全局 TP/DP/PP/default group；Omni 只补齐 diffusion 路径自己负责的 \`_DP/_CFG/_PP/_SP\`。当前逻辑显式拒绝 world size、TP/DP/PP/SP/CFG 中任意一项大于 1，以及 EP/HSDP。

### 1.4 为什么必须补 \`graph_capture(context)\` coordinator 契约

这不是 Python 继承链上的 override 问题，而是**版本错位造成的结构化接口（duck-typing）不兼容**。

Omni 的 diffusion \`GroupCoordinator\` 是从较早的 vLLM/xDiT 实现适配而来的独立类，并不是当前 vLLM 0.31 \`GroupCoordinator\` 的子类。Omni-first 后，vLLM 的 \`_TP/_DP/_PP\` 槽里保存的是 Omni 创建的 coordinator。普通 eager 推理只使用这些对象已有的 rank、process-group 和通信接口，所以可以工作；CUDA Graph capture 额外要求每个 coordinator 实现 \`graph_capture(context)\`。

vLLM 0.31 的真实调用是：

\`\`\`python
@contextmanager
def graph_capture(device: torch.device):
    context = GraphCaptureContext(torch.cuda.Stream(device=device))
    with (
        get_tp_group().graph_capture(context),
        get_pp_group().graph_capture(context),
        get_dp_group().graph_capture(context),
    ):
        yield context
\`\`\`

因此它不是可选 hook。Python 进入 \`with\` 之前就必须在三个对象上找到该方法。首次 Generator-first graph 实验的原始错误为：

\`\`\`text
文件：profiling_results/gate_1_single_process/representative_serial_graph_cache_on_v1.log
vllm/v1/worker/gpu/cudagraph_utils.py:415
    with graph_capture(device=self.device), ExitStack() as stack:
vllm/distributed/parallel_state.py:1658
    get_tp_group().graph_capture(context),
AttributeError: 'GroupCoordinator' object has no attribute 'graph_capture'
\`\`\`

兼容层只在缺少方法的单 rank coordinator 实例上用 \`MethodType\` 动态补齐：

\`\`\`python
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

...
if not hasattr(group, "graph_capture"):
    group.graph_capture = MethodType(graph_capture, group)
\`\`\`

这个补丁只实现 rank 1 下确实需要的 stream 排序和共享 capture context。它没有伪造 NCCL/custom-all-reduce capture，因此不能推广为多 rank coordinator 的兼容实现。

### 1.5 Generator 的所有权和两个初始化顺序的关闭逻辑

关闭逻辑必须同时描述“谁创建了进程级 distributed environment”和“谁只创建了 diffusion 私有组”。

#### Generator-first

此时 Omni Generator 是进程级环境及 vLLM TP/DP/PP 的所有者，Reasoner 是借用者。最终关闭顺序为：

\`\`\`python
clear_reasoner_context_group_aliases()
generator.close()
reasoner.llm_engine.engine_core.shutdown()
\`\`\`

含义是：

1. 先删除 Reasoner 添加的 PCP/DCP/ETP→TP 别名；
2. \`generator.close()\` 关闭 Omni orchestrator、stage pool、diffusion worker，并由所有者销毁其创建的 distributed environment；
3. 此后不再执行模型工作，只释放 Reasoner EngineCore/model/KV/allocator 状态。

早期 \`generator_first_v3.json\` 的推理本身成功，但旧顺序先关闭 Reasoner。Reasoner 的标准 shutdown 提前执行全局 distributed cleanup，使 Omni \`_WORLD\` 仍引用已销毁的 process group；随后 \`generator.close()\` 再次销毁，出现 \`ValueError: Invalid process group specified\`。\`generator_first_v4.json\` 证明修正顺序可正常结束。

#### Reasoner-first

此时 vLLM Reasoner 拥有默认 process group 和 vLLM TP/DP/PP；Generator 只拥有兼容层创建的 Omni diffusion \`_SP/_CFG/_PP/_DP\`。\`generator.close()\` 会进入我们替换的 \`destroy_shared()\`：

\`\`\`python
for name in ("_SP", "_CFG", "_PP", "_DP"):
    group = getattr(omni_ps, name)
    if group is not None:
        group.destroy()
        setattr(omni_ps, name, None)
omni_ps._EXPERT_PARALLEL_GROUP_RANKS = None
omni_ps._WORLD = None
\`\`\`

它只销毁 Generator 自己创建的 diffusion groups，不销毁借来的 vLLM groups/default process group。随后 Reasoner shutdown 作为真正所有者完成进程级清理：

\`\`\`python
generator.close()
reasoner.llm_engine.engine_core.shutdown()
\`\`\`

因此两个顺序表面上都是先调用 \`generator.close()\`，但其内部所有权语义不同：Generator-first 使用 Omni 原生的全局销毁职责；Reasoner-first 使用 patched \`destroy_shared()\`，只销毁 Generator 私有组。

### 1.6 Gate 1 到底写了什么代码，哪些属于源码 patch

> **口径修正。** 本节的关键不是区分“技术代码”和“测试代码”，而是区分：仅组合框架公开/准公开 API 的上层胶水，以及越过 Engine API、直接处理框架私有全局状态和 worker 生命周期的框架内兼容逻辑。下述统计完全排除 PID/显存采集、输入构造、warmup/repeat、CLI、结果序列化和分析代码。

#### 结论

Gate 1 不是重写模型，也没有把补丁永久写入安装的 vLLM 0.31 或 vLLM-Omni 执行源码。主要工作分为：

1. 我们仓库中的单进程双 Engine 构造与验证 runner；
2. 我们仓库中的单 rank parallel-state 兼容层；
3. 运行时 monkey patch：替换 Omni worker 已绑定的初始化/销毁函数，并给 coordinator 实例动态增加方法；
4. 隔离环境脚本对 vLLM-Omni \`requirements/common.txt\` 的 Transformers 版本约束做一行 metadata 修改。

因此 3.2--3.7 不是全都属于“修改上游源码”：

- 3.2、3.3 是我们的 Engine 构造参数和进程环境；
- 3.4--3.7 的兼容行为实现在仓库内，通过运行时 monkey patch/对象别名生效；
- 唯一直接改隔离 Omni checkout 文件的是依赖约束，它不改变模型或 runtime 算法。

#### 当前源文件的物理 LOC 口径

以下按当前文件函数边界统计，包含注释、docstring 和空行；不是 Git commit 的净新增行，也不把 CLI、prompt 模板、请求构造和性能统计全部冒充为 runtime 改造：

| 部分 | 当前物理 LOC | 说明 |
|---|---:|---|
| Gate 1 parallel-state 兼容 | 139 | \`_install_single_rank_graph_capture_compat\` 31；\`install_reasoner_context_groups_from_omni\` 12；\`clear_reasoner_context_group_aliases\` 7；\`install_single_rank_shared_parallel_state\` 89 |
| Engine 构造函数 | 79 | \`create_reasoner\` 42；\`create_generator\` 37；包含执行模式、KV、SeaCache 和超时参数 |
| 双顺序构造、检查与最终关闭控制 | 约 55 | \`main()\` 中 461--482、502--535 的集成控制；按连续源码区间估算，含少量空行/注释 |
| PID/GPU PID/CUDA context 证据采集 | 47 | \`cuda_context\` 8；\`process_snapshot\` 39 |
| **Gate 1 runtime 集成小计** | **约 273** | 前三项；不含证据采集 |
| **含验证采集** | **约 320** | 加上 47 LOC 快照代码 |
| 隔离环境安装脚本 | 72 | clone、固定版本、建 venv、安装和导入检查；单独计，不属于 runtime |

\`omni_shared_parallel.py\` 当前总计 198 行，其中还有后来 Gate 3 为“重建 Reasoner 但保留共享 groups”新增的 \`shutdown_reasoner_preserving_shared_parallel_state\` 36 行；它不应倒算为 Gate 1 原始工作量，所以上表排除了这 36 行。

\`single_process_dual_engine.py\` 当前总计 539 行，但其中大部分是输入构造、warmup/repeat、结果序列化和 CLI。把整个文件都算成“框架适配代码”会显著夸大 Gate 1 的技术修改量。

#### 上游源码修改量

\`\`\`text
vLLM 执行源码持久修改：             0 LOC
vLLM-Omni 执行源码持久修改：        0 LOC
隔离 Omni checkout 依赖 metadata：  1 行约束替换
仓库内 runtime monkey patch/兼容层：见上表，约 139 LOC 核心兼容代码
\`\`\`

这里的“0 LOC”是指没有直接编辑安装包中的 model runner、scheduler、attention 或 distributed 实现。运行时 monkey patch 仍然是实际工程改造，只是改造代码保存在本仓库，程序启动后通过 Python 对象赋值生效；它不应被误写成“完全没有 patch”。


#### 按框架 API 边界重新统计

| 层次 | 物理 LOC | 做了什么 |
|---|---:|---|
| API 之上：Engine 配置和构造包装 | 79 | 包装 `vllm.LLM(...)` 与 `vllm_omni.Omni(...)`，设置 KV、执行模式、SeaCache、超时等参数 |
| API 之上：双初始化顺序和生命周期编排 | 约 44 | Generator-first/Reasoner-first 构造顺序，以及最终关闭顺序；不含结果检查和序列化 |
| API 之上：导入前进程配置 | 3 | 禁用 vLLM EngineCore multiprocessing 和 DeepGEMM |
| **API 之上小计** | **约 126 LOC** | 单进程双 Engine 的上层部署/生命周期胶水 |
| API 以下：Generator-first group 借用 | 19 | 操作 vLLM 私有 `_TP/_PCP/_DCP/_ETP` 槽并清理对象别名 |
| API 以下：CUDA Graph coordinator 兼容 | 31 | 给 Omni coordinator 实例动态补 `graph_capture(context)` |
| API 以下：Reasoner-first Omni 初始化/销毁替换 | 89 | 替换 diffusion worker 已绑定的 `initialize_model_parallel` 和 `destroy_distributed_env`，只创建/销毁 Omni 私有 groups |
| **API 以下小计** | **139 LOC** | 直接涉及 private parallel state、coordinator 契约和 worker lifecycle 的框架兼容逻辑 |
| **Gate 1 非测试代码总计** | **约 265 LOC** | 约 126 LOC 上层胶水 + 139 LOC API 以下兼容 |

这里“API 之上约 126 LOC”仍是偏宽松口径：两个 Engine 构造函数中包含大量普通参数展开。如果只计算控制结构而不计算参数列表，真正的调度逻辑更接近 50 行；但为了让数字可从源码行号复核，本表保留完整构造函数。

“API 以下 139 LOC”则是 Gate 1 最核心、也最不能被官方离线 API 自动提供的工作：如果删除这一层，仅保留 `LLM(...)`、`Omni(...)` 和串行调用，两种构造顺序至少有一种会因重复 parallel-state 初始化、coordinator 接口缺失或关闭时 double destroy 而失败。

另有两点边界：

1. `reasoner.llm_engine.engine_core.shutdown()` 本身访问了 vLLM 私有对象链，但它在 Gate 1 里只是 1 行生命周期调用，已随上层关闭编排统计，没有因此把整段算入 139 LOC 框架兼容层。
2. 上述 139 LOC 是保存在本仓库、启动时注入的 monkey patch，不是对 site-packages 或上游 checkout 的持久源码 diff；从“是否修改框架行为”看它属于框架逻辑修改，从“是否直接 fork/编辑上游源码”看仍是 0 LOC。

---

## 2. Gate 2 阅读问题：并发骨架、后端优化与并行组

### 2.1 Gate 2 是否几乎没有 technical 改动

这个判断需要按边界拆开。**Gate 2 没有新增 API 以下的框架兼容修改，也没有直接修改 vLLM/vLLM-Omni 上游源码；但它新增了 API 之上的并发实验调度骨架。**

Gate 2 直接复用了 Gate 1 已经完成的工作：

- 两个 offline Engine 在同一个 PID 中构造；
- Generator 先初始化，Reasoner 借用已经存在的 process-global parallel groups；
- 显存预算、单进程 EngineCore、CUDA Graph coordinator 兼容和关闭所有权；
- 两个后端各自已有的 eager/compile/graph 执行能力。

Gate 2 本身新增的运行逻辑可以概括为：

```python
generator = create_generator(...)
reasoner = create_reasoner(borrow_omni_parallel_state=True, ...)

reasoner_stream = torch.cuda.Stream()
generator_stream = torch.cuda.Stream()

start_barrier = threading.Barrier(3)
reasoner_future = executor.submit(reasoner_worker)
generator_future = executor.submit(generator_worker)
start_barrier.wait(timeout=60.0)

reasoner_result = reasoner_future.result()
generator_result = generator_future.result()
```

每个 worker 在自己的 host thread 中调用已经构造好的 Engine；eager 路径还进入各自的普通 CUDA stream。这里没有 Green Context，也没有 SM 空间隔离。两个线程只是让两个框架有机会同时向 GPU 提交工作，最终是 kernel overlap 还是 time multiplexing，要靠 Nsys 判断，不能由 host wall overlap 推断。

因此 Gate 2 的代码工作量应写成：

| 边界 | Gate 2 新增量 | 含义 |
|---|---:|---|
| 直接修改 vLLM/vLLM-Omni 上游源码 | 0 LOC | 没有新增源码 patch |
| API 以下 private runtime 兼容 | 0 LOC | 沿用 Gate 1 的约 139 LOC，没有新增并行组/coordinator patch |
| API 之上核心并发控制 | 约 40--50 LOC | 两个 worker、barrier、submit/join、stream context |
| `concurrent_once()` 全部物理代码 | 约 92 LOC | 还包含计时、NVTX、异常传播和 overlap 统计，不应全部算作部署逻辑 |

Gate 2 更准确的定位是一个**并发可行性和干扰测量 gate**，而不是完整生产 scheduler。它回答“同进程中的两个现有 Engine 能否同时执行、干扰有多大”，没有实现请求队列、准入、抢占、动态分区或服务级调度。

另一个重要的技术选择是：Reasoner graph 模式不强行套外部普通 stream，而保留 vLLM capture/replay 自己的 stream/event 契约。这个选择避免破坏 graph replay，但它是调用策略，不是新的底层 graph 实现。

### 2.2 对主文档 4.3 中各项优化的机制级解释

这里关心的是每项优化**具体改写了哪段执行过程**，而不是它由谁实现。下面只写能由配置、源码或运行日志确认的机制；未做严格消融的副作用不作扩展。


先明确“来源”的含义：模型 checkpoint 提供权重和 Cosmos3/Wan 的网络结构；Paged KV、compile、CUDA Graph、cache 和 fast path 都是推理框架在执行模型时提供的能力，不是 checkpoint 自带的优化。

| 能力 | 本实验中的实现来源 | 与原生 Cosmos 基线的关系 |
|---|---|---|
| Paged KV、block table、KV scheduler | vLLM | 替代原生 Reasoner 的连续/static KV 管理路径 |
| FlashAttention v2 backend | vLLM/CUDA attention backend | 原生 Cosmos 也使用 FlashAttention 类优化，但这里实际执行的是 vLLM backend |
| AOT compile、`FULL_AND_PIECEWISE` CUDA Graph | vLLM | serving 框架优化，不来自模型权重 |
| BF16 | 模型 dtype + 框架执行配置 | 与基模 BF16 路径对齐，不是额外算法 |
| Regional `torch.compile` | vLLM-Omni diffusion runtime | Omni 对 Cosmos3 Generator 重复 block 的执行优化 |
| SeaCache | vLLM-Omni 的 Cosmos3 SeaCache backend | 参数与原生 Cosmos cache-on 基线对齐；本实验实际运行的是 Omni hook/backend |
| Wan VAE lossless fast path | vLLM-Omni 针对 Wan VAE 的实现 | VAE 结构和权重来自模型；cuDNN/Triton 替换实现来自 Omni |
| VAE tiling/slicing | Diffusers 风格 VAE 能力，经 vLLM-Omni 配置 | 不是 checkpoint 优化；当前保持默认关闭 |


#### Reasoner：Paged KV cache

普通逐 token decode 在概念上需要不断保存每层历史 token 的 K/V。Paged KV 将这部分显存预先组织成固定大小的物理 block pool；scheduler 为每个请求维护 block table，把逻辑 token 位置映射到物理 block：

```text
request logical tokens
  [0 ... block_size-1]      -> physical block 17
  [block_size ... 2B-1]     -> physical block 03
  ...

attention metadata
  -> block table
  -> FlashAttention 从对应的 KV blocks 读取历史 K/V
```

因此 decode 新增 token 时，主要动作是给新 token 分配/定位可用 slot、写入该 token 的 K/V，并更新元数据，而不是重新申请一块更大的连续 tensor 再复制完整历史 KV。实验显式指定 1 GiB KV pool，实际含义是固定 vLLM 可用于这些 KV blocks 的显存预算，并使双 Engine 初始化时的显存需求可预测。

它和原生 Cosmos 的 static KV 目标相似——都避免 KV tensor 随 token 逐步扩容——但物理组织不同：static KV 是按最大长度预留连续槽位，Paged KV 是 pool + block table。

#### Reasoner：FlashAttention v2

标准 attention 可以写成：

```text
S = Q @ K^T
P = softmax(S)
O = P @ V
```

朴素实现会把较大的 `S/P` 中间结果写入显存。FlashAttention 将 QK、online softmax 和 PV 按 tile 融合，在片上存储中维护局部统计量，最后只写输出；decode 时再通过 block table 访问 Paged KV。它直接减少的是：

- attention score/probability 中间张量的 HBM 读写；
- 被拆开的 QK、softmax、PV kernel launch；
- 对 padding token 的无效计算（结合变长序列 metadata）。

在我们的路径中，Paged KV 负责“历史 KV 放在哪里”，FlashAttention 负责“如何消费这些 KV 完成 attention”，两者是配套关系，不是两个互相替代的 cache。

#### Reasoner：AOT compile 与 `FULL_AND_PIECEWISE` CUDA Graph

这项配置包含两个层次。

第一层是 compile：vLLM 将可编译的模型区域交给编译后端，生成专门化执行代码，并有机会融合相邻算子。直接减少 Python dispatcher、通用 eager operator 和一部分零碎 kernel 的开销。

第二层是 CUDA Graph：warmup/capture 阶段记录一组 GPU 操作及其依赖，后续相同 execution descriptor 不再由 CPU 逐 kernel 重新提交，而是执行 graph replay：

```text
Eager token step:
  Python/dispatcher -> launch A -> launch B -> launch C -> ...

Captured token step:
  graph key/shape dispatch -> cudaGraphLaunch(captured DAG)
```

`FULL_AND_PIECEWISE` 的含义是：满足条件的执行形态尽量走完整 graph；不能整体捕获的形态在 graph break 之间使用 piecewise graph，其余部分保留普通执行。vLLM 会针对支持的 batch/capture size 保存相应 graph，并在运行时根据当前 batch descriptor 选择。

我们在 Nsys 中已经看到大量 `cudaGraphLaunch_v10000`，因此 replay 确实发生。Gate 2 保留 vLLM 自己的 capture/replay stream 和 event 协议，因为 replay 使用的是 capture 时建立的执行依赖；外层临时切换一个普通 stream 并不能重新定义 graph 内节点的 stream 归属。

#### Reasoner：BF16、无量化与 prefix cache off

BF16 让权重、activation 和 KV 使用 16-bit 浮点路径，Tensor Core GEMM/GEMV 也按相应 dtype 执行。无量化表示本实验不引入 INT8/FP8/INT4 权重格式以及对应的 scale、quantize/dequantize kernel。

`prefix cache off` 则是实验控制：即使两个请求具有相同前缀，也不会直接复用上一个请求已经计算出的 prefix KV。这样每轮请求都会实际执行当前输入对应的 prefill，避免 cache hit 改变工作量。

#### Generator：regional `torch.compile`

Omni 不对整个 T2V pipeline 做一次整体 CUDA Graph capture，而是寻找 diffusion Transformer 声明的 `_repeated_blocks`，对这些重复 block 安装编译后的 callable：

```text
35 diffusion steps
  -> each step invokes repeated DiT blocks
       eager: Python module/operator dispatch on every invocation
       compile: invoke compiled region; fused/specialized kernels execute
```

这主要作用在反复执行的 DiT block，减少重复 Python/operator dispatch，并允许 Inductor 融合适合的逐元素/布局操作。condition preparation、scheduler 的 Python 控制、VAE decode 和输出保存并不会因为 DiT regional compile 自动变成同一个 compiled region。

它也不是持久 CUDA Graph：编译后的 Triton/cuBLAS/cuDNN kernel 仍在每次调用时提交。因此后续 Gate 3 才能够在 Omni worker 的真实执行边界切换 current stream，使 compiled kernel 进入 Generator GC stream。

#### Generator：SeaCache

SeaCache 针对相邻 diffusion steps 的 Transformer 结果变化较平缓这一性质，决定当前 step 是执行完整 Transformer，还是使用历史 residual 做外推。

其主要控制流程可简化为：

```python
indicator = spectral_filter(current_latent, sigma, power_exp)
distance = indicator_distance(indicator, previous_indicator)
accumulated_distance += distance

if first_or_last_step or history_invalid:
    run_full_transformer()
elif consecutive_cached >= max_consecutive_cached:
    run_full_transformer()
elif accumulated_distance >= threshold:
    run_full_transformer()
else:
    extrapolate_cached_residual(order=residual_order)
```

当前配置的具体作用：

- `threshold=0.25`：累计 spectral indicator distance 达到阈值时刷新完整 Transformer 结果；
- `residual_order=1`：缓存 step 使用一阶历史 residual 外推；
- `max_consecutive_cached=2`：最多连续两个 step 不运行完整 Transformer，之后强制刷新；
- `power_exp=3.0`：构造 spectral indicator 时使用的频谱权重指数。

首次 step、最后 step、没有历史、step 不连续、indicator 不可用等情况直接执行完整 Transformer。它减少的是部分 diffusion step 中完整 DiT forward 的次数，因此会直接改变 sampling 阶段实际发射的 Transformer kernel 数量。我们的各组对照保持 SeaCache 参数一致。

#### Generator：Wan VAE lossless fast path

该 fast path 替换的是 Wan VAE encoder/decoder 内部的一些执行实现，而不是 diffusion sampling：

```text
latent/video
  -> padding / causal cache update
  -> Conv3D / residual blocks / normalization / activation
  -> spatial-temporal upsample or downsample
  -> video/latent
```

其中 convolution 走 cuDNN；padding、normalization、residual 和部分 upsample/data-layout 操作使用融合 Triton kernel，减少独立逐元素 kernel 和中间 tensor 搬运。`lossless` 档的目标是保持原运算定义与输出等价，只替换执行实现。它同时覆盖所使用的 Wan VAE encode/decode 路径，但不会加速 Reasoner 或 DiT attention。

#### 未开启的三个能力具体会改变什么

- `layerwise offload`：在层执行前后搬运 CPU/GPU 参数，用传输换显存；当前关闭。
- `prompt embedding cache`：相同 prompt/conditioning 命中时直接复用已编码 embedding；当前关闭。
- `VAE tiling/slicing`：是显存优化开关；Omni CLI 也明确将其描述为用于缓解 OOM，默认均为 `False`。

当前没有开启 VAE tiling/slicing，原因不是它“不好”，而是本轮固定的 832x480、121-frame、batch=1 workload 在现有显存预算下能够完成；研究目标是测完整 VAE decode 与 Reasoner 的干扰，不需要为避免 OOM 改写 VAE workload。保持关闭也与此前非分块 VAE 的 profiling/scaling 基线一致。

两者需要分别理解：

**VAE slicing 是 batch slicing。** Wan VAE 代码只在 `use_slicing=True` 且 batch size 大于 1 时逐样本处理。当前 batch=1，因此即使打开 slicing，按这一路径也不会实质拆分 workload。batch>1 时，它会把一个较大的 batched VAE 调用改成多个较小调用：峰值 activation 显存下降，但 batched 大 kernel 会变成多轮较小 kernel，launch 数量和串行片段增加。

**VAE tiling 是空间 tiling。** 当 latent 的宽或高超过 VAE 的 tile 阈值时，输入被切成带 overlap 的空间 tile；每个 tile 分别经过 `post_quant_conv` 和 decoder，随后执行横向/纵向 overlap blending、裁剪和拼接：

```text
full latent [T,H,W]
  -> overlapping spatial tiles
  -> decoder(tile_0), decoder(tile_1), ...
  -> blend_h / blend_v
  -> crop + concatenate
  -> full video
```

它会把负载特征从“少量、尺寸较大的 convolution/upsample kernel”改成“多轮、尺寸较小的同类 kernel + blend/crop/concat kernel”：

- 单个 tile 的 activation 较小，峰值显存下降；
- overlap 区域会被重复计算；
- kernel launch 数量增加，单 kernel 尺寸和持续时间下降；
- VAE 的 SM scaling、kernel 间 gap 和与另一负载的可重叠粒度都会随之变化。

最后一点正是本实验暂时不开 tiling 的主要方法学原因：它不是只改变显存占用，也会改变我们正在测量的 VAE workload 形状。若以后因更高分辨率/更长视频而必须开启，应把 tiled VAE 当作一条新的 profiling 与 colocation 基线，而不能直接与当前 non-tiled 曲线混用。

因此 Gate 2 的稳态 workload 仍然完整执行当前请求的 prompt conditioning 和 non-tiled、non-sliced VAE；diffusion sampling 使用固定的 regional compile 选择和 SeaCache 配置。

### 2.3 Gate 2 并发与 Gate 1 并行组初始化的关系

两者处于不同层次：

```text
串行生命周期阶段
  Generator 初始化并建立 process-global groups
  Reasoner 初始化并借用这些 groups
  串行 warmup
          |
          v
并发执行阶段
  host thread R -> Reasoner Engine -> CUDA work
  host thread G -> Generator Engine -> CUDA work
          |
          v
串行关闭阶段
  等待两个 worker 结束
  按所有权关闭 Reasoner / Generator
```

Gate 2 的并发发生在两个 Engine 和 parallel groups 已经初始化完成之后。并发请求不会重新运行 3.4/3.5 的初始化，不会为每个线程创建一套 group，也不会改变谁拥有 group。初始化仍由主线程串行完成；所有并发 future join 后，关闭也串行执行。

CUDA stream 和 parallel group 也不是同一个概念：

- CUDA stream 决定当前进程向 GPU 提交工作的顺序和依赖；
- parallel group 描述 rank 拓扑、collective communicator 和分布式通信所有权。

当前实验是单 GPU、`world_size=1`，推理阶段没有真正的跨 rank collective 流量，所以 parallel group 本身不是 Gate 2 的性能竞争资源。它的重要性是让两个框架在同一进程中不因重复初始化或 double destroy 而失败。

不过它们存在生命周期上的约束：**不能让两个线程并发初始化或并发销毁 process-global groups。** vLLM 和 Omni 都访问模块级全局槽、coordinator 和 distributed process group；如果把构造也放入两个 worker 并发执行，会有覆盖全局状态、读取半初始化对象和重复销毁的风险。当前实现正是通过“串行初始化 -> 只读复用 -> 并发执行 -> 串行关闭”避开这个问题。

如果未来扩到多 GPU/多 rank，结论会改变：两个模型同时执行可能在 NCCL communicator、collective 顺序和线程安全上发生竞争，仅靠 Gate 1 的单卡 group 借用不能直接外推。当前兼容层因此把适用范围严格限制在单进程、单 rank。

---

## 3. Gate 3 实验结果补缺审计

这一节先记录已经落盘但主文档没有完整呈现的数据，并明确仍缺少的实验格子；不把单轮结果写成正式多轮结论。

### 3.1 Gate 3 是否写全了实际启用的后端优化

主文档 Gate 3 已经在 5.3 说明 Generator regional compile 的 GC stream 迁移与 Nsys placement 验证，但 6.1 的主要性能表仍然只有 `graph + generator eager`，没有列出 compile 修复后的正式性能结果。因此“技术实现写到了，结果表没有补全”。

Gate 3 的后端配置应当像 Gate 2 一样明确列为：

| 路径 | Gate 3 实际启用/固定的能力 | 相比 Gate 2 的变化 |
|---|---|---|
| Reasoner | vLLM Paged KV（1 GiB pool）、FlashAttention backend、BF16、无量化、prefix cache off；`eager` 或 AOT compile + `FULL_AND_PIECEWISE` CUDA Graph | 算法/后端配置不变；新增 eager runner stream 绑定，或在 Reasoner GC 上重新 capture graph |
| Generator diffusion | SeaCache on；`eager` 或 vLLM-Omni regional `torch.compile(dynamic=True)` | 后端配置与 Gate 2 对齐；新增 worker 内 GC stream 注入，并验证 compiled kernel 可随 current stream 迁移 |
| Generator VAE | Wan VAE lossless fast path；cuDNN convolution 与 Triton fused ops | 配置与 Gate 2 对齐；新增 output pack/postprocess GC stream 覆盖 |
| 未启用 | layerwise offload、VAE tiling/slicing、prompt embedding cache | 与 Gate 2 一致 |

这里需要避免一个误解：GC 本身不替代 compile、SeaCache、Paged KV 或 VAE fast path。Gate 3 是在 Gate 2 的后端优化之上改变 kernel 的资源归属与并发方式。

### 3.2 Generator compile 修复后的正式结果

现有正式文件：

```text
profiling_results/decode_diffusion_sampling_matched/
  vllm_graph_omni_compile_g120_r50_tok1350x2_repeat3.json
  vllm_graph_omni_compile_g120_r50_tok1350x2_repeat3.log
```

固定 workload：

```text
Reasoner: vLLM graph，固定 1350 completion tokens，请求 2 次
Generator: T2V 832x480，121 frames，35 steps，CFG 6.0
Generator backend: regional compile + SeaCache on
GC split: Generator 120 SM，Reasoner 50 SM
warmup 后 full-SM / partition-alone / colocated 各 3 次，以下取中位数
```

端到端阶段数据：

| 指标 | 时间/比值 |
|---|---:|
| Reasoner full-SM alone（2 requests） | 8.668 s |
| Generator full-SM alone | 17.828 s |
| full-SM 串行 wall | 26.496 s |
| Reasoner 50-SM partition-alone | 10.257 s |
| Generator 120-SM partition-alone | 23.950 s |
| partition 串行 wall | 34.207 s |
| Reasoner colocated | 11.687 s |
| Generator colocated | 24.771 s |
| colocated wall | 24.770 s |
| Reasoner 共置 slowdown（相对 partition-alone） | 1.139x |
| Generator 共置 slowdown（相对 partition-alone） | 1.034x |
| full-SM serial wall / colocated wall | 1.070x |

这个吞吐收益较低不能只看 slowdown；当前两次 1350-token Reasoner 在 full SM 下仍只有 8.668 s，而 Generator 为 17.828 s，覆盖长度并未配平。

已有的 Generator eager matched 文件只有一轮：

```text
profiling_results/decode_diffusion_sampling_matched/
  vllm_graph_g120_r50_tok1350x2_plain.json
```

它可以作为诊断参考，但不能和 compile 的三轮中位数做严格消融。其 Generator full-SM/partition/colocated 分别为 19.644/26.136/26.921 s；如要正式量化 compile 的收益，仍缺同配置的 eager repeat3。

### 3.3 diffusion sampling-only 结果

同一个 compile repeat3 JSON 已经记录了 diffusion sampling 的 GPU event 时间，主文档没有单独列出：

| diffusion sampling 区间 | 时间 | 相对关系 |
|---|---:|---:|
| full-SM alone | 12.576 s | 1.000x |
| 120-SM partition-alone | 17.257 s | 1.372x（相对 full-SM） |
| 120-SM colocated with Reasoner | 18.099 s | 1.049x（相对 partition-alone） |

因此这一组数据能够区分：

- SM 从 170 限制到 120 带来的 sampling slowdown，约 `17.257 / 12.576 = 1.372x`；
- 与 50-SM Reasoner graph 共置带来的额外 sampling slowdown，约 `18.099 / 17.257 = 1.049x`。

Generator 端到端的 partition slowdown 是 1.343x，与 sampling 的 1.372x 接近，但两者不是同一指标：端到端还包含 condition preparation、VAE decode 和框架开销。

### 3.4 部署框架与原生 Cosmos matched 对照：现有数据

#### 3.4.1 Reasoner decode x diffusion sampling

工作量身份已经对齐：

```text
Reasoner prompt token hash:
  d44adce7cc1a357324c05b9aed8a1941c9c825a1db14f7ba17a4ed97d28fcc8c
Reasoner: 1350 fixed completion tokens x 2 requests
Generator workload hash:
  494faabf76aa384e78148518f7aaf154d60854fd87d6a34792044b9554650e59
Generator: 832x480, 121 frames, 35 steps, CFG 6.0
GC: Generator 120 SM / Reasoner 50 SM
```

目前可直接对照的 partition-alone 与 colocated 数据为：

| 实现 | R partition-alone | R colocated | R slowdown | Sampling partition-alone | Sampling colocated | Sampling slowdown |
|---|---:|---:|---:|---:|---:|---:|
| Native Cosmos graph + native compiled Generator | 16.831 s | 19.482 s | 1.158x | 17.978 s | 20.063 s | 1.116x |
| vLLM graph + Omni regional compile | 10.257 s | 11.687 s | 1.139x | 17.257 s | 18.099 s | 1.049x |

Native 文件：

```text
profiling_results/decode_diffusion_sampling_matched/
  native_graph_g120_r50_tok1350x2_plain.log
```

Serving 文件：

```text
profiling_results/decode_diffusion_sampling_matched/
  vllm_graph_omni_compile_g120_r50_tok1350x2_repeat3.json
```

需要保留统计口径警告：Native 当前只有一轮 measured，serving compile 是三轮中位数；虽然输入、token 数和 Generator workload identity 已对齐，重复次数还没有完全对齐。

更关键的是，Native 这份 runner 没有记录 full-SM Reasoner、full-SM Generator 或 full-SM sampling。Serving 侧已有：

| Serving compile full-SM 指标 | 时间 |
|---|---:|
| Reasoner，2 x 1350 tokens | 8.668 s |
| Generator 端到端 | 17.828 s |
| diffusion sampling | 12.576 s |

因此当前还不能形成完整的“Native vs serving：full-SM -> partition-alone -> colocated”三段对照。缺口是 Native matched workload 的 full-SM 三轮基线，以及最好把 Native partition/colocated 也补成三轮。

#### 3.4.2 Reasoner decode x VAE decode

这组 matched 对照已经在主文档 6.2 出现，但同样缺 Native full-SM。三轮中位数约为：

| 实现 | R partition-alone | R colocated | R slowdown | VAE partition-alone | VAE colocated | VAE slowdown |
|---|---:|---:|---:|---:|---:|---:|
| Native Cosmos graph + native VAE | 8.076 s | 10.922 s | 1.353x | 7.631 s | 8.615 s | 1.129x |
| vLLM graph + Omni VAE fast path | 4.902 s | 8.768 s | 1.788x | 6.285 s | 6.371 s | 1.014x |

Serving 三轮还记录了 full-SM：

| Serving VAE matched full-SM 指标 | 中位数 |
|---|---:|
| Reasoner，1300 tokens | 4.172 s |
| Generator 端到端 | 18.217 s |

Native 三轮日志只记录 partition-alone 与 colocated，没有 full-SM baseline。因此这组也只能严谨回答“固定 50:120 分区后的共置影响”，还不能完整回答两个部署栈各自从 full SM 限制到分区时付出了多少成本。

### 3.5 目前真正缺少的实验格子

| 对照 | full-SM | partition-alone | colocated | 当前状态 |
|---|---|---|---|---|
| vLLM graph x Omni compile diffusion，三轮 | 已有 | 已有 | 已有 | 完整 |
| vLLM graph x Omni eager diffusion，三轮 | 仅单轮 | 仅单轮 | 仅单轮 | 缺正式 repeat3，用于 compile 消融 |
| Native graph x native diffusion，matched | 缺 | 单轮已有 | 单轮已有 | 缺 full-SM 和 repeat3 |
| vLLM graph x Omni VAE，matched | 已有 | 三轮已有 | 三轮已有 | 完整 |
| Native graph x native VAE，matched | 缺 | 三轮已有 | 三轮已有 | 只缺 full-SM |

所以主文档目前缺的不只是“把已有数字抄进去”：diffusion compile 和 sampling-only 可以直接补表；完整 Native-vs-serving 三段对照则仍需要补跑 Native full-SM，不能从 partition 时间反推。
