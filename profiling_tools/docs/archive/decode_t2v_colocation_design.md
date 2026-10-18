# Reasoner Decode 与 T2V Generator 共置设计

> 状态：历史设计稿，仅用于追溯早期接口和实验决策。当前实现与结论见
> `../cosmos3_reasoner_decode_colocation.md` 和
> `../cosmos3_architecture_decode_colocation_study.md`；后续 vLLM 方案见
> `../unified_vllm_generator_colocation_plan.md`。

## 路径约定

本文所有本地文件和目录均使用宿主机绝对路径。当前 Git 仓库根目录是：

```text
/data/cjh/cosmos3/cosmos3-workspace/cosmos-framework
```

该目录在容器内对应：

```text
/workspace/cosmos-framework
```

注意这里存在两个容易混淆的层级：外层 `cosmos-framework` 是 Git 仓库，内层 `cosmos_framework` 是 Python package。例如：

```text
/data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/cosmos_framework/model/generator/mot
└──────────────────────── Git 仓库 ────────────────────────┘└──── Python package ────┘
```

代码块中的 `from profiling_tools...` 等内容是 Python import，不是文件系统相对路径，因此保持模块名写法。

## 1. 目标与粒度

我们要搭建一个保真的单进程运行时，将以下两个负载共置：

- 当前请求的 Reasoner 自回归 decode；
- 上一个请求的完整 T2V generator。

完整 generator 包括：

```text
condition_prepare -> diffusion_sampling -> vae_decode
```

实验必须保留 Cosmos 原本的模型实现、输入、缓存和 CUDA kernel。目标稳态调度为：

```text
generator(request i) || reasoner_decode(request i + 1)
```

不能共置同一请求的 decode 与 generator，因为后者的 prompt 依赖前者的输出。

## 2. 固定负载

使用已经验证的工作流：

```text
/data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/outputs/local_agentic_t2v/rich02/
```

固定参数：

- checkpoint：`Cosmos3-Edge`；
- T2V：480、16:9、5 秒、24 FPS；
- Reasoner 最大输出：1350 token；
- Reasoner：`do_sample=False`；
- generator seed：0；
- diffusion cache：开启。

当前 warm full-SM 参考值：

```text
Reasoner prefill             0.619 s
Reasoner decode             29.159 s
Reasoner 端到端             30.554 s

Generator condition         1.544 s
Generator diffusion        13.139 s
Generator VAE decode        6.843 s
Generator 总计             21.514 s
```

这些只是参考值。共置 runner 必须在同一进程中重新测量相同 SM 分区下的 alone 时间。

## 3. 数据依赖与 20 请求流水线

请求 `i` 的真实依赖为：

```text
reasoner_prefill_i
  -> reasoner_decode_i
  -> upsampled_prompt_i
  -> generator_i
```

因此 20 个请求的保真流水线是：

```text
prefill_0 -> decode_0                                  # fill

准备 generator_0
执行 prefill_1
generator_0 || decode_1                                # overlap 1

准备 generator_1
执行 prefill_2
generator_1 || decode_2                                # overlap 2

...

准备 generator_18
执行 prefill_19
generator_18 || decode_19                              # overlap 19

generator_19                                           # drain
```

20 个请求形成 19 次满足数据依赖的 overlap。当前 decode 尚未产生 prompt 时，下一 generator 不能提前运行。

## 4. 运行时结构

使用一个进程和一个常驻 pipeline，不为每个请求启动新的 inference 子进程：

```text
主线程 / Reasoner 线程                     Generator worker
----------------------                     ----------------
准备 request i+1                           接收 request i
执行 prefill i+1                           构造新的 generator batch
保留活跃 KV cache                          等待 barrier
在 decode region 等待 barrier              等待 barrier
decode i+1 on Reasoner GC          ||       full generator i on Generator GC
解析 Reasoner 输出                         同步 Generator GC stream
将 request i+1 交给下一轮
```

第一版采用严格的 wave barrier。一侧提前结束后，需要等待另一侧完成并产生下一轮所需数据。这是实际依赖，不应人为消除。

## 5. 不序列化 Decode 状态

Decode 的活跃状态包括：

- `ReasonerKVCache`；
- `next_token`；
- repetition/presence penalty mask；
- EOS/finished mask；
- mRoPE position delta；
- sampling RNG state；
- output token buffer。

不要把这些状态写到磁盘，也不要在独立伪造 module 中重建。

保持官方 `_impl_generate_reasoner_text()` 调用处于活跃状态。完成 prefill 后，在 decode 边界暂停，使全部状态继续留在 Python 调用栈和 GPU 显存中；barrier 释放后继续运行原始 decode loop。

每个请求必须创建新的 KV cache 并重新 prefill，不能复用已经增长过的 cache。

## 6. 核心代码修改

### 6.1 通用 Runtime Region Hook

新增：

```text
/data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/cosmos_framework/runtime/regions.py
```

提供上下文局部的 region factory：

```python
def runtime_region(name: str): ...
def install_region_factory(factory): ...
def reset_region_factory(token): ...
```

未安装实验 runtime 时，`runtime_region()` 返回 `nullcontext()`，普通 Cosmos 推理的行为必须完全不变。

使用 `contextvars`，不要使用未保护的全局 callback，避免 Reasoner 和 generator 线程错误继承对方配置。

#### 6.1.1 三个接口的底层技术

三个接口应建立在 Python 标准库的 `ContextVar`、`Token`、`ContextManager` 和 `nullcontext` 上。它们只负责依赖反转和上下文隔离，不直接创建 Green Context：

```python
from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar, Token
from typing import Callable, ContextManager, TypeAlias

RegionFactory: TypeAlias = Callable[[str], ContextManager[None]]

_REGION_FACTORY: ContextVar[RegionFactory | None] = ContextVar(
    "cosmos_runtime_region_factory",
    default=None,
)


def runtime_region(name: str) -> ContextManager[None]:
    factory = _REGION_FACTORY.get()
    if factory is None:
        return nullcontext()
    return factory(name)


def install_region_factory(
    factory: RegionFactory,
) -> Token[RegionFactory | None]:
    return _REGION_FACTORY.set(factory)


def reset_region_factory(token: Token[RegionFactory | None]) -> None:
    _REGION_FACTORY.reset(token)
```

各接口的职责如下：

| 接口 | 底层机制 | 职责 |
| --- | --- | --- |
| `runtime_region(name)` | `ContextVar.get()` + `nullcontext()` | 模型代码只声明逻辑区域；没有 runtime 时保持原行为 |
| `install_region_factory(factory)` | `ContextVar.set()` | 在当前执行上下文安装实验 runtime，并返回可恢复的 `Token` |
| `reset_region_factory(token)` | `ContextVar.reset()` | 精确恢复安装前的 factory，支持嵌套安装和异常清理 |

`Token` 不能用简单的 `set(None)` 代替。假设外层已经安装 factory A，内层临时安装 factory B，使用 token reset 可以恢复 A；直接写入 `None` 会破坏外层上下文。

安装和恢复必须使用 `try/finally`：

```python
token = install_region_factory(region_factory)
try:
    run_reasoner_request()
finally:
    reset_region_factory(token)
```

token 必须在执行 `_REGION_FACTORY.set()` 的同一 logical context 中 reset，并按嵌套关系逆序恢复。

#### 6.1.2 线程语义

`ContextVar` 为线程和异步 task 提供独立的 logical context，但实验不能依赖新线程是否继承创建者上下文。在哪个线程执行 Reasoner，就在哪个线程显式安装 factory：

```python
def reasoner_worker():
    token = install_region_factory(region_factory)
    try:
        run_reasoner_request()
    finally:
        reset_region_factory(token)
```

Generator worker 不安装 Reasoner factory；它直接在自己的 Generator GC stream 中调用完整 generator。这样 generator 内部即使以后也出现同名 `runtime_region()`，也不会误用 Reasoner 的 barrier 和 stream。

如果 Reasoner 始终在主线程运行，则在主线程安装即可。不要在模块 import 时永久安装 factory。

#### 6.1.3 Factory 如何连接 GC Runtime

`RegionFactory` 是由实验 runner 创建的 closure。它可以捕获以下对象：

- Reasoner `torch.cuda.ExternalStream`；
- overlap `threading.Barrier`；
- CUDA start/end event；
- trial ID 和计时记录器；
- 异常/取消状态。

示意：

```python
def make_region_factory(runtime):
    def factory(name: str) -> ContextManager[None]:
        if name == "reasoner_decode":
            return runtime.reasoner_decode_region()
        return nullcontext()

    return factory
```

未知 region 必须返回 `nullcontext()`，不要因为以后 Cosmos 增加了新 region 而使旧 runner 失败。

#### 6.1.4 现有 Green Context 底层支持

直接复用：

```text
/data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/profiling_tools/green_context.py
```

共置必须使用 `GreenContextPair`，不能独立创建两个 `GreenContextStream`。`GreenContextPair` 通过 CUDA Driver API 从同一个设备 SM resource 中切出第一组和 remainder，从而保证两组 SM 不重叠：

```python
from profiling_tools.core.green_context import GreenContextPair

owner = GreenContextPair(
    requested_first_sm_count=reasoner_sm_count,
    device=0,
)

reasoner_stream = torch.cuda.ExternalStream(
    owner.first_stream_ptr,
    device=owner.device,
)
generator_stream = torch.cuda.ExternalStream(
    owner.second_stream_ptr,
    device=owner.device,
)
```

这里的技术层次是：

```text
runtime_region(name)
    -> ContextVar 中的 RegionFactory
        -> ReasonerDecodeRegion context manager
            -> threading.Barrier
            -> torch.cuda.Event
            -> torch.cuda.ExternalStream
                -> GreenContextPair 创建的 CUDA Driver stream
                    -> 互不重叠的 SM resource
```

第一版保持 `ignore_sm_coscheduling=False`。是否启用该 CUDA flag 应作为后续独立实验变量，不能混入初始正确性验证。

`GreenContextPair` owner 必须比两个 `ExternalStream` 活得更久，退出前先同步两个 stream，再调用 `owner.close()`：

```python
owner = GreenContextPair(reasoner_sm_count, device=0)
try:
    reasoner_stream = torch.cuda.ExternalStream(owner.first_stream_ptr, device=0)
    generator_stream = torch.cuda.ExternalStream(owner.second_stream_ptr, device=0)
    run_trials()
    reasoner_stream.synchronize()
    generator_stream.synchronize()
finally:
    owner.close()
```

核心模型代码中的 `regions.py` 不应 import `profiling_tools.core.green_context`。依赖方向必须保持为：模型层只声明 region，实验 runner 决定 region 使用 GC、普通 stream，还是 no-op。

### 6.2 标记完整 Reasoner Decode 区域

修改：

```text
/data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/cosmos_framework/model/generator/mot/unified_mot.py
```

在 `_impl_generate_reasoner_text()` 内包裹完整 decode：

```python
with runtime_region("reasoner_decode"):
    output_buf = ...
    for ...:
        # 保持现有 decode loop 不变
```

region 必须覆盖一次完整 decode，不能逐个包裹 `reasoner_forward()`。

现有 profiling wrapper 不适合直接复用：它会为每个 token 记录并同步 CUDA event，改变真实 decode 的执行行为。

## 7. Prefill 到 Decode 的 Stream 交接

prefill 在原 CUDA stream 上运行，decode 切换到 Reasoner Green Context stream。context manager 必须：

1. 在原 stream 上记录 `prefill_done`；
2. 让 Reasoner GC stream 等待 `prefill_done`；
3. 进入 `torch.cuda.stream(reasoner_gc_stream)`；
4. 等待 overlap barrier；
5. 执行原始 decode loop；
6. 在 Reasoner GC stream 上记录 `decode_done`；
7. 退出 GC stream context；
8. 让原 stream 等待 `decode_done`。

伪代码：

```python
class ReasonerDecodeRegion:
    def __enter__(self):
        self.original_stream = torch.cuda.current_stream()
        self.prefill_done = torch.cuda.Event()
        self.original_stream.record_event(self.prefill_done)
        self.reasoner_gc_stream.wait_event(self.prefill_done)

        self.stream_context = torch.cuda.stream(self.reasoner_gc_stream)
        self.stream_context.__enter__()
        self.start_barrier.wait()

    def __exit__(self, *exc):
        self.decode_done = torch.cuda.Event()
        self.reasoner_gc_stream.record_event(self.decode_done)
        self.stream_context.__exit__(*exc)
        self.original_stream.wait_event(self.decode_done)
```

不要用全局 `torch.cuda.synchronize()` 启动 overlap，它可能串行化两个 stream。stream 依赖使用 CUDA event，只在 trial/result 边界做必要同步。

## 8. Generator 执行

第一版不拆分 T2V generator，直接通过已加载的 `OmniInference` pipeline 运行完整路径：

```python
pipe.generate_batch(
    sample_args,
    fresh_data_batch,
    save_outputs=False,
)
```

generator worker 在 Generator GC stream 中调用：

```python
with torch.cuda.stream(generator_gc_stream):
    pipe.generate_batch(...)
```

每次实验前必须断言：

```python
assert pipe.vae_decode_stream is None
```

否则 VAE decode 可能切换到普通 CUDA stream，逃离 Generator Green Context。

每轮必须使用新构造的 data dictionary、latent 和 scheduler state，不能复用可能已被上一请求原地修改的对象。

## 9. 实验 Runner

新增：

```text
/data/cjh/cosmos3/cosmos3-workspace/cosmos-framework/profiling_tools/colocate_decode_t2v.py
```

runner 支持两种模式。

### 9.1 固定输入 Microbenchmark

每轮复用 `rich02` generator 输入，同时创建新的 Reasoner 请求和 KV cache。

用途：

- 验证线程和 stream；
- 快速枚举 GC 分区；
- 与已有 scaling 曲线直接比较；
- 减少内容导致的延迟波动。

固定 generator 输入代表 request `i`，活跃 Reasoner decode 代表 request `i+1`，并非同一请求内部共置。

### 9.2 满足依赖的 Streaming 模式

把 `decode_i` 的真实输出交给 `generator_i`，然后让该 generator 与 `decode_{i+1}` 共置。

用途：

- 最终运行时验证；
- 验证 prompt 解析与数据交接；
- 测量稳态吞吐；
- 检查共享模型/cache 是否线程安全。

两种模式共用 scheduler、GC 创建、stream 交接、计时和结果格式。

## 10. Warmup 与测量协议

在同一进程中测量 alone 和 colocated：

```text
Reasoner alone warmup              2 个请求
Reasoner alone measured            3 个请求
Generator alone warmup             2 个请求
Generator alone measured           3 个请求
Colocated pipeline warmup           3 个 overlap wave
Colocated pipeline measured        20 个 overlap wave
```

每轮 Reasoner 都使用新请求和新 KV cache；每轮 generator 都使用新的 request data 和 latent。模型权重、已编译 kernel、CUDA allocator 和 GC 对象保持常驻。

pipeline fill/drain 不计入 overlap latency，应单独报告；它们可以计入整批请求的端到端完成时间。

## 11. 初始 SM 分区

总 actual SM 为 170，先测试：

```text
Reasoner decode    T2V generator
---------------    -------------
42                 128
34                 136
28                 142
26                 144
```

必须记录 requested SM 和 actual SM，分析时使用 actual SM。

每个共置组合之前，先分别测量两个 workload 在相同分区下的 alone 时间。不能用 full-SM alone 时间代替 partitioned-alone 时间计算干扰。

## 12. 计时与指标

在各自 stream 上使用 CUDA event 记录：

- Reasoner decode start/end；
- 完整 generator start/end；
- 可选的 generator component start/end，但组件间不能同步；
- overlap wave wall-clock start/end。

每种分区至少报告：

```text
T_reasoner_full
T_generator_full

T_reasoner_partition_alone
T_generator_partition_alone

T_reasoner_colocated
T_generator_colocated
T_wave_colocated = max(T_reasoner_colocated, T_generator_colocated)

reasoner_interference = T_reasoner_colocated / T_reasoner_partition_alone
generator_interference = T_generator_colocated / T_generator_partition_alone

partitioned_serial = T_reasoner_partition_alone + T_generator_partition_alone
colocation_speedup = partitioned_serial / T_wave_colocated
```

runner 同时报告两个串行口径：`serial_gpu_sum` 是两个 partitioned-alone CUDA event 时间之和；`serial_host_wall` 是这两个 workload 在同一常驻进程中真实顺序执行的同步 wall-clock，包含 prefill、CPU preparation 和 launch overhead。对应地，`throughput_gain_gpu_sum` 用 GPU-only 求和作分子，`throughput_gain_host_wall` 用真实串行 wall 除以共置 wall。端到端结论优先使用后一项，同时保留前者便于分析纯 GPU 干扰。

另外报告含 fill/drain 的流水线端到端吞吐，但不能与单 wave speedup 混淆。

## 13. 正确性检查

接受计时结果前必须验证：

- 等价运行生成的 token 数一致；
- 确定性 Reasoner 的 alone/colocated token hash 一致；
- 每个 temporal caption 均能解析；
- 视频 tensor 不含 NaN/Inf；
- 视频 shape、dtype 和帧数一致；
- 记录 generator alone/colocated 输出 checksum 或统计值；
- 计时区域内没有工作意外进入旧 default stream；
- `vae_decode_stream is None`；
- actual GC SM 数符合记录；
- instrumentation 没有引入逐 token CUDA 同步。

并发可能改变合法的浮点归约顺序，所以视频不一定 bitwise 一致。如果不能逐位一致，应使用适当数值容差和输出统计比较，并明确记录。

## 14. 实现顺序

1. 增加 no-op runtime-region 机制和单元测试。
2. 用 `runtime_region("reasoner_decode")` 包裹完整官方 decode loop。
3. 不安装 runtime，运行普通 Reasoner，验证输出 token 不变。
4. 在普通 CUDA stream 上实现 decode 边界暂停/恢复。
5. 将 decode stream 替换为 Reasoner GC stream。
6. 使用完整现有路径搭建单进程 generator worker。
7. 分别在目标分区测量 Reasoner 和 generator alone。
8. 运行一个固定输入 overlap wave，验证 stream 和输出。
9. 重复固定输入 wave，枚举 SM 分区。
10. 加入满足依赖的真实 request handoff。
11. warmup 后运行 20 个 measured overlap wave。
12. 比较 partitioned-alone、colocated 和端到端流水线指标。

## 15. 第一版明确不做

- 不把 KV cache 序列化到磁盘；
- 不在伪造 module 中重建 decode；
- 不把 decode 拆成逐 token 任务；
- 不把 diffusion 拆成逐 step 任务；
- 不动态改变正在运行的 GC 大小；
- 不使用分别加载模型的多个子进程；
- 不复制模型权重；
- 不在 generator 计时区域中包含文件保存或 guardrail；
- 不跨 trial 复用已被修改的请求 tensor。

只有静态双分区 runtime 得到正确、稳定且可重复的结果后，才考虑这些扩展。

## 16. Reasoner decode CUDA Graph（实验路径）

逐 token eager decode 会为每层、每个 token 从 Python 发起大量 kernel。当前实现增加了一个默认关闭的 CUDA Graph 路径：

- prefill 仍走原路径；
- prefill 后把逐步 `torch.cat` 的 KV cache 转为固定容量 buffer；
- decode attention 使用 `flash_attn_with_kvcache` 原地写入 KV；
- 捕获一个 token 的完整 reasoner forward、LM head、argmax、EOS bookkeeping 和位置/KV 长度递增；
- 后续 token 只做 graph replay，并把 token 写入预分配输出 buffer；
- capture 和 replay 都发生在 `reasoner_decode` region 当前的 Green Context stream 上。

注意必须向 `torch.cuda.graph(..., stream=current_gc_stream)` 显式传入 GC stream。PyTorch 在省略 `stream=` 时不会沿用外层 current stream，而会切换到进程级默认 capture side stream；该 side stream 属于 primary context，会导致 NCU 显示 `Uses Green Context = 0`、`# SMs = 170`。2026-10-17 之前未显式传入 stream 的 CUDA Graph 测量因此不能作为 GC 共置结果，必须重测；eager GC 测量不受影响。

为了不静默改变采样语义，目前只支持单进程、greedy、无 repetition/presence penalty。CUDA Graph 与 fixed-length 已解耦：默认每次 replay 后检查 EOS 并精确提前结束；只有设置 `COSMOS_REASONER_FIXED_LENGTH_DECODE=1` 才固定运行到 `max_new_tokens`，用于可重复 profiling。普通 eager 路径不受影响。

启用方式：

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate

export LD_LIBRARY_PATH=''
export CUDA_DEVICE_MAX_CONNECTIONS=8
export COSMOS_REASONER_CUDA_GRAPH=1
export COSMOS_GENERATOR_SM_COUNT=136

# 可选：profiling 时固定运行到 max_new_tokens；普通推理不要设置
export COSMOS_REASONER_FIXED_LENGTH_DECODE=1

python -m profiling_tools.runners.colocate_decode_t2v
```

关闭 CUDA Graph、回到同一 fixed-length eager 基线：

```bash
unset COSMOS_REASONER_CUDA_GRAPH
export COSMOS_REASONER_FIXED_LENGTH_DECODE=1
python -m profiling_tools.runners.colocate_decode_t2v
```

2026-10-16 的首次验证（actual reasoner 24 SM，1350 tokens）：

```text
CUDA Graph reasoner decode: 7.450 s
T2V generator:             39.540 s
GPU makespan:              39.540 s
```

Graph 输出的 `reasoner_text.txt` 与已有 `rich02` eager 输出逐字节一致（SHA-256 均为 `139b3278b8fba01f2729d8d19b05d58f6136a09a186cb9954d6589eb6656ceaf`）。首次结果包含两个 warmup forward 和 capture 开销；正式测量仍应在同一常驻进程内区分 warmup 与 measured rounds。

为了让 measured request 只包含 replay，可显式启用跨请求 graph runtime：

```bash
export COSMOS_REASONER_CUDA_GRAPH_PERSISTENT=1
```

该 runtime 不是只缓存 `CUDAGraph`。graph 会绑定 tensor 地址，因此它同时持有固定地址的 KV、token、position、finished、pad/EOS buffer。每个新请求的 prefill 完成后，新 KV 前缀会复制进固定 KV buffer，并重置全部递归状态，再开始 replay。缓存键包含模型、GC stream、batch/position shape、prefill 长度和容量；形状或 stream 改变时会重新 capture。该选项仅用于常驻单进程 profiling，默认关闭。

runner 检测到该开关后，会先执行一次不计入结果的 reasoner-alone graph prime；其 NVTX range 为 `reasoner_decode_graph_prime`。这样 capture 不与 T2V 的另一个 CPU/CUDA 提交线程并发。后续 warmup、measured alone 和 colocated 请求均复用该 graph；measured alone 的独立 range 为 `reasoner_decode_alone`。

若只需在完全相同的 graph/fixed-length 输入下补测 full-SM decode，设置 `COSMOS_FULL_SM_DECODE_ONLY=1`。runner 会在一个普通的、非 default、未受 GC 限制的 primary-context stream 上先 prime，再测一次 warm full-SM decode 并退出，不加载第二套 GC graph。其 measured NVTX range 为 `reasoner_decode_full_sm_alone`。

正确性验证时设置 `COSMOS_REASONER_VALIDATE_TOKENS=1`。实现会在 decode 计时结束后把生成 token ids 复制到 CPU，并打印 shape、dtype 和 SHA-256；该 D2H 不计入 decode CUDA event。相同 fixed-length 输入下，eager/full、graph/full、graph/GC-alone 和 graph/GC-colocated 的 hash 必须完全一致。

只测受限分区上的 warm graph decode 时设置 `COSMOS_GC_DECODE_ONLY=1`；它会在 reasoner GC stream 上 prime、measured，然后打印 requested/actual SM 与秒数并退出。批量 scaling 使用 `profiling_tools/scripts/run_reasoner_graph_sm_scaling.sh`，默认测 8 到 160（步长 8）的 reasoner 目标值和一个 full-SM 参考点，日志写入 `profiling_results/reasoner_graph_sm_scaling/`。

正式 scaling 和共置默认保留 EOS early-exit，不设置 `COSMOS_REASONER_FIXED_LENGTH_DECODE`。批量脚本的默认输出目录为 `profiling_results/reasoner_graph_sm_scaling_early_exit/`。只有需要固定工作量的控制实验时才显式设置 `FIXED_LENGTH_DECODE=1`，其结果写入 `profiling_results/reasoner_graph_sm_scaling_fixed_length/`，不得与 early-exit 主结果混用。此前 `profiling_results/reasoner_graph_sm_scaling/` 下的结果均为 fixed-length 历史结果。

分区顺序以 T2V 为主：`COSMOS_GENERATOR_SM_COUNT` 指定 generator 的硬目标，Reasoner decode 使用 CUDA driver 返回的剩余 SM。例如 170-SM GPU 上指定 136，实际分区为 generator 136 + decode 34。不要再先请求较小的 reasoner 分区，否则 CUDA 的 SM 粒度向上量化可能反向挤占 T2V。

共置 capture 使用 `capture_error_mode="thread_local"`。默认的 `global` 模式会在另一个 CPU 线程提交 T2V CUDA 工作时令 reasoner capture invalidated。

修正后 136+34 的实测结果：

```text
CUDA Graph reasoner decode: 7.336 s
T2V generator:             40.495 s
GPU makespan:              40.495 s
```

### 16.1 Warm 配平与干扰测试

runner 默认在同一个常驻进程中执行：1 个 `2×decode + 1×T2V` warmup、`2×decode` partitioned-alone、`1×T2V` partitioned-alone、最后一个 `2×decode + 1×T2V` measured colocation。两个 decode 是两个独立的 1350-token 请求，不共享增长后的 KV 序列。最终自动报告双方 slowdown、串行时间、共置 makespan 和 throughput gain。

```bash
export COSMOS_REASONER_CUDA_GRAPH=1
export COSMOS_REASONER_CUDA_GRAPH_PERSISTENT=1
export COSMOS_REASONER_FIXED_LENGTH_DECODE=1
export COSMOS_GENERATOR_SM_COUNT=136
export COSMOS_REASONER_REPEATS=2
export COSMOS_WARMUP_RUNS=1
python -m profiling_tools.runners.colocate_decode_t2v
```
