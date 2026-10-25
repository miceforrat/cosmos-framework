# Gate 1：Edge Reasoner 与 Generator 同 PID 串行验证

本阶段只验证两套引擎能够在一个 Python PID / CUDA primary context 内共同驻留并串行推理。
明确不测试并发、CUDA Graph 或 Green Context。

```bash
cd /workspace/cosmos-framework
source .runtime/vllm_omni_tf517/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=''

python -m profiling_tools.runners.single_process_dual_engine \
  --order reasoner-first \
  --output profiling_results/gate_1_single_process/reasoner_first.json

python -m profiling_tools.runners.single_process_dual_engine \
  --order generator-first \
  --output profiling_results/gate_1_single_process/generator_first.json
```

每次运行前 GPU 必须空闲。两个顺序必须作为两个独立 Python 进程测试，避免前一个顺序留下的
allocator、全局 registry 或编译缓存污染另一个顺序。

Reasoner 固定 eager、1 GiB KV、`gpu_memory_utilization=0.70`；Generator 固定 eager，并使用
最小 256×256、单帧、2-step 请求。通过后再把 Generator 请求提升到代表性 T2V。

这里的 `gpu_memory_utilization=0.70` 不是把 KV cache 扩到 GPU 的 70%。本实验已经通过
`kv_cache_memory_bytes=1 GiB` 显式限制 KV。vLLM 0.31.0 在采用该显式限制之前，仍会先执行
基于 `gpu_memory_utilization` 的启动准入检查。Generator-first 实测 Generator 驻留后剩余
23.59/31.36 GiB（约 75.2%）；默认值 0.92 要求空闲 28.85 GiB，因而在真正分配 Reasoner
前报错。将准入阈值设为 0.70 后仍保留约 1.6 GiB 检查余量，而实际 KV 上限继续由 1 GiB
参数控制。

Reasoner 离线输入必须先经过 checkpoint tokenizer 的 chat template，并带
`add_generation_prompt=True`。直接传裸文本会立即生成 EOS；这与 server 的
`/v1/chat/completions` 路径不同，因为 server 会自动应用模板。

## 单 rank parallel-state 兼容层

Reasoner 先构造时，vLLM 已占用进程全局 TP/DP/PP 等 coordinator；原始 Omni 会因此拒绝
初始化。`profiling_tools/core/omni_shared_parallel.py` 在 world size 1 且所有并行度为 1 时：

- 保留并复用 Reasoner 的 vLLM TP/DP/PP groups；
- 只创建 Generator 自己的 diffusion DP/CFG/PP/SP groups；
- Generator 关闭时不销毁 Reasoner 拥有的 groups；
- 对多卡、EP、HSDP 或非 1 并行度直接报错。

反向构造时 Omni 已建立 TP/DP/PP，但没有 vLLM Reasoner 要求的 PCP/DCP/ETP。
在单 rank 限制下，兼容层将这三个缺失槽临时映射到 Omni 建立的 vLLM TP coordinator。
关闭时先关闭借用者 Reasoner，再清除三个别名，最后由所有者 Generator 销毁 groups，
避免同一 coordinator 被重复销毁。

## 2026-10-20 `reasoner-first` 实测

`reasoner_first_v2.json` 的五项检查全部通过：

- 两个 Engine 的 GPU 工作均由 PID 874379 发起，没有额外 GPU Worker；
- 主线程 CUDA context 在两次构造及两次推理后均为 `0x1608f880`；
- 双 Engine ready 驻留为 15,784 MiB；
- Reasoner 生成 32 tokens、非空文本，Generator 返回 1 个完成的视频输出；
- 进程退出后 GPU 占用归零。

因此 Gate 1 的 `reasoner-first` 构造顺序通过。该结论仍然只覆盖串行 eager
推理与最小 Generator 请求，不代表已经存在 kernel overlap 或 GC 隔离。

## Generator-first 重测

修正上述 vLLM 启动准入阈值后，使用独立进程重测：

```bash
cd /workspace/cosmos-framework
source .runtime/vllm_omni_tf517/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=''

python -m profiling_tools.runners.single_process_dual_engine \
  --order generator-first \
  --output profiling_results/gate_1_single_process/generator_first_v4.json \
  2>&1 | tee profiling_results/gate_1_single_process/generator_first_v4.log
```

`generator_first_v3.json` 已证明两套 Engine 的构造和串行推理成功，但原清理顺序先关闭
Reasoner。由于该顺序下进程级 distributed environment 由 Omni 创建并拥有，Reasoner 的
shutdown 会提前销毁 Omni `_WORLD` 引用的 process group，随后 `generator.close()` 再次
销毁时产生 `ValueError: Invalid process group specified`。修正后的顺序为：先清除 Reasoner
借用的 PCP/DCP/ETP aliases，再关闭所有者 Generator，最后释放 Reasoner。

## Gate 2 前的代表性串行时间基线

使用 NVIDIA Hugging Face `Cosmos3-Super-Text2Image/agentic_upsampling/prompt_upsampler.py`
中的官方 T2I JSON upsampler 模板，替换此前为拉长 decode 人工编写的 rich T2V prompt。
模板快照保存在 `profiling_tools/core/official_upsampler_workload.py`。Reasoner 输出上限仍设为
1350 tokens；Generator 使用 832×480、121 帧、35-step T2V。两阶段在双 Engine 同时驻留时
严格串行运行，所得时间作为 Gate 2 的部署形态基线：

```bash
cd /workspace/cosmos-framework
source .runtime/vllm_omni_tf517/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=''

python -m profiling_tools.runners.single_process_dual_engine \
  --order generator-first \
  --representative-workloads \
  --reasoner-max-tokens 1350 \
  --output profiling_results/gate_1_single_process/representative_serial_v1.json \
  2>&1 | tee profiling_results/gate_1_single_process/representative_serial_v1.log
```

JSON 中 `reasoner.seconds` 和 `generator.seconds` 分别为两个阶段的实测 wall time；
`reasoner.decode_tokens_per_second` 是包含该次 `generate()` 调用完整时间的端到端生成吞吐，
不是剔除 prefill 后的纯 decode TPOT。Gate 2 若需要纯 prefill/decode 分解，应增加 NVTX/engine
级计时，不能从该字段反推。

### Reasoner CUDA Graph 调用

`--reasoner-mode eager` 保留 Gate 1 的原始 eager 路径；`--reasoner-mode graph` 使用 vLLM
`FULL_AND_PIECEWISE` CUDA Graph。Omni Generator 暂时继续 eager，避免同时改变两个变量：

```bash
python -m profiling_tools.runners.single_process_dual_engine \
  --order generator-first \
  --representative-workloads \
  --reasoner-mode graph \
  --generator-cache on \
  --reasoner-max-tokens 1350 \
  --output profiling_results/gate_1_single_process/representative_serial_graph_v1.json \
  2>&1 | tee profiling_results/gate_1_single_process/representative_serial_graph_v1.log
```

验收时日志必须显示 CUDA Graph capture/启用信息，且不能出现
`Cudagraph is disabled under eager mode`。性能结论还需至少一次不计入统计的同负载 warmup；
上面这条单次命令目前用于功能和显存验证。

### 当前 Omni 优化边界

- Cosmos3 diffusion 支持 regional/full `torch.compile`；它为 lazy compile，首请求不能直接作为
  steady-state 时间。后续应增加 warmup/repeat 后再接入。
- `enable_cuda_graph_decode` 在当前源码中明确服务于 Qwen-Image-2.1 固定形状 decode，不能
  当作 Cosmos3 DiT CUDA Graph 开关。
- Cosmos3 已接入 Cache-DiT、TeaCache 和 SeaCache，但这些会改变 diffusion 执行/复用策略，
  应作为近似推理实验单列，不能放进精确 eager/compile 基线。
- Wan VAE lossless fast path 已默认启用；tiling/slicing 主要降低峰值显存，通常不是延迟优化。
- prompt embedding cache 只对重复 prompt 有收益，不应影响每次不同 prompt 的 Gate 2 基线。

### Cosmos3 SeaCache 对齐

旧 Cosmos Framework `rich02` Generator 并非 cache-off。其日志明确记录：

```text
diffusion_cache_thresh=0.25
residual_order=1
max_consecutive_cached=2
power_exp=3.0
full=36 skipped=34
```

vLLM-Omni 的 Cosmos3 专用 `sea_cache` 后端提供完全同名语义的四个默认值。runner 现在通过
`--generator-cache on|off` 显式选择；`on` 会传入上述四项参数。为避免悄悄改变已经完成的
Gate 1 smoke test，CLI 默认仍为 `off`；但后续代表性基线与 Gate 2 主实验必须显式使用：

```bash
--generator-cache on
```

此前 `representative_serial_v1.json` 的 Omni Generator 时间 32.324 s 是 cache-off，不能与
旧 `rich02` cache-on 时间直接比较，也不能作为后续 cache-on 共置实验的串行基线。

### Generator-first CUDA Graph coordinator 兼容

第一次 `graph + cache-on` 验证在 Reasoner capture 阶段失败：Omni 创建并借给 vLLM 的
单-rank `GroupCoordinator` 没有 vLLM 0.31 要求的 `graph_capture(context)` 方法。compile
本身已经成功，SeaCache 也已安装；失败与 cache 无关。

`profiling_tools/core/omni_shared_parallel.py` 现为借用的 TP/DP/PP coordinator 补充仅限
world-size=1 的 capture-stream context manager。它等待原 current stream 后进入 vLLM 传入
的共享 capture stream；由于 rank=1 不存在 NCCL/custom-all-reduce capture，不执行通信侧
操作。多 rank 仍直接拒绝，不将这一兼容逻辑推广到未经验证的配置。

### Generator torch.compile 冷热分离

`--generator-mode compile` 令 Omni diffusion 退出 `enforce_eager`，进入其
`torch.compile` 路径。Reasoner 的 `--reasoner-mode graph` 已同时包含 vLLM compile 与
CUDA Graph；这两个开关对应不同模型路径，不能混为一项。

以下命令执行一次完整 Generator warmup，并仅用随后三次请求计算稳态中位数。warmup 的
完整时间仍保存在 JSON 的 `generator.warmups`，不会丢掉基模的首次 shape 编译/JIT 开销：

```bash
cd /workspace/cosmos-framework
source .runtime/vllm_omni_tf517/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=''

python -m profiling_tools.runners.single_process_dual_engine \
  --order generator-first \
  --representative-workloads \
  --reasoner-mode graph \
  --generator-mode compile \
  --generator-cache on \
  --generator-warmups 1 \
  --generator-repeats 3 \
  --reasoner-max-tokens 1350 \
  --output profiling_results/gate_1_single_process/representative_serial_both_compile_cache_on_v1.json \
  2>&1 | tee profiling_results/gate_1_single_process/representative_serial_both_compile_cache_on_v1.log
```

核验时必须同时查看：Generator warmup 时间、三次 measured 时间、日志中的 compile/JIT
证据以及是否出现 graph break。若首次 measured 仍出现新 shape 的 JIT，应增加 warmup，
不能把该轮当作稳态。

### 2600-token 代表性负载结果

本轮统一负载配置：

```text
Reasoner max_model_len: 6144
Reasoner max output: 2600 tokens
Reasoner actual output: eager 2248 tokens; graph 2266 tokens
Generator resolution: 832 x 480 (W x H)
Generator frames: 121
Generator diffusion steps: 35
Generator guidance scale: 6.0
Generator cache: SeaCache on
KV cache allocation: 1 GiB, 9360-token capacity
```

稳态中位数（两边均为 1 次 warmup、3 次 measured）：

| Reasoner | Generator | Reasoner | Generator | 串行总时间 |
|---|---|---:|---:|---:|
| eager | eager | 39.413 s | 18.253 s | 57.666 s |
| graph | eager | 7.550 s | 18.354 s | 25.904 s |
| eager | compile | 38.469 s | 16.494 s | 54.963 s |
| graph | compile | 7.513 s | 16.572 s | 24.084 s |

全优化相对全 eager 的串行稳态加速为 2.39x。四组峰值显存约 21.2--21.7 GiB，
未发生 OOM；6144-token 单请求对应 KV 最大并发为 1.52x。

四组 Reasoner 均以 `finish_reason=stop` 自然结束，嵌入输出可被标准 JSON 解析。但输出没有
严格满足官方 schema：模型把 `aspect_ratio` 放入了 `resolution` 对象，遗漏顶层
`aspect_ratio`；部分输出的 H/W 与比例还不一致。因此该输出可用于性能负载，不应在没有
校验和覆盖 resolution/aspect_ratio 的情况下直接作为生产 upsampler 结果。
