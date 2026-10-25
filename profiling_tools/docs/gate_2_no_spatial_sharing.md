# Gate 2：无空间切分的单进程并发

本阶段不使用 Green Context、MPS 或任何 SM 配额。目标是在 Gate 1 的同 PID/同 CUDA context 双模型
基础上，用两个 host worker thread 和两条普通 CUDA stream 同时调用 Reasoner 与 Generator。
这最多证明普通 CUDA 调度下的并发/time multiplexing；只有 Nsys kernel timeline 能证明实际 GPU overlap。

## 固定负载

- Reasoner：官方 prompt upsampler，6144 context；benchmark 默认固定生成 2600 tokens（`seed=0`、`min_tokens=max_tokens`、`ignore_eos=True`）；
- Generator：832x480、121 frames、35 steps、CFG 6.0、SeaCache on；
- 构造顺序：Generator first；
- 串行 warmup 后，先测两边 alone，再测 concurrent；
- concurrent 使用 `ThreadPoolExecutor(max_workers=2)`、三方 barrier、两条不同普通 CUDA stream；
- NVTX ranges：`gate2.reasoner.*` 与 `gate2.generator.*`。

输出中的 `host_overlap_seconds` 只表示两个同步 Python 调用的时间区间重合，不等于 GPU kernel overlap。
`gpu_kernel_overlap_verified` 在没有 Nsys 证据前固定为 false。

## 1. Eager 快速归因

```bash
cd /workspace/cosmos-framework
source .runtime/vllm_omni_tf517/venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=''

python -m profiling_tools.runners.concurrent_reasoner_generator \
  --reasoner-mode eager \
  --generator-mode eager \
  --generator-cache on \
  --reasoner-max-tokens 2600 \
  --reasoner-fixed-tokens \
  --warmups 1 \
  --alone-repeats 1 \
  --concurrent-repeats 1 \
  --output profiling_results/gate_2_no_spatial/eager_eager_fixed_quick_v3.json \
  2>&1 | tee profiling_results/gate_2_no_spatial/eager_eager_fixed_quick_v3.log
```

## 2. 优化配置快速验证

注意：Reasoner 的 graph replay 必须保留 vLLM 自己管理的 stream/event 契约，runner 不再用外部 `torch.cuda.Stream` 包裹 graph 模式；eager 模式仍使用独立外部 stream。Generator 这里的优化模式是 `torch.compile`，不是显式 CUDA Graph。

```bash
python -m profiling_tools.runners.concurrent_reasoner_generator \
  --reasoner-mode graph \
  --generator-mode compile \
  --generator-cache on \
  --reasoner-max-tokens 2600 \
  --reasoner-fixed-tokens \
  --warmups 1 \
  --alone-repeats 1 \
  --concurrent-repeats 1 \
  --output profiling_results/gate_2_no_spatial/graph_compile_fixed_quick_v3.json \
  2>&1 | tee profiling_results/gate_2_no_spatial/graph_compile_fixed_quick_v3.log
```

runner 会在计算 slowdown 前核对 warmup、alone、concurrent 的 `completion_tokens`；任一阶段不是 2600 都会直接报错。需要自然 EOS 时可显式传 `--no-reasoner-fixed-tokens`，但该模式不得用于本轮性能对照。

## 3. 固定负载 2×2 消融

```bash
for pair in eager:eager eager:compile graph:eager graph:compile; do
  reasoner_mode=${pair%%:*}
  generator_mode=${pair##*:}
  name=${reasoner_mode}_${generator_mode}

  python -m profiling_tools.runners.concurrent_reasoner_generator \
    --reasoner-mode "$reasoner_mode" \
    --generator-mode "$generator_mode" \
    --generator-cache on \
    --reasoner-max-tokens 2600 \
    --reasoner-fixed-tokens \
    --warmups 2 \
    --alone-repeats 3 \
    --concurrent-repeats 3 \
    --output "profiling_results/gate_2_no_spatial/${name}_fixed_full_v3.json" \
    2>&1 | tee "profiling_results/gate_2_no_spatial/${name}_fixed_full_v3.log"
done
```

快速验证通过后，把 alone/concurrent repeats 都提高到 3。吞吐收益严格使用同一进程内测得的：

```text
(serial reasoner host wall + serial generator host wall) / concurrent host wall
```

并分别报告两边 slowdown，不能用两个 colocated 时间之和或 `max()` 代替实测 wall time。

## 4. Nsys overlap 证据

先确保 GPU 空闲，然后直接 profile 一轮 quick workload：

```bash
mkdir -p profiling_results/gate_2_no_spatial/nsys

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output profiling_results/gate_2_no_spatial/nsys/eager_eager_fixed_v3 \
  python -m profiling_tools.runners.concurrent_reasoner_generator \
    --reasoner-mode eager \
    --generator-mode eager \
    --generator-cache on \
    --reasoner-max-tokens 2600 \
    --reasoner-fixed-tokens \
    --warmups 1 \
    --alone-repeats 1 \
    --concurrent-repeats 1 \
    --output profiling_results/gate_2_no_spatial/eager_eager_fixed_nsys_v3.json
```


重点检查 graph replay 是否被 Generator 串行阻塞时，再采集 graph+compile：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output profiling_results/gate_2_no_spatial/nsys/graph_compile_fixed_v3 \
  python -m profiling_tools.runners.concurrent_reasoner_generator \
    --reasoner-mode graph \
    --generator-mode compile \
    --generator-cache on \
    --reasoner-max-tokens 2600 \
    --reasoner-fixed-tokens \
    --warmups 1 \
    --alone-repeats 1 \
    --concurrent-repeats 1 \
    --output profiling_results/gate_2_no_spatial/graph_compile_fixed_nsys_v3.json
```
验收时必须在 `gate2.reasoner.iter0` 与 `gate2.generator.iter0` 的共同时间窗内检查 CUDA kernel，确认：

1. 两类 kernel 是否来自不同 stream；
2. 是否存在时间轴上真实重叠，而不只是 CUDA API 调用重叠；
3. 是否有 `cudaDeviceSynchronize`、event wait 或默认 stream 依赖将两边串行化；
4. graph 配置下的 replay 是否仍与 diffusion kernel overlap。

## 当前边界

- 没有空间隔离，两边竞争全部 GPU 资源；
- 外层 stream context 不保证框架内部 stream 或 CUDA Graph node 遵循该 stream；
- Engine/scheduler 是否线程安全由本 Gate 实测验证；
- 当前两条负载数据独立，不存在请求 i 的 Reasoner 到请求 i 的 Generator 数据依赖；可视为流水线中
  不同请求的两个阶段并发。
