# Cosmos3 Nsys kernel discovery and NCU Roofline workflow

This document describes how to:

1. Use Nsight Systems and the existing component NVTX ranges to find representative kernels.
2. Select kernels by GPU-time coverage and computational role.
3. Profile those kernels on the full 170-SM RTX 5090 with Nsight Compute.
4. Export Roofline data without opening a graphical interface.

Roofline is fundamentally a kernel-level analysis. A component such as `diffusion_sampling` contains many kernels and therefore does not correspond to one Roofline point.

## 1. Components currently instrumented

Generation tasks:

```text
cosmos.component.condition_prepare
cosmos.component.text_reasoner
cosmos.component.diffusion_sampling
cosmos.component.vae_decode
cosmos.component.action_postprocess
```

Video-to-Text:

```text
cosmos.component.vision_encoder
cosmos.component.reasoner_prefill
cosmos.component.reasoner_decode
cosmos.component.reasoner_e2e
```

`reasoner_e2e` contains the other Video-to-Text GPU components and must not be added to them when calculating a breakdown.

## 2. Install Nsys and NCU in the container

Enter the container from the host:

```bash
docker exec -it cosmos3-clean bash
```

The container is Ubuntu 22.04 and already has the NVIDIA CUDA apt repository configured. Install the CUDA 12.8-aligned packages:

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  cuda-nsight-systems-12-8 \
  cuda-nsight-compute-12-8
```

The CUDA 12.8 meta-packages currently resolve to versions compatible with the existing environment:

```text
cuda-nsight-systems-12-8 -> nsight-systems-2024.6.2 or newer compatible revision
cuda-nsight-compute-12-8 -> nsight-compute-2025.1.1 or newer compatible revision
```

NCU 2025.1.1 includes the GB20x Roofline ceiling fix needed for the RTX 5090. Do not install the old generic `nsight-systems` package from Ubuntu's base repository.

Initialize the project environment and verify the tools:

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate

nsys --version
ncu --version

mkdir -p profiling_results/kernel_discovery
mkdir -p profiling_results/roofline
mkdir -p outputs/kernel_discovery
mkdir -p outputs/roofline
```

The packages are installed into the current container. If the container is recreated, repeat the installation or add the packages to the image's Dockerfile.

## 3. Capture Nsys traces for the four tasks

Use unique output directories. Cosmos may skip an inference request when its output already exists.

### T2I

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o profiling_results/kernel_discovery/t2i \
  python -m profiling_tools.runners.run_profiled \
    --parallelism-preset=latency \
    --no-guardrails \
    -i inputs/omni/t2i.json \
    -o outputs/kernel_discovery/t2i \
    --checkpoint-path Cosmos3-Edge \
    --seed=0
```

### T2V

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o profiling_results/kernel_discovery/t2v \
  python -m profiling_tools.runners.run_profiled \
    --parallelism-preset=latency \
    --no-guardrails \
    -i inputs/omni/t2v.json \
    -o outputs/kernel_discovery/t2v \
    --checkpoint-path Cosmos3-Edge \
    --seed=0
```

### Action Policy

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o profiling_results/kernel_discovery/action \
  python -m profiling_tools.runners.run_profiled \
    --parallelism-preset=latency \
    --no-guardrails \
    -i inputs/omni/action_policy_robot.json \
    -o outputs/kernel_discovery/action \
    --checkpoint-path Cosmos3-Edge \
    --seed=0
```

### Video-to-Text

```bash
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  -o profiling_results/kernel_discovery/reasoner_video \
  python -m profiling_tools.runners.run_profiled \
    --parallelism-preset=latency \
    --no-guardrails \
    -i inputs/reasoner/reasoner_video.json \
    -o outputs/kernel_discovery/reasoner_video \
    --checkpoint-path Cosmos3-Edge \
    --seed=0
```

## 4. Export NVTX-to-kernel summaries

Example for T2V:

```bash
nsys stats \
  --report nvtx_kern_sum \
  --format csv \
  profiling_results/kernel_discovery/t2v.nsys-rep \
  > profiling_results/kernel_discovery/t2v_nvtx_kernels.csv
```

Export all four:

```bash
for task in t2i t2v action reasoner_video; do
  nsys stats \
    --report nvtx_kern_sum \
    --format csv \
    "profiling_results/kernel_discovery/${task}.nsys-rep" \
    > "profiling_results/kernel_discovery/${task}_nvtx_kernels.csv"
done
```

Do not add `:base`: it groups templated kernels by shortened base name. The
default report preserves full demangled kernel names for precise NCU filtering.

Inspect a CSV:

```bash
column -s, -t profiling_results/kernel_discovery/t2v_nvtx_kernels.csv | less -S
```

Search a component:

```bash
grep 'cosmos.component.diffusion_sampling' \
  profiling_results/kernel_discovery/t2v_nvtx_kernels.csv \
  | less -S
```

## Decode Green Context：alone GEMV 的调度/等待基线

NCU 2025.3.1 下用内部 decode NVTX range 过滤，避免误抓 prefill。`/` 表示 NVTX push/pop range。persistent graph 会先以独立的 `reasoner_decode_graph_prime` range 串行 capture；后续 `reasoner_decode_alone` 只 replay 同一张图。NCU 仍可能将 graph node 展开为 kernel 进行采集。

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate

export LD_LIBRARY_PATH=''
export CUDA_DEVICE_MAX_CONNECTIONS=8
export COSMOS_REASONER_CUDA_GRAPH=1
export COSMOS_REASONER_CUDA_GRAPH_PERSISTENT=1
export COSMOS_REASONER_FIXED_LENGTH_DECODE=1
export COSMOS_GENERATOR_SM_COUNT=136
export COSMOS_REASONER_REPEATS=1
export COSMOS_WARMUP_RUNS=1
export COSMOS_COLOCATE_GENERATOR_REGION=diffusion_sampling

NCU=/opt/nvidia/nsight-compute/2025.3.1/ncu

"$NCU" \
  --target-processes all \
  --graph-profiling node \
  --replay-mode kernel \
  --clock-control none \
  --nvtx \
  --nvtx-include 'reasoner_decode_alone/' \
  --kernel-name-base demangled \
  --kernel-name 'regex:internal::gemvx::kernel' \
  --launch-count 1 \
  --kill 1 \
  --section LaunchStats \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --print-metric-attribution \
  --force-overwrite \
  -o profiling_results/ncu/decode_t2v/decode_alone_g136_gemv_full \
  python -m profiling_tools.runners.colocate_decode_t2v
```

检查报告：

```bash
"$NCU" --import \
  profiling_results/ncu/decode_t2v/decode_alone_g136_gemv_full.ncu-rep \
  --page details
```

必须确认 `Uses Green Context = 1`、`# SMs = 34`。若命中的是首次 capture/materialization，而不是 measured replay，先保留该报告作为 kernel 基线，再用 Nsight Compute UI/CLI 的 graph-node 标识核对目标 launch；不要把 prefill 的 170-SM GEMV 当成 decode 数据。

如果报告显示 `Uses Green Context = 0`、`# SMs = 170`，说明 graph capture 落到了 PyTorch 默认 side stream，结果无效。实现必须显式使用 `torch.cuda.graph(..., stream=current_gc_stream)`，修复后重新运行功能验证和本节 NCU 命令。

## 5. Select kernels by cumulative GPU time

For normal runs, use the parameterized script; `--task` derives both the input and output paths:

```bash
python -m profiling_tools.analysis.select_kernels \
  --task t2v \
  --component diffusion_sampling
```

For this command, the paths are derived as:

```text
input:  profiling_results/kernel_discovery/t2v_nvtx_kernels.csv
output: profiling_results/kernel_discovery/t2v_diffusion_sampling_selected_kernels.csv
```

Change only `--task`, `--component`, or optional `--coverage`. Use `--input` and
`--output` only for nonstandard paths; run `python -m profiling_tools.analysis.select_kernels --help`
for the full interface.

To select kernels automatically for every component present in all four task CSVs:

```bash
python -m profiling_tools.analysis.select_all_kernels
```

The default coverage is 90%. Results are written to:

```text
profiling_results/kernel_selection/<task>_<component>_selected_kernels.csv
profiling_results/kernel_selection/all_selected_kernels.csv
```

The script discovers full NVTX names such as `cosmos.component.reasoner_prefill`
directly from each CSV instead of relying on abbreviated component names.

## 6. Representative-kernel selection rules

Do not select only the longest kernel. For each component, use the following rules:

1. Start with kernels covering 80% to 90% of cumulative component GPU time.
2. Keep at least one representative from each major computational category that occurs in the component.
3. If one kernel name appears with substantially different grid or tensor sizes, profile representative large and small invocations separately.
4. Prefer a median-duration invocation over the first invocation, because the first call may include warmup effects.

Typical categories:

```text
Diffusion:
  GEMM/Tensor Core
  Attention
  MoE/MLP
  Normalization
  Elementwise/cache

VAE:
  Convolution
  GEMM
  Normalization
  Upsampling/elementwise

Reasoner Prefill:
  Prompt-scale GEMM
  Prefill attention
  MLP/MoE
  Normalization

Reasoner Decode:
  Single-token GEMM/GEMV
  Decode attention
  KV-cache operations
  Normalization/logit processing

Vision Encoder:
  Vision attention
  GEMM
  Patch/projector operations
  Normalization
```

Three to eight representative kernel types per component is a practical starting point.

## 7. NCU Roofline collection example

First validate a broad kernel-name fragment with one invocation. This example assumes the selected kernel name contains `cutlass`:

```bash
ncu \
  --set roofline \
  --replay-mode kernel \
  --nvtx \
  --nvtx-include "cosmos.component.diffusion_sampling/" \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*cutlass.*' \
  --launch-count 1 \
  --target-processes all \
  --force-overwrite \
  --export profiling_results/roofline/t2v_diffusion_cutlass \
  python -m profiling_tools.runners.run_profiled \
    --parallelism-preset=latency \
    --no-guardrails \
    -i inputs/omni/t2v.json \
    -o outputs/roofline/t2v_diffusion_cutlass \
    --checkpoint-path Cosmos3-Edge \
    --seed=0
```

After validating the filter, collect up to three invocations:

```text
--launch-count 3
```

Use a unique inference output directory for every NCU run. If the output already exists, Cosmos may skip the request and NCU will capture no target kernel.

### Batch NCU collection and weighted Roofline plots

```bash
python -m profiling_tools.analysis.run_ncu_batch --dry-run --limit 3
python -m profiling_tools.analysis.run_ncu_batch
```

The command resumes completed reports automatically. Use `--tasks t2v` to run one task first.
After all reports complete, render the overall three-panel plot and separate backups:

```bash
python -m profiling_tools.analysis.plot_roofline
```

Outputs are under `profiling_results/roofline/plots/`. Point weight is kernel component
share times component task share; tasks use color families and components use shades.

## 8. Avoid profiling every repeated kernel

Do not initially run NCU Roofline over every kernel in `diffusion_sampling` or every token in `reasoner_decode`. NCU uses multiple replay passes for metric collection. Profiling all repeated launches can turn a seconds-long inference into a many-hour experiment and produce a very large report.

Recommended sequence:

```text
Nsys full task trace
  -> NVTX component kernel summary
  -> 80%-90% coverage list
  -> choose 3-8 representative kernel types
  -> NCU launch-count=1 validation
  -> NCU launch-count=3 final collection
```

## 9. Export NCU results without the GUI

The `.ncu-rep` report is produced by the `--export` option. Export its raw metrics as CSV:

```bash
ncu \
  --import profiling_results/roofline/t2v_diffusion_cutlass.ncu-rep \
  --page raw \
  --csv \
  > profiling_results/roofline/t2v_diffusion_cutlass.csv
```

Nsight Compute CLI does not directly render the interactive GUI Roofline chart to PNG. A headless workflow is still possible:

```text
.ncu-rep
  -> ncu --page raw --csv, or ncu_report.py
  -> extract arithmetic intensity, achieved FLOP/s and ceilings
  -> Matplotlib PNG/PDF
```

Locate the installed Python Report Interface with:

```bash
dpkg -L nsight-compute-2025.1.1 | grep '/extras/python/ncu_report.py$'
```

If the CUDA repository resolves the meta-package to a newer revision, replace the version in the `dpkg -L` command with the value shown by:

```bash
dpkg -l | grep nsight-compute
```

## 10. Common failures

### No kernels were profiled

Check that:

- The inference output directory did not already exist.
- The NVTX component name exactly matches the wrapper range.
- The kernel regex matches the demangled kernel name.
- `--kernel-name-base demangled` is present when using a demangled name.

### ERR_NVGPUCTRPERM

The NVIDIA driver is blocking access to GPU performance counters. The container must have access to the counters, and the host driver may need profiling permissions enabled by the administrator.

### Collection is extremely slow

Reduce the matched launches:

```text
--launch-count 1
```

Make the kernel regex more specific, and avoid profiling an entire repeated Diffusion or Decode component in the first run.

## 11. Final batch commands and weighted plots

Enter the running container from the host:

```bash
docker exec -it cosmos3-clean bash
```

Initialize the project environment and select kernels at 85% cumulative component time:

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate

python -m profiling_tools.analysis.select_all_kernels \
  --coverage 0.85
```

Preview the first three pending NCU commands without running them:

```bash
python -m profiling_tools.analysis.run_ncu_batch \
  --dry-run \
  --limit 3
```

Optionally validate one task first:

```bash
python -m profiling_tools.analysis.run_ncu_batch \
  --tasks t2v
```

Run or resume the complete batch:

```bash
python -m profiling_tools.analysis.run_ncu_batch
```

Completed reports are skipped automatically. After all reports finish, render the
overall three-panel Roofline and the separate L1, L2, and DRAM figures:

```bash
python -m profiling_tools.analysis.plot_roofline
```

The final outputs are:

```text
profiling_results/roofline/plots/roofline_overall.png
profiling_results/roofline/plots/roofline_overall.pdf
profiling_results/roofline/plots/roofline_l1.png
profiling_results/roofline/plots/roofline_l2.png
profiling_results/roofline/plots/roofline_dram.png
profiling_results/roofline/plots/roofline_points.csv
```


- Tasks use separate color families; Video Understanding (`reasoner_video`) uses red.
- Marker shape and area encode kernel importance; the single-column right-side legend is grouped by task and component, with kernels ordered by descending weight inside each component.
- Importance is `kernel share within component * component share within task`.

- Legends use a readable enlarged font even in the full 54-kernel figure.
- Each panel derives logarithmic axis limits from its plotted kernels, avoiding large empty low-performance regions while retaining the relevant hardware ceiling.
The plotting command also emits precision-separated figures, each with its own
compute ceiling:

```text
profiling_results/roofline/plots/roofline_bf16_tensor_overall.png
profiling_results/roofline/plots/roofline_bf16_tensor_{l1,l2,dram}.png
profiling_results/roofline/plots/roofline_scalar_fp16_overall.png
profiling_results/roofline/plots/roofline_scalar_fp16_{l1,l2,dram}.png
profiling_results/roofline/plots/roofline_scalar_fp32_overall.png
profiling_results/roofline/plots/roofline_scalar_fp32_{l1,l2,dram}.png
```


Pure copy kernels with zero counted floating-point or Tensor operations are omitted from the plotted points.
