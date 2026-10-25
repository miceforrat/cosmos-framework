# Cosmos3 component and Green Context profiling

Profiling 代码按职责放在 `core/`、`runners/`、`analysis/` 和 `scripts/`；
说明文档统一放在 `profiling_tools/docs/`。

```text
profiling_tools/
├── core/       # runtime instrumentation 与 Green Context
├── runners/    # 模型执行和共置实验入口
├── analysis/   # 汇总、绘图、kernel 选择与 NCU 批处理
├── scripts/    # 可复现的批量实验 shell 入口
└── docs/       # 当前文档与 archive 历史设计
```

## Documentation index

- 当前综合实验结论：`docs/cosmos3_architecture_decode_colocation_study.md`
- vLLM Reasoner + Generator 统一调度新方案：`docs/unified_vllm_generator_colocation_plan.md`
- Gate 3 GC stream 绑定证据与改造面评估：`docs/gate_3_green_context_stream_binding.md`
- Reasoner-only vLLM 阶段 1 验证：`docs/vllm_reasoner_stage1.md`
- Reasoner decode/GC 详细实现：`docs/cosmos3_reasoner_decode_colocation.md`
- cold/warm SM scaling：`docs/cold_warm_sm_scaling.md`
- Nsys/NCU 与 Roofline：`docs/nsys_ncu.md`
- agentic/local workflow：`docs/local_agentic_upsampling_guide.md`、`docs/local_agentic_t2i_quickstart.md`、`docs/serial_agentic_workflows.md`
- diffusion 共置：`docs/diffusion_colocation.md`
- 历史设计稿：`docs/archive/`

统一从仓库根目录以模块方式运行 Python 入口，例如：

```bash
python -m profiling_tools.runners.run_profiled --help
python -m profiling_tools.runners.colocate_decode_t2v
```

不要直接执行包内 Python 文件。这种运行方式会改变
`sys.path`，也是过去产生双层 `profiling_tools/profiling_tools/` 兼容副本的原因。

This directory contains non-invasive runtime wrappers for profiling four Cosmos3-Edge tasks:

The single-process vLLM/vLLM-Omni investigation starts with
[`docs/gate_minus1_unified_environment.md`](docs/gate_minus1_unified_environment.md) and
[`docs/gate_0_5_memory_budget.md`](docs/gate_0_5_memory_budget.md).

- T2I
- T2V
- Action Policy
- Video-to-Text (Video Reasoner)

All commands below run inside the existing `cosmos3-clean` container. The repository is mounted at
`/workspace/cosmos-framework`; the Hugging Face and uv caches are mounted under `/root/.cache`.

## 1. Enter the environment

Run this on the host:

```bash
docker exec -it cosmos3-clean bash
```

Then run this inside the container:

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate
mkdir -p profiling_results/components outputs/component_profile
python -m py_compile profiling_tools/*.py
```

## 2. Check the GPU and Green Context support

```bash
python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print("GPU:", p.name)
print("SM count:", p.multi_processor_count)
print("CUDA:", torch.version.cuda)
print("PyTorch:", torch.__version__)
PY
```

```bash
python - <<'PY'
from profiling_tools.core.green_context import GreenContextStream
ctx = GreenContextStream(8, device=0)
print("requested SMs:", ctx.requested_sm_count)
print("actual SMs:", ctx.actual_sm_count)
print("stream:", hex(ctx.stream_ptr))
ctx.close()
PY
```

On the RTX 5090 the expected values are 170 total SMs and an actual 8-SM partition.

## 3. Full-GPU component baselines

The commands use standard multiline Bash. A line-continuation backslash must be the final character on its line; do not put spaces after it.

### T2I

```bash
COSMOS_COMPONENT_PROFILE_OUTPUT=profiling_results/components/t2i_full.json \
python -m profiling_tools.runners.run_profiled \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/omni/t2i.json \
  -o outputs/component_profile/t2i_full \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

### T2V

```bash
COSMOS_COMPONENT_PROFILE_OUTPUT=profiling_results/components/t2v_full.json \
python -m profiling_tools.runners.run_profiled \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/omni/t2v.json \
  -o outputs/component_profile/t2v_full \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

### Action Policy

```bash
COSMOS_COMPONENT_PROFILE_OUTPUT=profiling_results/components/action_full.json \
python -m profiling_tools.runners.run_profiled \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/omni/action_policy_robot.json \
  -o outputs/component_profile/action_full \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

### Video-to-Text

```bash
COSMOS_COMPONENT_PROFILE_OUTPUT=profiling_results/components/reasoner_video_full.json \
python -m profiling_tools.runners.run_profiled \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/reasoner/reasoner_video.json \
  -o outputs/component_profile/reasoner_video_full \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

If a run needs to be repeated, remove both its inference output and its profiling JSON first. Example:

```bash
rm -rf outputs/component_profile/reasoner_video_full
rm -f profiling_results/components/reasoner_video_full.json
```

## 4. Component alignment

| Component | T2I | T2V | Action | Video-to-Text |
| --- | --- | --- | --- | --- |
| `condition_prepare` | yes | yes | yes | no |
| `text_reasoner` | prompt upsampling only | prompt upsampling only | prompt upsampling only | no |
| `diffusion_sampling` | yes | yes | yes | no |
| `vae_decode` | yes | yes | yes | no |
| `action_postprocess` | no | no | yes | no |
| `vision_encoder` | no | no | no | yes |
| `reasoner_prefill` | no | no | no | yes |
| `reasoner_decode` | no | no | no | yes |

Normal T2I, T2V, and Action inference does not run a separate Reasoner forward before diffusion. Text tokens participate in the unified MoT diffusion forward. A standalone `text_reasoner` appears only when native prompt upsampling is enabled.

## 5. Quick Green Context validation sweep

Run four representative partition sizes before launching the complete sweep.

### T2V diffusion quick check

```bash
SM_VALUES="8 64 128 168" \
bash profiling_tools/scripts/run_sm_sweep.sh t2v diffusion_sampling
```

### Video-to-Text decode quick check

```bash
SM_VALUES="8 64 128 168" \
bash profiling_tools/scripts/run_sm_sweep.sh reasoner_video reasoner_decode
```

## 6. Complete 8-SM-step sweep

The RTX 5090 has 170 SMs. The Green Context sweep uses `8, 16, ..., 168`; the full-GPU baseline represents 170 SMs.

### T2I

```bash
bash profiling_tools/scripts/run_sm_sweep.sh t2i
```

### T2V

```bash
bash profiling_tools/scripts/run_sm_sweep.sh t2v
```

### Action Policy

```bash
bash profiling_tools/scripts/run_sm_sweep.sh action
```

### Video-to-Text

```bash
bash profiling_tools/scripts/run_sm_sweep.sh reasoner_video
```

## 7. Summarize full-GPU baselines

```bash
python - <<'PY'
import json
import statistics
from pathlib import Path

for path in sorted(Path("profiling_results/components").glob("*_full.json")):
    records = json.loads(path.read_text())["records"]
    print(f"\n=== {path.name} ===")
    for component in sorted({r["component"] for r in records}):
        values = [r["gpu_ms"] for r in records if r["component"] == component and r["gpu_ms"] is not None]
        print(f"{component:24s} calls={len(values):4d} sum={sum(values):10.3f} ms mean={statistics.mean(values):10.3f} ms median={statistics.median(values):10.3f} ms")
PY
```

## 8. Generate the SM-scaling CSV

```bash
python - <<'PY'
import csv
import json
import re
import statistics
from pathlib import Path

root = Path("profiling_results/components")
output = Path("profiling_results/sm_scaling.csv")
pattern = re.compile(r"^(?P<task>t2i|t2v|action|reasoner_video)_(?P<component>.+)_sm(?P<sm>\d{3})\.json$")
rows = []

for path in sorted(root.glob("*_sm*.json")):
    match = pattern.match(path.name)
    if not match:
        continue
    records = json.loads(path.read_text())["records"]
    component = match.group("component")
    records = [r for r in records if r["component"] == component and r["gpu_ms"] is not None]
    if not records:
        print("WARNING: no matching records in", path)
        continue
    values = [float(r["gpu_ms"]) for r in records]
    actual = {r["actual_sm_count"] for r in records if r["actual_sm_count"] is not None}
    rows.append({
        "task": match.group("task"),
        "component": component,
        "requested_sm_count": int(match.group("sm")),
        "actual_sm_count": next(iter(actual)) if len(actual) == 1 else "",
        "calls": len(values),
        "gpu_total_ms": sum(values),
        "gpu_mean_ms": statistics.mean(values),
        "gpu_median_ms": statistics.median(values),
        "gpu_min_ms": min(values),
        "gpu_max_ms": max(values),
        "source": str(path),
    })

rows.sort(key=lambda r: (r["task"], r["component"], r["requested_sm_count"]))
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} rows to {output}")
PY
```

```bash
column -s, -t profiling_results/sm_scaling.csv | less -S
```

## 9. Check completeness

The complete sweep contains 273 Green Context JSON files and four full-GPU baseline files.

```bash
find profiling_results/components -maxdepth 1 -name '*_sm*.json' | wc -l
find profiling_results/components -maxdepth 1 -name '*_full.json' | wc -l
```

Expected output:

```text
273
4
```

## 10. Plot component shares and speedup versus SM count

Run this after generating `profiling_results/sm_scaling.csv`:

```bash
python -m profiling_tools.analysis.plot_sm_scaling
```

The script reads the four `*_full.json` files as 170-SM reference points and creates exactly two main figures, in both PNG and PDF formats:

```text
profiling_results/plots/component_time_breakdown.png
profiling_results/plots/component_time_breakdown.pdf
profiling_results/plots/speedup_vs_sm_count.png
profiling_results/plots/speedup_vs_sm_count.pdf
```

`component_time_breakdown` is a 2x2 set of task-level pie charts. Each pie is normalized over the independently profiled major components; overlapping container ranges such as `reasoner_e2e` are excluded.

`speedup_vs_sm_count` is a 2x2 set of task-level curves using:

```text
speedup = full-170-SM time / restricted-SM time
```

The 170-SM point is therefore 1.0. Values below 1.0 mean that restricting the component to fewer SMs made it slower. `reasoner_decode` uses the sum of all per-token decode records, so both figures represent the complete decode stage rather than one token.

Custom paths can be supplied with:

```bash
python -m profiling_tools.analysis.plot_sm_scaling \
  --csv profiling_results/sm_scaling.csv \
  --components-dir profiling_results/components \
  --output-dir profiling_results/plots \
  --full-sm-count 170
```

## 11. Persist with Git

```bash
git add profiling_tools
git status --short
git commit -m "Add Cosmos component and Green Context profiling tools"
```

Add the JSON/CSV results only if they should be version controlled:

```bash
git add profiling_results
git commit -m "Add Cosmos component profiling results"
```
