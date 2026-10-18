# Serial prompt upsampling and agentic T2I refinement

Run every command from `/workspace/cosmos-framework` inside the clean
container.  These recipes are deliberately serial: each inference process
exits before the next one starts, so one GPU is sufficient.

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate
mkdir -p inputs/serial_workflows outputs/serial_workflows
```

The commands below use the repository's `inputs/omni/t2i.json` as the base
request and `Cosmos3-Edge` for both reasoner and generator stages.

## A. Prompt upsampling -> T2I

Create a text-only Reasoner request containing the repository's official T2I
upsampling instructions and schema:

```bash
python -m profiling_tools.runners.serial_agentic_workflows make-upsample-request \
  --base-input inputs/omni/t2i.json \
  --output inputs/serial_workflows/upsample_reasoner.json \
  --resolution 480 \
  --aspect-ratio 1,1
```

Run the Reasoner:

```bash
python -m profiling_tools.runners.run_profiled \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/serial_workflows/upsample_reasoner.json \
  -o outputs/serial_workflows/01_upsample_reasoner \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

Convert its structured text output into an ordinary T2I request:

```bash
python -m profiling_tools.runners.serial_agentic_workflows make-t2i-from-upsample \
  --base-input inputs/omni/t2i.json \
  --reasoner-text outputs/serial_workflows/01_upsample_reasoner/upsample_reasoner/reasoner_text.txt \
  --output inputs/serial_workflows/t2i_upsampled.json
```

Run T2I with the expanded prompt:

```bash
python -m profiling_tools.runners.run_profiled \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/serial_workflows/t2i_upsampled.json \
  -o outputs/serial_workflows/02_t2i_upsampled \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

The image is written to:

```text
outputs/serial_workflows/02_t2i_upsampled/t2i_upsampled/vision.jpg
```

## B. Initial T2I -> visual critique -> refined T2I

First run the unchanged baseline request:

```bash
python -m profiling_tools.runners.run_profiled \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/omni/t2i.json \
  -o outputs/serial_workflows/10_t2i_initial \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

Build an image-conditioned Reasoner request:

```bash
python -m profiling_tools.runners.serial_agentic_workflows make-refinement-request \
  --base-input inputs/omni/t2i.json \
  --image outputs/serial_workflows/10_t2i_initial/t2i/vision.jpg \
  --output inputs/serial_workflows/t2i_critique.json
```

Run the visual critique:

```bash
python -m profiling_tools.runners.run_profiled \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/serial_workflows/t2i_critique.json \
  -o outputs/serial_workflows/11_t2i_critique \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

Turn `revised_prompt` from the critique JSON into the next T2I request:

```bash
python -m profiling_tools.runners.serial_agentic_workflows make-refined-t2i \
  --base-input inputs/omni/t2i.json \
  --reasoner-text outputs/serial_workflows/11_t2i_critique/t2i_critique/reasoner_text.txt \
  --output inputs/serial_workflows/t2i_refined.json
```

Run the second T2I round:

```bash
python -m profiling_tools.runners.run_profiled \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/serial_workflows/t2i_refined.json \
  -o outputs/serial_workflows/12_t2i_refined \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

The refined image is written to:

```text
outputs/serial_workflows/12_t2i_refined/t2i_refined/vision.jpg
```

For a second refinement round, use `t2i_refined.json` as `--base-input` and
the refined image as `--image`, then repeat the final three commands with new
names and output paths.

## Profiling outputs

`run_profiled` installs the existing component hooks, but it writes component
records only when `COSMOS_COMPONENT_PROFILE_OUTPUT` is set.  Give every stage
its own file, for example:

```bash
COSMOS_COMPONENT_PROFILE_OUTPUT=profiling_results/serial_workflows/t2i_refined.json \
python -m profiling_tools.runners.run_profiled \
  --parallelism-preset=latency \
  --no-guardrails \
  -i inputs/serial_workflows/t2i_refined.json \
  -o outputs/serial_workflows/12_t2i_refined \
  --checkpoint-path Cosmos3-Edge \
  --seed=0
```

Do not point multiple stages at the same profiling JSON file: each process
writes its own result file.
