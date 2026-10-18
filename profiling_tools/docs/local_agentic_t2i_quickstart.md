# Minimal local prompt upsampling -> T2I

This is the first, deliberately serial version of the local agentic T2I
workflow.  It uses Cosmos3-Edge for both stages and makes no remote API calls:

```text
natural-language prompt
-> local text-only Cosmos Reasoner
-> validated structured T2I JSON
-> local Cosmos T2I generator
-> image
```

Each stage runs in a separate process.  This is useful for correctness and
component profiling, but its end-to-end wall time includes two checkpoint
loads and is not the final warm-service latency.

## Run

Inside the clean container:

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate
export LD_LIBRARY_PATH=''

python -m profiling_tools.runners.run_local_agentic_t2i \
  --prompt 'A red robot repairs a bicycle in a bright workshop.' \
  --output-dir outputs/local_agentic_t2i/smoke01 \
  --checkpoint-path Cosmos3-Edge \
  --resolution 480 \
  --aspect-ratio 1,1 \
  --seed 0
```

For a prompt stored in a text file, use `--prompt-file` instead of `--prompt`:

```bash
python -m profiling_tools.runners.run_local_agentic_t2i \
  --prompt-file inputs/my_single_t2i_prompt.txt \
  --output-dir outputs/local_agentic_t2i/smoke02 \
  --checkpoint-path Cosmos3-Edge \
  --seed 0
```

`--prompt-file` must contain exactly one non-empty line. Batch prompt
scheduling is intentionally deferred.

Component profiling is enabled by default. Disable it only for a functional
smoke test:

```bash
python -m profiling_tools.runners.run_local_agentic_t2i \
  --prompt 'A red robot repairs a bicycle in a bright workshop.' \
  --output-dir outputs/local_agentic_t2i/smoke_no_profile \
  --no-component-profile
```

## Compact upsampling

For colocation experiments, compact mode asks the local Reasoner for only the
80-140 word caption consumed by T2I. Its default generation limit is 768 tokens;
full mode remains unchanged at 4096 tokens. Use a new output directory:

```bash
python -m profiling_tools.runners.run_local_agentic_t2i \
  --prompt 'A red robot repairs a bicycle in a bright workshop.' \
  --output-dir outputs/local_agentic_t2i/compact01 \
  --checkpoint-path Cosmos3-Edge \
  --resolution 480 \
  --aspect-ratio 1,1 \
  --compact-upsampling \
  --seed 0
```


## Green Context stage split

Restrict the Upsampler and Generator stages to different SM counts:

```bash
python -m profiling_tools.runners.run_local_agentic_t2i \
  --prompt 'A red robot repairs a bicycle in a bright workshop.' \
  --output-dir outputs/local_agentic_t2i/gc_smoke01 \
  --checkpoint-path Cosmos3-Edge \
  --upsampler-sm-count 64 \
  --generator-sm-count 128 \
  --seed 0
```

The Upsampler GC wraps `reasoner_e2e`, `reasoner_prefill`, and
`reasoner_decode`. They use one shared 64-SM Green Context in this example.
The Generator GC wraps `condition_prepare`, `diffusion_sampling`, and
`vae_decode`, using one shared 128-SM Green Context.

This is a coarse two-stage allocation. Component records remain split at five
fine boundaries, but this runner does not yet assign different SM counts to
prefill versus decode or to individual Generator components.

## Fine-grained SM scaling

Scaling uses one component and one Green Context size per process. It reuses
the frozen inputs from the completed smoke run.

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate
export LD_LIBRARY_PATH=''

export AGENTIC_WORKFLOW_DIR=outputs/local_agentic_t2i/smoke01
export REPEATS=3

./profiling_tools/scripts/run_full_baseline.sh agentic_upsampler
./profiling_tools/scripts/run_full_baseline.sh agentic_generator

./profiling_tools/scripts/run_sm_sweep.sh agentic_upsampler
./profiling_tools/scripts/run_sm_sweep.sh agentic_generator
```

The default sweep is `8, 16, ..., 168` SMs. For a quick validation:

```bash
SM_VALUES='8 64 128 168' REPEATS=1 \
  ./profiling_tools/scripts/run_sm_sweep.sh agentic_upsampler
```

Aggregate and plot:

```bash
python -m profiling_tools.analysis.aggregate_results \
  --input-dir profiling_results/components \
  --output profiling_results/sm_scaling.csv

python -m profiling_tools.analysis.plot_local_agentic_sm_scaling \
  --csv profiling_results/sm_scaling.csv \
  --components-dir profiling_results/components \
  --output-dir profiling_results/plots_local_agentic \
  --full-sm-count 170
```

The five independent curves are:

```text
agentic_upsampler: reasoner_prefill, reasoner_decode
agentic_generator: condition_prepare, diffusion_sampling, vae_decode
```

`reasoner_e2e` is excluded because it overlaps prefill and decode.


## Cold and warm phased scaling (recommended)

Increasing `REPEATS` in the legacy sweep only starts more fresh processes, so
it does not turn those runs into warm measurements. Use the phased runners:

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate
export LD_LIBRARY_PATH=''

export AGENTIC_WORKFLOW_DIR=outputs/local_agentic_t2i/smoke01
export WARMUP_ITERS=3
export PROFILE_ITERS=5
export PROCESS_REPEATS=2
export DIFFUSION_CACHE=on
export RESULTS_DIR=profiling_results/components_agentic_phased_on
export OUTPUTS_DIR=outputs/component_profile_agentic_phased_on

./profiling_tools/scripts/run_full_baseline_phased.sh agentic_upsampler
./profiling_tools/scripts/run_full_baseline_phased.sh agentic_generator
./profiling_tools/scripts/run_sm_sweep_phased.sh agentic_upsampler
./profiling_tools/scripts/run_sm_sweep_phased.sh agentic_generator

python -m profiling_tools.analysis.aggregate_phased_results \
  --input-dir "$RESULTS_DIR" \
  --output profiling_results/sm_scaling_agentic_phased_on.csv

for phase in cold measured; do
  python -m profiling_tools.analysis.plot_local_agentic_sm_scaling \
    --csv profiling_results/sm_scaling_agentic_phased_on.csv \
    --components-dir "$RESULTS_DIR" \
    --output-dir profiling_results/plots_local_agentic_phased_on \
    --phase "$phase" \
    --full-sm-count 170
done
```

The plotting command also writes `local_agentic_component_time_breakdown_cold`
and `local_agentic_component_time_breakdown_measured` in PNG/PDF format. Each
legend entry includes the component percentage and absolute full-SM median time;
the panel title includes the summed component time.


For each fresh process, pass 0 is `cold`; two more passes complete the three
warmups; then five `measured` passes form the warm distribution.
`PROCESS_REPEATS=2` repeats that complete sequence in fresh processes. This
cold definition begins after checkpoint loading and measures first inference
initialization/cache effects, not model-load wall time.

## Outputs

```text
<output-dir>/
  request.json
  upsampled_prompt.json
  summary.json
  inputs/
    upsample_reasoner.json
    t2i_upsampled.json
  profiles/
    upsampler.json
    generator.json
  stages/
    01_upsampler/upsample_reasoner/
      reasoner_text.txt
      sample_args.json
      sample_outputs.json
    02_generator/t2i_upsampled/
      vision.jpg
      sample_args.json
      sample_outputs.json
```

Use a new output directory for every run.  The script does not delete or
overwrite a previous experiment directory.

## Current boundary

This version stops after the first generated image.  The next extension will
reuse the same stage contracts and append:

```text
03_critic -> 04_rewriter -> 05_generator_iter01
```

The final performance version should keep one full `OmniInference` instance
resident instead of reloading the checkpoint for every stage.
