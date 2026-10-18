# Cold/warm component SM scaling

This procedure measures component latency and component-time shares separately
for cold and warm execution. It covers T2I, T2V, Action Policy, and Video
Reasoner, using the same timing wrapper for unrestricted full-SM baselines and
Green Context SM sweeps.

## Measurement semantics

Each process runs:

- pass 0: `cold`;
- passes 1 and 2: additional `warmup` passes;
- five subsequent passes: `measured` warm samples.

The defaults below use three fresh processes. This produces three independent
cold samples and fifteen measured warm samples per task/component/SM point.
Every JSON record contains `pass_index`, `phase`, `iteration_index`, `gpu_ms`,
`requested_sm_count`, and `actual_sm_count`.

Do not compare the old `t2v_full.json` directly with cache-off colocation data.
The former used the default diffusion cache, while the colocation capture used
`--no-diffusion-cache`. Cache mode and timing boundaries must match.

## Enter the container

```bash
docker exec -it cosmos3-clean bash
cd /workspace/cosmos-framework
source .venv/bin/activate
```

## Full-SM baselines

Run all four tasks with the production/default cache mode:

```bash
export WARMUP_ITERS=3
export PROFILE_ITERS=5
export PROCESS_REPEATS=3
export DIFFUSION_CACHE=on

for task in t2i t2v action reasoner_video; do
  bash profiling_tools/scripts/run_full_baseline_phased.sh "$task"
done
```

These files are written under:

```text
profiling_results/components_phased_on/
```

Repeat the three diffusion-generation tasks with diffusion cache disabled:

```bash
export DIFFUSION_CACHE=off

for task in t2i t2v action; do
  bash profiling_tools/scripts/run_full_baseline_phased.sh "$task"
done
```

These files are written under:

```text
profiling_results/components_phased_off/
```

Video Reasoner does not execute diffusion sampling and does not need a second
cache-off baseline.

## High-SM diffusion validation

Measure the region most relevant to the colocation experiment before launching
the complete sweep:

```bash
export WARMUP_ITERS=3
export PROFILE_ITERS=5
export PROCESS_REPEATS=3
export SM_VALUES="80 96 112 128 136 144 152 160 168"
```

Cache-on:

```bash
export DIFFUSION_CACHE=on

for task in t2i t2v action; do
  bash profiling_tools/scripts/run_sm_sweep_phased.sh "$task" diffusion_sampling
done
```

Cache-off:

```bash
export DIFFUSION_CACHE=off

for task in t2i t2v action; do
  bash profiling_tools/scripts/run_sm_sweep_phased.sh "$task" diffusion_sampling
done
```

Use these results to check whether the earlier scaling curve is reproducible,
whether cache mode changes scaling, and whether T2I/Action retain a broad
low-sensitivity SM region after warmup.

## High-SM sweeps for the other components

The default cache-on mode is sufficient for these components:

```bash
export DIFFUSION_CACHE=on
export WARMUP_ITERS=3
export PROFILE_ITERS=5
export PROCESS_REPEATS=3
export SM_VALUES="80 96 112 128 136 144 152 160 168"
```

T2I:

```bash
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  t2i \
  condition_prepare \
  vae_decode
```

T2V:

```bash
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  t2v \
  condition_prepare \
  vae_decode
```

Action Policy:

```bash
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  action \
  condition_prepare \
  vae_decode \
  action_postprocess
```

Video Reasoner:

```bash
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  reasoner_video \
  vision_encoder \
  reasoner_prefill \
  reasoner_decode
```

## Complete 8-SM-step sweep

After validating the high-SM results, unset `SM_VALUES`. The scripts then use
`8, 16, ..., 168`; unrestricted execution supplies the 170-SM reference.

```bash
unset SM_VALUES
export WARMUP_ITERS=3
export PROFILE_ITERS=5
export PROCESS_REPEATS=3
export DIFFUSION_CACHE=on
```

```bash
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  t2i condition_prepare diffusion_sampling vae_decode

bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  t2v condition_prepare diffusion_sampling vae_decode

bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  action condition_prepare diffusion_sampling vae_decode action_postprocess

bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  reasoner_video vision_encoder reasoner_prefill reasoner_decode
```

The exhaustive sweep is expensive. To reduce total time while retaining five
warm samples per process, use two fresh processes:

```bash
export PROCESS_REPEATS=2
```

## Resuming and custom result directories

Non-empty JSON files are skipped, so every command is safe to rerun after an
interruption. Override the result and inference-output locations if required:

```bash
RESULTS_DIR=profiling_results/my_component_run \
OUTPUTS_DIR=outputs/my_component_run \
bash profiling_tools/scripts/run_sm_sweep_phased.sh t2v diffusion_sampling
```

Keep cache-on and cache-off data in different directories. Mixing them would
make both the SM curves and component shares ambiguous.

## Intended analysis

Produce separate summaries for:

- cold full-SM component shares;
- warm full-SM component shares;
- cold latency versus SM count;
- warm latency versus SM count;
- cache-on versus cache-off diffusion scaling.

For components with multiple records per generation, especially
`reasoner_decode`, first sum all calls sharing the same `pass_index`, then
compute statistics across passes and processes. Use the median as the main
latency and report mean, standard deviation, minimum, and maximum alongside it.

## What cache-on and cache-off mean

`DIFFUSION_CACHE=on` uses the inference CLI default and installs Cosmos
SeaCache during diffusion generation. SeaCache observes how the noisy vision
state changes between denoising steps. When its accumulated relative-L1
indicator remains below the configured threshold, it can reuse/extrapolate a
cached language-model residual instead of executing the complete diffusion
transformer computation for that step. Cosmos defaults to a threshold of 0.25
and limits consecutive cached steps. This reduces diffusion latency, with a
possible quality tradeoff.

`DIFFUSION_CACHE=off` passes `--no-diffusion-cache`. Every denoising step runs
the full model path. This mode is slower, but it is useful as an uncached
compute baseline and matches the current colocation replay, where SeaCache was
disabled to avoid sharing mutable cache state between concurrent workloads.

This setting is unrelated to CUDA compilation caches, model-weight caching,
the allocator cache, or the autoregressive KV cache. It only controls the
diffusion-time SeaCache optimization. Video Reasoner does not execute diffusion
sampling, so cache-on/off does not affect its workload.

The first T2V phased run produced the following diffusion-sampling values:

| Mode | Cold | Warm median |
| --- | ---: | ---: |
| cache-on | 23.658 s | 13.478 s |
| cache-off | 35.762 s | 25.840 s |

This resolves the earlier apparent contradiction: the old approximately
23.5-second result was cache-on cold, while the approximately 25.7-second
colocation baseline was cache-off warm. Warm is faster than cold when cache
mode and timing boundaries match.

## Commands after the initial T2V run

At the time of the first check, only `t2v_full_run01.json` existed in each
cache-mode directory. Use inline environment assignments below so the requested
values remain attached to the command even when running in a new shell. Existing
`run01` files are skipped, and `run02`/`run03` are generated.

Complete the T2V full-SM repetitions:

```bash
WARMUP_ITERS=3 \
PROFILE_ITERS=5 \
PROCESS_REPEATS=3 \
DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_full_baseline_phased.sh t2v

WARMUP_ITERS=3 \
PROFILE_ITERS=5 \
PROCESS_REPEATS=3 \
DIFFUSION_CACHE=off \
bash profiling_tools/scripts/run_full_baseline_phased.sh t2v
```

Collect cache-on full-SM cold/warm component shares for the remaining tasks:

```bash
for task in t2i action reasoner_video; do
  WARMUP_ITERS=3 \
  PROFILE_ITERS=5 \
  PROCESS_REPEATS=3 \
  DIFFUSION_CACHE=on \
  bash profiling_tools/scripts/run_full_baseline_phased.sh "$task"
done
```

Collect cache-off full-SM results for the remaining diffusion tasks:

```bash
for task in t2i action; do
  WARMUP_ITERS=3 \
  PROFILE_ITERS=5 \
  PROCESS_REPEATS=3 \
  DIFFUSION_CACHE=off \
  bash profiling_tools/scripts/run_full_baseline_phased.sh "$task"
done
```

Then run the cache-on high-SM sweep for every major component:

```bash
SM_VALUES="80 96 112 128 136 144 152 160 168" \
WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  t2i condition_prepare diffusion_sampling vae_decode

SM_VALUES="80 96 112 128 136 144 152 160 168" \
WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  t2v condition_prepare diffusion_sampling vae_decode

SM_VALUES="80 96 112 128 136 144 152 160 168" \
WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  action condition_prepare diffusion_sampling vae_decode action_postprocess

SM_VALUES="80 96 112 128 136 144 152 160 168" \
WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  reasoner_video vision_encoder reasoner_prefill reasoner_decode
```

Finally, collect the matching cache-off high-SM diffusion curves:

```bash
for task in t2i t2v action; do
  SM_VALUES="80 96 112 128 136 144 152 160 168" \
  WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=off \
  bash profiling_tools/scripts/run_sm_sweep_phased.sh "$task" diffusion_sampling
done
```

## Current experiment policy: cache-on is primary

For production-representative component shares, SM scaling, and scheduling
decisions, run all remaining experiments with `DIFFUSION_CACHE=on`. The
cache-off T2V `run01` is retained only as a diagnostic that explains the old
timing mismatch. Do not spend time completing cache-off repetitions or
cache-off SM sweeps unless an explicit uncached ablation is needed later.

The next command from the current repository state is therefore:

```bash
for task in t2v t2i action reasoner_video; do
  WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
  bash profiling_tools/scripts/run_full_baseline_phased.sh "$task"
done
```

The existing T2V `run01` is skipped automatically, so T2V continues with
`run02`/`run03`; the other tasks run `run01` through `run03`.

Then run the high-SM cache-on sweep:

```bash
SM_VALUES="80 96 112 128 136 144 152 160 168" \
WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  t2i condition_prepare diffusion_sampling vae_decode

SM_VALUES="80 96 112 128 136 144 152 160 168" \
WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  t2v condition_prepare diffusion_sampling vae_decode

SM_VALUES="80 96 112 128 136 144 152 160 168" \
WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  action condition_prepare diffusion_sampling vae_decode action_postprocess

SM_VALUES="80 96 112 128 136 144 152 160 168" \
WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  reasoner_video vision_encoder reasoner_prefill reasoner_decode
```

The colocation experiment must also be rerun with cache-on before drawing a
deployment conclusion. SeaCache currently stores mutable state on the shared
model, so concurrent workloads need isolated cache state; simply removing
`--no-diffusion-cache` from the existing colocation script would introduce a
race and is not a valid cache-on experiment.

## Full cache-on sweep and updated plots

Run every 8-SM point after the full-SM baselines have completed:

```bash
unset SM_VALUES

WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  t2i condition_prepare diffusion_sampling vae_decode

WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  t2v condition_prepare diffusion_sampling vae_decode

WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  action condition_prepare diffusion_sampling vae_decode action_postprocess

WARMUP_ITERS=3 PROFILE_ITERS=5 PROCESS_REPEATS=3 DIFFUSION_CACHE=on \
bash profiling_tools/scripts/run_sm_sweep_phased.sh \
  reasoner_video vision_encoder reasoner_prefill reasoner_decode
```

Aggregate each component/pass before plotting. `reasoner_decode` is summed over
all tokens in the same pass. Cold and measured warm samples remain separate:

```bash
python -m profiling_tools.analysis.aggregate_phased_results \
  --input-dir profiling_results/components_phased_on \
  --output profiling_results/sm_scaling_phased_on.csv
```

Generate the updated figures:

```bash
python -m profiling_tools.analysis.plot_phased_sm_scaling \
  --csv profiling_results/sm_scaling_phased_on.csv \
  --components-dir profiling_results/components_phased_on \
  --output-dir profiling_results/plots_phased_on \
  --full-sm-count 170
```

Outputs:

```text
profiling_results/plots_phased_on/component_time_breakdown_cold.png
profiling_results/plots_phased_on/component_time_breakdown_cold.pdf
profiling_results/plots_phased_on/component_time_breakdown_measured.png
profiling_results/plots_phased_on/component_time_breakdown_measured.pdf
profiling_results/plots_phased_on/speedup_vs_sm_count_cold.png
profiling_results/plots_phased_on/speedup_vs_sm_count_cold.pdf
profiling_results/plots_phased_on/speedup_vs_sm_count_measured.png
profiling_results/plots_phased_on/speedup_vs_sm_count_measured.pdf
```

The speedup definition is `full-SM median / restricted-SM median`; therefore
the unrestricted 170-SM point is 1.0. Shaded regions show the speedup range
induced by the minimum and maximum restricted-SM samples.
