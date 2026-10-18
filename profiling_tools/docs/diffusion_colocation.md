# Diffusion sampling Green Context colocation

This experiment captures the real `diffusion_sampling` invocation from Cosmos3
T2V, T2I, and Action inference. It then replays only each sampler invocation;
condition preparation, VAE decode, guardrails, and output writing are outside the
measured regions.

The implementation follows FastVideo's proven pair allocation: one
`cuDevSmResourceSplitByCount` call produces a requested first partition and its
remainder, and one Green Context/stream is created for each resource. Do not
replace this with two independent single-stream Green Context allocations,
because those do not guarantee disjoint SM sets.

## Quick validation

Enter the existing container:

```bash
docker exec -it cosmos3-clean bash
cd /workspace/cosmos-framework
```

Run all pairwise cases with one measured iteration after two warmup iterations
at every stage:

```bash
.venv/bin/python -m profiling_tools.runners.colocate_diffusion \
  --checkpoint-path Cosmos3-Edge \
  --profile-iters 1 \
  --warmup-iters 2 \
  --output profiling_results/colocation/diffusion_pairwise_quick.json
```

The default cases are:

- `t2v+t2i@128`, `@136`, and `@142`: T2V receives the stated SM count and
  T2I receives the remainder.
- `t2v+action@128`, `@136`, and `@142`: T2V receives the stated SM count and
  Action receives the remainder.
- `t2i+action@80`: T2I receives 80 SMs and Action receives the remainder.

On the current GPU the driver exposes 170 SMs, so the actual splits are
128+42 and 136+34; a request for 142 is rounded by the driver to 144+26;
80+90. The result always records the actual
driver-returned SM counts.

For a less noisy confirmation run:

```bash
.venv/bin/python -m profiling_tools.runners.colocate_diffusion \
  --checkpoint-path Cosmos3-Edge \
  --profile-iters 3 \
  --warmup-iters 3 \
  --output profiling_results/colocation/diffusion_pairwise_confirm.json
```

To run only one case or change the split, repeat `--case` as needed:

```bash
.venv/bin/python -m profiling_tools.runners.colocate_diffusion \
  --case t2v+t2i@136 \
  --case t2v+action@136 \
  --profile-iters 1 \
  --output profiling_results/colocation/diffusion_pairwise_136.json
```

The integer after `@` is always the SM request for the first task. The second
task receives the exact remainder from the same CUDA resource split.

## Warm and cold measurements

The script uses `--warmup-iters` independently at three stages: unrestricted
full-SM replay for each task, each workload alone on its Green Context
partition, and the two balanced workloads running concurrently. Measurements
start only after the corresponding warmups finish. The JSON preserves every
measured unrestricted baseline in `full_sm_alone_samples` as well as its mean
in `full_sm_alone_ms`.

Use warm measurements as the primary result for a persistent inference worker:
model weights, compiled graphs, CUDA libraries, and allocator state normally
survive across requests. Cold behavior matters for process startup, aggressive
autoscaling/scale-to-zero, model replacement, or a previously unseen input
shape, and should be reported separately as startup latency. A one-shot CLI
process is effectively cold on every invocation.

## Work balancing and metrics

Before every colocated pair, the script runs both tasks alone on the same two
partitions. It uses these measurements to choose a small integer repeat count
that makes the two colocated workloads similar in duration. With the current
scaling data, T2I and Action are expected to repeat about twice when paired with
T2V.

Each case reports:

- `partition_alone_ms_per_request`: one request alone on its assigned partition;
- `partition_alone_ms_balanced_workload`: partition-alone time after applying
  the selected repeat count;
- `colocated_ms_balanced_workload`: per-task elapsed time during colocation;
- `interference_slowdown_pct`: colocated divided by partition-alone for the
  same SM count and same repeated work;
- `full_sm_pure_serial_ms`: the same work run serially using unrestricted SMs;
- `partition_pure_serial_ms`: the same work serially on the assigned partitions;
- `colocated_gpu_makespan_ms`: the pair completion time;
- `speedup_vs_full_sm_serial` and `speedup_vs_partition_serial`;
- `balanced_tail_ms`: how much one workload continues after the other finishes.

Diffusion cache is explicitly disabled during capture so that replayed calls do
not share mutable SeaCache state. Transformer parameters are shared read-only
between the two host workers; sampler input tensors are cloned for every replay.

## Weight sharing

All three inputs in this experiment use `--checkpoint-path Cosmos3-Edge`. The
inference entry point creates one pipeline/model, builds every sample against
that model configuration, and then generates all samples. Consequently T2V,
T2I, and Action use the same loaded checkpoint and the same in-memory model
parameters. They do not execute identical workloads: model mode, conditioning,
latent shapes, sequence lengths, sampling defaults, and task-specific pathways
still differ.
