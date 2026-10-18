#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

task=${1:?Usage: run_full_baseline_phased.sh TASK}
case "$task" in
  t2i) input=inputs/omni/t2i.json ;;
  t2v) input=inputs/omni/t2v.json ;;
  action) input=inputs/omni/action_policy_robot.json ;;
  reasoner_video) input=inputs/reasoner/reasoner_video.json ;;
  agentic_upsampler)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2i/smoke01}
    input="$workflow_dir/inputs/upsample_reasoner.json"
    ;;
  agentic_video_upsampler)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2v/compact01}
    input="$workflow_dir/inputs/upsample_reasoner.json"
    ;;
  agentic_video_critic)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2v/agentic01}
    input="$workflow_dir/inputs/critic_reasoner.json"
    ;;
  agentic_video_generator)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2v/compact01}
    input="$workflow_dir/inputs/t2v_upsampled.json"
    ;;
  agentic_generator)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2i/smoke01}
    input="$workflow_dir/inputs/t2i_upsampled.json"
    ;;
  *) echo "Unknown task: $task" >&2; exit 2 ;;
esac
[[ -s "$input" ]] || { echo "Input not found: $input" >&2; exit 2; }
warmup_iters=${WARMUP_ITERS:-2}; profile_iters=${PROFILE_ITERS:-5}; process_repeats=${PROCESS_REPEATS:-1}
cache_mode=${DIFFUSION_CACHE:-on}
results_dir=${RESULTS_DIR:-profiling_results/components_phased_${cache_mode}}
outputs_dir=${OUTPUTS_DIR:-outputs/component_profile_phased_${cache_mode}}
case "$cache_mode" in on) cache_args=() ;; off) cache_args=(--no-diffusion-cache) ;; *) echo "DIFFUSION_CACHE must be on or off" >&2; exit 2 ;; esac
mkdir -p "$results_dir" "$outputs_dir"
for repeat in $(seq 1 "$process_repeats"); do
  suffix=$(printf "run%02d" "$repeat"); name="${task}_full_${suffix}"; result="${results_dir}/${name}.json"
  [[ -s "$result" ]] && { echo "Skipping completed ${name}"; continue; }
  echo "Running ${name}: cold=1 extra_warmup=$((warmup_iters - 1)) measured=${profile_iters} cache=${cache_mode}"
  COSMOS_COMPONENT_WARMUP_ITERS="$warmup_iters" COSMOS_COMPONENT_PROFILE_OUTPUT="$result" \
  python -m profiling_tools.runners.run_profiled \
    --parallelism-preset=latency --no-guardrails --benchmark \
    --warmup "$warmup_iters" --num-iterations "$profile_iters" \
    "${cache_args[@]}" -i "$input" -o "${outputs_dir}/${name}" \
    --checkpoint-path Cosmos3-Edge --seed=0
done
