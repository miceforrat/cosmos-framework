#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

task=${1:?Usage: run_sm_sweep_phased.sh TASK [COMPONENT ...]}
shift
case "$task" in
  t2i) input=inputs/omni/t2i.json; default_components=(condition_prepare diffusion_sampling vae_decode) ;;
  t2v) input=inputs/omni/t2v.json; default_components=(condition_prepare diffusion_sampling vae_decode) ;;
  action) input=inputs/omni/action_policy_robot.json; default_components=(condition_prepare diffusion_sampling vae_decode action_postprocess) ;;
  reasoner_video) input=inputs/reasoner/reasoner_video.json; default_components=(vision_encoder reasoner_prefill reasoner_decode) ;;
  agentic_upsampler)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2i/smoke01}
    input="$workflow_dir/inputs/upsample_reasoner.json"
    default_components=(reasoner_prefill reasoner_decode)
    ;;
  agentic_generator)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2i/smoke01}
    input="$workflow_dir/inputs/t2i_upsampled.json"
    default_components=(condition_prepare diffusion_sampling vae_decode)
    ;;
  agentic_video_upsampler)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2v/compact01}
    input="$workflow_dir/inputs/upsample_reasoner.json"
    default_components=(reasoner_prefill reasoner_decode)
    ;;
  agentic_video_generator)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2v/compact01}
    input="$workflow_dir/inputs/t2v_upsampled.json"
    default_components=(condition_prepare diffusion_sampling vae_decode)
    ;;
  *) echo "Unknown task: $task" >&2; exit 2 ;;
esac
if (( $# > 0 )); then components=("$@"); else components=("${default_components[@]}"); fi
[[ -s "$input" ]] || { echo "Input not found: $input" >&2; exit 2; }
if [[ -n ${SM_VALUES:-} ]]; then read -r -a sm_values <<< "$SM_VALUES"; else mapfile -t sm_values < <(seq 8 8 168); fi
warmup_iters=${WARMUP_ITERS:-2}
profile_iters=${PROFILE_ITERS:-5}
process_repeats=${PROCESS_REPEATS:-1}
cache_mode=${DIFFUSION_CACHE:-on}
results_dir=${RESULTS_DIR:-profiling_results/components_phased_${cache_mode}}
outputs_dir=${OUTPUTS_DIR:-outputs/component_profile_phased_${cache_mode}}
case "$cache_mode" in on) cache_args=() ;; off) cache_args=(--no-diffusion-cache) ;; *) echo "DIFFUSION_CACHE must be on or off" >&2; exit 2 ;; esac
mkdir -p "$results_dir" "$outputs_dir"
for component in "${components[@]}"; do
  for sm in "${sm_values[@]}"; do
    for repeat in $(seq 1 "$process_repeats"); do
      tag=$(printf "%03d" "$sm"); suffix=$(printf "_run%02d" "$repeat")
      name="${task}_${component}_sm${tag}${suffix}"
      result="${results_dir}/${name}.json"
      [[ -s "$result" ]] && { echo "Skipping completed ${name}"; continue; }
      echo "Running ${name}: cold=1 extra_warmup=$((warmup_iters - 1)) measured=${profile_iters} cache=${cache_mode}"
      COSMOS_GREEN_COMPONENT="$component" \
      COSMOS_GREEN_SM_COUNT="$sm" \
      COSMOS_COMPONENT_WARMUP_ITERS="$warmup_iters" \
      COSMOS_COMPONENT_PROFILE_OUTPUT="$result" \
      python -m profiling_tools.runners.run_profiled \
        --parallelism-preset=latency --no-guardrails --benchmark \
        --warmup "$warmup_iters" --num-iterations "$profile_iters" \
        "${cache_args[@]}" -i "$input" -o "${outputs_dir}/${name}" \
        --checkpoint-path Cosmos3-Edge --seed=0
    done
  done
done
