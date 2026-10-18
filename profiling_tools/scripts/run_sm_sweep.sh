#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../.."

task=${1:?Usage: run_sm_sweep.sh TASK [COMPONENT ...]}
shift

case "$task" in
  t2i)
    input=inputs/omni/t2i.json
    default_components=(condition_prepare diffusion_sampling vae_decode)
    ;;
  t2v)
    input=inputs/omni/t2v.json
    default_components=(condition_prepare diffusion_sampling vae_decode)
    ;;
  action)
    input=inputs/omni/action_policy_robot.json
    default_components=(condition_prepare diffusion_sampling vae_decode action_postprocess)
    ;;
  reasoner_video)
    input=inputs/reasoner/reasoner_video.json
    default_components=(vision_encoder reasoner_prefill reasoner_decode)
    ;;
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
  *)
    echo "Unknown task: $task" >&2
    exit 2
    ;;
esac

if (( $# > 0 )); then
  components=("$@")
else
  components=("${default_components[@]}")
fi

if [[ -n ${SM_VALUES:-} ]]; then
  read -r -a sm_values <<< "$SM_VALUES"
else
  mapfile -t sm_values < <(seq 8 8 168)
fi
if [[ ! -f "$input" ]]; then
  echo "Missing input: $input" >&2
  echo "Run profiling_tools/runners/run_local_agentic_t2i.py first or set AGENTIC_WORKFLOW_DIR." >&2
  exit 2
fi
mkdir -p profiling_results/components outputs/component_profile
repeats=${REPEATS:-1}

for component in "${components[@]}"; do
  for sm in "${sm_values[@]}"; do
    for repeat in $(seq 1 "$repeats"); do
      tag=$(printf '%03d' "$sm")
      if (( repeats > 1 )); then
        suffix=$(printf '_run%02d' "$repeat")
      else
        suffix=""
      fi
      name="${task}_${component}_sm${tag}${suffix}"
      result="profiling_results/components/${name}.json"

      if [[ -s "$result" ]]; then
        echo "Skipping completed ${name}"
        continue
      fi

      echo "Running ${name}"
      if ! COSMOS_GREEN_COMPONENT="$component" \
        COSMOS_GREEN_SM_COUNT="$sm" \
        COSMOS_COMPONENT_PROFILE_OUTPUT="$result" \
        python -m profiling_tools.runners.run_profiled \
          --parallelism-preset=latency \
          --no-guardrails \
          -i "$input" \
          -o "outputs/component_profile/${name}" \
          --checkpoint-path Cosmos3-Edge \
          --seed=0; then
        echo "FAILED: ${name}" >&2
        echo "Before retrying, remove outputs/component_profile/${name}" >&2
        exit 1
      fi
    done
  done
done
