#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../.."

task=${1:?Usage: run_full_baseline.sh TASK}
repeats=${REPEATS:-5}

case "$task" in
  t2i) input=inputs/omni/t2i.json ;;
  t2v) input=inputs/omni/t2v.json ;;
  action) input=inputs/omni/action_policy_robot.json ;;
  reasoner_video) input=inputs/reasoner/reasoner_video.json ;;
  agentic_upsampler)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2i/smoke01}
    input="$workflow_dir/inputs/upsample_reasoner.json"
    ;;
  agentic_generator)
    workflow_dir=${AGENTIC_WORKFLOW_DIR:-outputs/local_agentic_t2i/smoke01}
    input="$workflow_dir/inputs/t2i_upsampled.json"
    ;;
  *) echo "Unknown task: $task" >&2; exit 2 ;;
esac
if [[ ! -f "$input" ]]; then
  echo "Missing input: $input" >&2
  echo "Run profiling_tools/runners/run_local_agentic_t2i.py first or set AGENTIC_WORKFLOW_DIR." >&2
  exit 2
fi

mkdir -p profiling_results/components outputs/component_profile
for repeat in $(seq 1 "$repeats"); do
  suffix=$(printf 'run%02d' "$repeat")
  name="${task}_full_${suffix}"
  result="profiling_results/components/${name}.json"
  if [[ -s "$result" ]]; then
    echo "Skipping completed ${name}"
    continue
  fi
  echo "Running ${name}"
  COSMOS_COMPONENT_PROFILE_OUTPUT="$result" \
  python -m profiling_tools.runners.run_profiled \
    --parallelism-preset=latency \
    --no-guardrails \
    -i "$input" \
    -o "outputs/component_profile/${name}" \
    --checkpoint-path Cosmos3-Edge \
    --seed=0 || exit 1
done
