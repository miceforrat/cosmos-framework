#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

export LD_LIBRARY_PATH=
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export COSMOS_REASONER_REPEATS="${COSMOS_REASONER_REPEATS:-1}"
export COSMOS_WARMUP_RUNS="${COSMOS_WARMUP_RUNS:-1}"
export COSMOS_COLOCATE_GENERATOR_REGION=full
export COSMOS_REASONER_VALIDATE_TOKENS=1

unset COSMOS_REASONER_FIXED_LENGTH_DECODE
unset COSMOS_FULL_SM_DECODE_ONLY
unset COSMOS_GC_DECODE_ONLY
unset COSMOS_NSYS_CAPTURE

# Green Context allocation is quantized. These requested generator values
# produce the following actual partitions on the 170-SM test GPU:
#   requested 154 -> actual Reasoner 10 + Generator 160
#   requested 146 -> actual Reasoner 18 + Generator 152
#   requested 130 -> actual Reasoner 34 + Generator 136
#   requested 114 -> actual Reasoner 50 + Generator 120
requested_generator_values="${REQUESTED_GENERATOR_VALUES:-154 146 130 114}"
implementations="${IMPLEMENTATIONS:-eager graph}"
run_tag="${RUN_TAG:-run01}"
output_root="${OUTPUT_ROOT:-profiling_results/decode_t2v_impact_matrix}"

mkdir -p "$output_root"

for implementation in $implementations; do
  case "$implementation" in
    eager)
      unset COSMOS_REASONER_CUDA_GRAPH
      unset COSMOS_REASONER_CUDA_GRAPH_PERSISTENT
      unset COSMOS_REASONER_STATIC_KV_EAGER
      ;;
    static_eager)
      unset COSMOS_REASONER_CUDA_GRAPH
      unset COSMOS_REASONER_CUDA_GRAPH_PERSISTENT
      export COSMOS_REASONER_STATIC_KV_EAGER=1
      ;;
    graph)
      unset COSMOS_REASONER_STATIC_KV_EAGER
      export COSMOS_REASONER_CUDA_GRAPH=1
      export COSMOS_REASONER_CUDA_GRAPH_PERSISTENT=1
      ;;
    *)
      echo "Unknown implementation: $implementation (expected eager, static_eager, or graph)" >&2
      exit 2
      ;;
  esac

  for requested_generator in $requested_generator_values; do
    export COSMOS_GENERATOR_SM_COUNT="$requested_generator"
    log="$output_root/${implementation}_requested_generator_sm$(printf %03d "$requested_generator")_${run_tag}.log"
    echo
    echo "=== implementation=$implementation requested_generator=$requested_generator log=$log ==="
    python -m profiling_tools.runners.colocate_decode_t2v 2>&1 | tee "$log"
  done
done
