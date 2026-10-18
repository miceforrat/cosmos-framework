#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

export LD_LIBRARY_PATH=''
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export COSMOS_REASONER_CUDA_GRAPH=1
export COSMOS_REASONER_CUDA_GRAPH_PERSISTENT=1
export COSMOS_REASONER_VALIDATE_TOKENS=1
export COSMOS_REASONER_REPEATS=1
export COSMOS_GC_DECODE_ONLY=1
unset COSMOS_FULL_SM_DECODE_ONLY COSMOS_NSYS_CAPTURE

# Realistic EOS early-exit is the default. Fixed length is retained only as an
# explicit profiling control and writes to a separate directory.
fixed_length="${FIXED_LENGTH_DECODE:-0}"
if [[ "$fixed_length" == "1" ]]; then
  export COSMOS_REASONER_FIXED_LENGTH_DECODE=1
  default_output_dir="profiling_results/reasoner_graph_sm_scaling_fixed_length"
else
  unset COSMOS_REASONER_FIXED_LENGTH_DECODE
  default_output_dir="profiling_results/reasoner_graph_sm_scaling_early_exit"
fi

total_sms="${TOTAL_SMS:-170}"
generator_sm_values="${GENERATOR_SM_VALUES:-8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168}"
output_dir="${OUTPUT_DIR:-$default_output_dir}"
mkdir -p "$output_dir"

for generator_sms in $generator_sm_values; do
  reasoner_sms=$((total_sms - generator_sms))
  if (( reasoner_sms <= 0 )); then
    echo "Skipping requested generator=${generator_sms}: reasoner remainder ${reasoner_sms} <= 0" >&2
    continue
  fi
  export COSMOS_GENERATOR_SM_COUNT="$generator_sms"
  log="$output_dir/requested_reasoner_sm$(printf '%03d' "$reasoner_sms").log"
  echo "=== requested_generator=${generator_sms} nominal_reasoner_remainder=${reasoner_sms} ==="
  python -m profiling_tools.runners.colocate_decode_t2v 2>&1 | tee "$log"
done

# Full-SM reference uses a regular unrestricted primary-context stream.
unset COSMOS_GC_DECODE_ONLY
export COSMOS_FULL_SM_DECODE_ONLY=1
export COSMOS_GENERATOR_SM_COUNT=136
python -m profiling_tools.runners.colocate_decode_t2v 2>&1 \
  | tee "$output_dir/full_sm.log"
