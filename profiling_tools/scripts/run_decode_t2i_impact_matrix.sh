#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

export LD_LIBRARY_PATH=
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-8}"
export COSMOS_REASONER_REPEATS=1
export COSMOS_WARMUP_RUNS="${COSMOS_WARMUP_RUNS:-1}"
export COSMOS_COLOCATE_GENERATOR_REGION=full
export COSMOS_GENERATOR_WORKLOAD=t2i
export COSMOS_REASONER_VALIDATE_TOKENS=1

unset COSMOS_REASONER_FIXED_LENGTH_DECODE
unset COSMOS_FULL_SM_DECODE_ONLY
unset COSMOS_GC_DECODE_ONLY
unset COSMOS_NSYS_CAPTURE

implementations="${IMPLEMENTATIONS:-eager graph}"
requested_generator_values="${REQUESTED_GENERATOR_VALUES:-154 146 130 114}"
run_tag="${RUN_TAG:-matrix01}"
output_root="${OUTPUT_ROOT:-profiling_results/decode_t2i_impact_matrix}"
mkdir -p "$output_root"

# Repeat counts estimated from the measured warm scaling curves so that the
# sequence of T2I requests has approximately the same duration as one decode.
# Keys are implementation:requested_generator_sm.
declare -A repeats=(
  [eager:154]=31
  [eager:146]=27
  [eager:130]=27
  [eager:114]=25
  [graph:154]=27
  [graph:146]=16
  [graph:130]=10
  [graph:114]=7
)

run_one() {
  local implementation="$1"
  local requested_generator="$2"
  local generator_repeats="$3"
  local log="$4"
  export COSMOS_GENERATOR_SM_COUNT="$requested_generator"
  export COSMOS_GENERATOR_REPEATS="$generator_repeats"
  echo
  echo "=== implementation=$implementation requested_generator=$requested_generator t2i_repeats=$generator_repeats ==="
  python -m profiling_tools.runners.colocate_decode_t2v 2>&1 | tee "$log"
}

field_from_log() {
  local field="$1"
  local log="$2"
  awk -v wanted="$field" '
    index($0, "  " wanted "=") == 1 {
      value=$0
      sub("^  " wanted "=", "", value)
      sub(" s$", "", value)
    }
    END {if (value == "") exit 1; print value}
  ' "$log"
}

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
      echo "Unknown implementation: $implementation" >&2
      exit 2
      ;;
  esac

  for requested_generator in $requested_generator_values; do
    key="$implementation:$requested_generator"
    if [[ "$implementation" == "static_eager" && -z "${repeats[$key]+set}" ]]; then
      calibration_log="$output_root/${implementation}_requested_generator_sm$(printf %03d "$requested_generator")_repeat01_calibration_${run_tag}.log"
      run_one "$implementation" "$requested_generator" 1 "$calibration_log"
      decode_alone="$(field_from_log decode_alone "$calibration_log")"
      t2i_once="$(field_from_log generator_alone "$calibration_log")"
      repeats[$key]="$(awk -v decode="$decode_alone" -v t2i="$t2i_once" 'BEGIN {n=int(decode/t2i+0.5); print (n < 1 ? 1 : n)}')"
      echo "Calibrated $key: decode=${decode_alone}s t2i_once=${t2i_once}s repeats=${repeats[$key]}"
    elif [[ -z "${repeats[$key]+set}" ]]; then
      echo "No T2I repeat estimate for $key" >&2
      exit 2
    fi
    generator_repeats="${repeats[$key]}"
    log="$output_root/${implementation}_requested_generator_sm$(printf %03d "$requested_generator")_repeat$(printf %02d "$generator_repeats")_${run_tag}.log"
    run_one "$implementation" "$requested_generator" "$generator_repeats" "$log"
  done
done
