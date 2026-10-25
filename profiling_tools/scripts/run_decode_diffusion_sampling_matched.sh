#!/usr/bin/env bash
set -euo pipefail

backend=${1:-}
capture=${2:-plain}
generator_mode=${MATCHED_GENERATOR_MODE:-eager}
if [[ "$backend" != native && "$backend" != vllm ]]; then
  echo "usage: $0 {native|vllm} [plain|nsys|nsys-metrics]" >&2
  exit 2
fi
if [[ "$capture" != plain && "$capture" != nsys && "$capture" != nsys-metrics ]]; then
  echo "usage: $0 {native|vllm} [plain|nsys|nsys-metrics]" >&2
  exit 2
fi
if [[ "$generator_mode" != eager && "$generator_mode" != compile ]]; then
  echo "MATCHED_GENERATOR_MODE must be eager or compile" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

reasoner_input="$repo_root/outputs/local_agentic_t2v/rich02/inputs/upsample_reasoner.json"
t2v_input="$repo_root/outputs/local_agentic_t2v/rich02/inputs/t2v_upsampled.json"
native_sample_args="$repo_root/outputs/local_agentic_t2v/rich02/stages/02_generator/t2v_upsampled/sample_args.json"
result_root="$repo_root/profiling_results/decode_diffusion_sampling_matched"
mkdir -p "$result_root/nsys"

for required in "$reasoner_input" "$t2v_input" "$native_sample_args"; do
  if [[ ! -f "$required" ]]; then
    echo "missing matched workload file: $required" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export LD_LIBRARY_PATH=''
export CUDA_DEVICE_MAX_CONNECTIONS=8
export COSMOS_MATCHED_GENERATOR_SAMPLE_ARGS="$native_sample_args"

run_profiled() {
  local output_prefix=$1
  shift
  if [[ "$capture" == nsys || "$capture" == nsys-metrics ]]; then
    export COSMOS_NSYS_CAPTURE=1
    local profile_name="$output_prefix"
    local -a metrics_args=()
    if [[ "$capture" == nsys-metrics ]]; then
      profile_name="${output_prefix}_metrics"
      metrics_args+=(
        --gpu-metrics-devices="${NSYS_GPU_METRICS_DEVICES:-0}"
        --gpu-metrics-frequency="${NSYS_GPU_METRICS_FREQUENCY:-10000}"
      )
      if [[ -n "${NSYS_GPU_METRICS_SET:-}" ]]; then
        metrics_args+=(--gpu-metrics-set="$NSYS_GPU_METRICS_SET")
      fi
    fi
    nsys profile \
      --trace=cuda-sw,nvtx,osrt \
      --cuda-graph-trace=node \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      --sample=none \
      --cpuctxsw=none \
      --force-overwrite=true \
      "${metrics_args[@]}" \
      --output "$result_root/nsys/$profile_name" \
      "$@"
  else
    unset COSMOS_NSYS_CAPTURE
    "$@"
  fi
}

if [[ "$backend" == native ]]; then
  source "$repo_root/.venv/bin/activate"
  export COSMOS_REASONER_CUDA_GRAPH=1
  export COSMOS_REASONER_CUDA_GRAPH_PERSISTENT=1
  export COSMOS_REASONER_FIXED_LENGTH_DECODE=1
  export COSMOS_REASONER_VALIDATE_TOKENS=1
  export COSMOS_GENERATOR_SM_COUNT=120
  export COSMOS_REASONER_REPEATS=2
  export COSMOS_GENERATOR_REPEATS=1
  export COSMOS_WARMUP_RUNS=1
  export COSMOS_COLOCATE_GENERATOR_REGION=diffusion_sampling
  export COSMOS_GENERATOR_WORKLOAD=t2v
  export COSMOS_REASONER_INPUT="$reasoner_input"
  export COSMOS_GENERATOR_INPUT="$t2v_input"
  unset COSMOS_FULL_SM_DECODE_ONLY
  unset COSMOS_FULL_SM_GENERATOR_ONLY
  unset COSMOS_FULL_SM_SERIAL_BASELINE_ONLY
  unset COSMOS_GC_DECODE_ONLY

  log="$result_root/native_graph_g120_r50_tok1350x2_${capture}.log"
  run_profiled "native_graph_g120_r50_tok1350x2" \
    python -m profiling_tools.runners.colocate_decode_t2v 2>&1 | tee "$log"
else
  source "$repo_root/.runtime/vllm_omni_tf517/venv/bin/activate"
  export VLLM_ENABLE_V1_MULTIPROCESSING=0
  export VLLM_USE_DEEP_GEMM=0

  mode_tag=""
  if [[ "$generator_mode" == compile ]]; then
    mode_tag="_omni_compile"
  fi
  output="$result_root/vllm_graph${mode_tag}_g120_r50_tok1350x2_${capture}.json"
  log="$result_root/vllm_graph${mode_tag}_g120_r50_tok1350x2_${capture}.log"
  run_profiled "vllm_graph${mode_tag}_g120_r50_tok1350x2" \
    python -m profiling_tools.runners.concurrent_reasoner_generator \
      --reasoner-mode graph \
      --generator-mode "$generator_mode" \
      --generator-cache on \
      --colocate-region diffusion_sampling \
      --reasoner-input "$reasoner_input" \
      --reasoner-max-tokens 1350 \
      --reasoner-runs-per-generator 2 \
      --reasoner-fixed-tokens \
      --generator-sm-count 120 \
      --isolate-ir-op-priority \
      --full-sm-warmups 1 \
      --full-sm-repeats 1 \
      --warmups 1 \
      --alone-repeats 1 \
      --concurrent-repeats 1 \
      --t2v-input "$t2v_input" \
      --output "$output" 2>&1 | tee "$log"
fi
