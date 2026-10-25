#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mode=${1:?Usage: serve_vllm_reasoner_stage1.sh eager|compile|graph}
case "$mode" in
  eager|compile|graph) ;;
  *) echo "Mode must be eager, compile, or graph" >&2; exit 2 ;;
esac

model=${VLLM_REASONER_MODEL:-nvidia/Cosmos3-Edge}
port=${VLLM_REASONER_PORT:-8000}
max_model_len=${VLLM_REASONER_MAX_MODEL_LEN:-4096}
gpu_memory_utilization=${VLLM_REASONER_GPU_MEMORY_UTILIZATION:-0.70}
vllm_version=${VLLM_REASONER_VLLM_VERSION:-0.31.0}
transformers_version=${VLLM_REASONER_TRANSFORMERS_VERSION:-5.17.0}
prefix_caching=${VLLM_REASONER_PREFIX_CACHING:-1}
result_dir=${VLLM_STAGE1_RESULT_DIR:-profiling_results/vllm_reasoner_stage1}
mkdir -p "$result_dir"
log_path="$result_dir/server_${mode}.log"

export VLLM_USE_DEEP_GEMM=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# uv's default network timeout is too short for the large vLLM dependency set
# on the current host.  Downloads are cached, so a retry resumes from cache.
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10}
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}

cmd=(
  uvx
  --torch-backend=cu130
  --with "transformers==${transformers_version}"
  "vllm@${vllm_version}"
  serve "$model"
  --tensor-parallel-size 1
  --mm-encoder-tp-mode data
  --async-scheduling
  --allowed-local-media-path "$(pwd)"
  --media-io-kwargs '{"video":{"num_frames":-1}}'
  --max-model-len "$max_model_len"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --port "$port"
)

if [[ "$prefix_caching" == 0 ]]; then
  cmd+=(--no-enable-prefix-caching)
elif [[ "$prefix_caching" != 1 ]]; then
  echo "VLLM_REASONER_PREFIX_CACHING must be 0 or 1" >&2
  exit 2
fi

if [[ "$mode" == eager ]]; then
  cmd+=(--enforce-eager)
elif [[ "$mode" == compile ]]; then
  cmd+=(--compilation-config '{"mode":3,"cudagraph_mode":"NONE"}')
else
  cmd+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}')
fi

{
  echo "mode=$mode"
  echo "model=$model"
  echo "vllm_version=$vllm_version"
  echo "transformers_version=$transformers_version"
  echo "prefix_caching=$prefix_caching"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "max_model_len=$max_model_len"
  echo "gpu_memory_utilization=$gpu_memory_utilization"
  echo "command: ${cmd[*]}"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader
} | tee "$log_path"

if [[ -n ${NSYS_OUTPUT:-} ]]; then
  nsys_delay=${NSYS_DELAY_SECONDS:-300}
  nsys_duration=${NSYS_DURATION_SECONDS:-120}
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --delay="$nsys_delay" \
    --duration="$nsys_duration" \
    --force-overwrite=true \
    --output="$NSYS_OUTPUT" \
    "${cmd[@]}" 2>&1 | tee -a "$log_path"
else
  "${cmd[@]}" 2>&1 | tee -a "$log_path"
fi
