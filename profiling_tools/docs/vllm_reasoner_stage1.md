# 阶段 1：Reasoner-only vLLM 验证

## 目标与边界

本阶段只加载 Cosmos3-Edge 的 Reasoner tower，使用固定 text-only 请求验证：

- vLLM eager 与 CUDA Graph 均能完成推理；
- 相同 greedy 输入的输出文本 hash 一致；
- 启动日志给出 Paged KV block/cache 证据；
- graph 模式的稳态 Nsys trace 中存在 `cudaGraphLaunch`；
- 保存版本、attention backend、显存配置、TTFT、E2E 和 decode token rate。

text-only 是刻意选择：它排除视觉编码器，使这一步专门验证 language prefill/decode。视觉输入在后续单独补测。

当前 Cosmos 主 `.venv` 没有安装 vLLM。本阶段使用 `uvx` 隔离环境，不修改已经冻结的 `.venv`。启动脚本默认使用 CUDA 13.0 后端、`vllm==0.26.0` 和 `transformers==5.5.3`。vLLM 0.23.0 只有 Nano/Super understanding tower 的 `Cosmos3ForConditionalGeneration`，不能加载 `cosmos3_edge`；vLLM 0.26.0 已确认包含独立的 `Cosmos3EdgeForConditionalGeneration`。Cosmos3 Edge processor 依赖 Transformers 5.5.3 中的 Qwen3-VL video processor API；不能放宽到当前 5.20.1，否则会因 `get_image_size` 被移除而导入失败。CUDA 13 wheel 依赖 `libcudart.so.13`，不能与 `--torch-backend=cu128` 混用；RTX 5090 当前的 580 驱动可运行 CUDA 13 用户态 runtime。

这里必须使用 vLLM 0.26+ 自带的原生 Edge 实现，不能再加载本仓库的 `packages/vllm-cosmos3`：后者是旧的 Nano/Super Qwen3-VL understanding-tower 适配，强行把 Edge 配置覆盖成该架构会导致 `Qwen3VLConfig` 与 `Cosmos3EdgeConfig` 类型冲突。启动命令也不再使用 `--hf-overrides`。如需试验更新版本，可设置 `VLLM_REASONER_VLLM_VERSION`，但正式结果必须记录实际版本。

### 为什么不改用 Certified NIM / Nano

- 当前 NVIDIA 文档中的 Cosmos3 Reasoner NIM 只公开 `nano` 和 `super` 两种 variant，没有 Edge variant；因此它不能作为本实验 Edge Reasoner 的直接后端。
- NVIDIA 已公开 `nvidia/Cosmos3-Edge` 单 GPU vLLM serving benchmark，并给出 Edge 的原生 `vllm serve nvidia/Cosmos3-Edge` 命令。
- 当前实验一直以 4B Edge 为对象，GPU 是 32 GB RTX 5090。换 Nano 会同时改变参数规模、显存占用和 scaling/共置特征，无法与此前 Edge 数据直接比较。
- NIM 的 API、启动参数和运行时证据可作为验收设计参考，但 Stage 1 的实现采用原生 vLLM Edge 路径。

## 公共准备

```bash
docker exec -it cosmos3-clean bash
cd /workspace/cosmos-framework
source .venv/bin/activate
export LD_LIBRARY_PATH=''
export CUDA_VISIBLE_DEVICES=0
```

首次运行 `uvx` 会解析/下载 vLLM 环境，但 Cosmos3-Edge checkpoint 已位于容器挂载的 Hugging Face cache 中。
启动脚本默认设置 `UV_HTTP_TIMEOUT=600`、`UV_HTTP_RETRIES=10` 和
`UV_CONCURRENT_DOWNLOADS=4`。若下载中途超时，直接重跑相同命令；已完成的 wheel
会从 uv cache 复用，不会修改主 `.venv`。

## 1. Eager server

终端 A：

```bash
cd /workspace/cosmos-framework
export CUDA_VISIBLE_DEVICES=0
bash profiling_tools/scripts/serve_vllm_reasoner_stage1.sh eager
```

等待日志出现服务 ready。终端 B：

```bash
docker exec -it cosmos3-clean bash
cd /workspace/cosmos-framework
source .venv/bin/activate

python -m profiling_tools.runners.validate_vllm_reasoner \
  --mode eager \
  --warmup 1 \
  --repeats 3 \
  --max-tokens 128 \
  --output profiling_results/vllm_reasoner_stage1/eager.json
```

保存显存状态：

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  > profiling_results/vllm_reasoner_stage1/eager_gpu_memory.csv
```

结束终端 A 的 server 后再启动 graph server，避免两个 engine 同时占用显存。

## 2. CUDA Graph server

终端 A：

```bash
cd /workspace/cosmos-framework
export CUDA_VISIBLE_DEVICES=0
bash profiling_tools/scripts/serve_vllm_reasoner_stage1.sh graph
```

终端 B：

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate

python -m profiling_tools.runners.validate_vllm_reasoner \
  --mode graph \
  --warmup 1 \
  --repeats 3 \
  --max-tokens 128 \
  --output profiling_results/vllm_reasoner_stage1/graph.json

nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  > profiling_results/vllm_reasoner_stage1/graph_gpu_memory.csv
```

## 3. 输出正确性和绝对时间

```bash
python -m profiling_tools.analysis.compare_vllm_reasoner_stage1 \
  --eager profiling_results/vllm_reasoner_stage1/eager.json \
  --graph profiling_results/vllm_reasoner_stage1/graph.json
```

通过条件：

- 两种模式各自的 3 次 measured 请求 hash 唯一；
- eager 与 graph 的 `output_hash_equal` 为 `True`；
- 输出内容满足固定 prompt，而不是空文本或报错信息；
- JSON 中存在 TTFT、E2E、completion token 和 decode token rate。

这里比较的是可观察输出的确定性，不是对所有输入的形式化数值等价证明。

## 4. Paged KV 和 backend 证据

```bash
grep -Ei \
  'vllm|attention backend|paged|kv cache|gpu blocks|num_gpu_blocks|cudagraph|cuda graph|enforce.eager|compilation' \
  profiling_results/vllm_reasoner_stage1/server_eager.log \
  profiling_results/vllm_reasoner_stage1/server_graph.log \
  | tee profiling_results/vllm_reasoner_stage1/runtime_evidence.txt
```

验收时至少记录：

- vLLM 版本；
- model architecture 为 vLLM 原生识别的 Cosmos3 Edge architecture（不得是旧适配伪装的 `Qwen3VLForConditionalGeneration`）；
- attention backend；
- GPU KV cache/block 数量或 KV cache capacity；
- eager 明确禁用 graph；
- graph 的最终 cudagraph mode，以及任何 downgrade/fallback 警告。

日志中出现 vLLM 并不足以单独证明 Paged KV；需要 KV block/cache 的运行时记录。

## 5. Nsys 验证 graph replay

先关闭普通 graph server。终端 A 使用延迟采集重新启动：

```bash
cd /workspace/cosmos-framework
export CUDA_VISIBLE_DEVICES=0
export NSYS_OUTPUT=profiling_results/vllm_reasoner_stage1/nsys_graph
export NSYS_DELAY_SECONDS=300
export NSYS_DURATION_SECONDS=120

bash profiling_tools/scripts/serve_vllm_reasoner_stage1.sh graph
```

默认前 300 秒不采集，用于创建隔离环境、加载模型和 capture graph。根据实际 ready 时间调整 `NSYS_DELAY_SECONDS`。在 Nsys 开始采集后的 120 秒窗口内，从终端 B 执行：

```bash
python -m profiling_tools.runners.validate_vllm_reasoner \
  --mode graph \
  --warmup 2 \
  --repeats 5 \
  --max-tokens 128 \
  --output profiling_results/vllm_reasoner_stage1/graph_nsys.json
```

采集完成并停止 server 后检查 CUDA API：

```bash
nsys stats \
  --report cuda_api_sum \
  profiling_results/vllm_reasoner_stage1/nsys_graph.nsys-rep \
  | tee profiling_results/vllm_reasoner_stage1/nsys_cuda_api_sum.txt

grep -E 'cudaGraphLaunch|cudaGraphInstantiate|cudaStreamSynchronize|cudaDeviceSynchronize' \
  profiling_results/vllm_reasoner_stage1/nsys_cuda_api_sum.txt
```

通过条件不是仅出现 graph capture/instantiate，而是 measured 请求窗口中存在大量 `cudaGraphLaunch`。如需区分启动 capture 与稳态 replay，再导出 SQLite 并按 API 时间戳对齐请求窗口。

## 6. 文件清单

```text
profiling_results/vllm_reasoner_stage1/
├── server_eager.log
├── server_graph.log
├── eager.json
├── graph.json
├── eager_gpu_memory.csv
├── graph_gpu_memory.csv
├── runtime_evidence.txt
├── nsys_graph.nsys-rep
├── graph_nsys.json
└── nsys_cuda_api_sum.txt
```

完成上述验收后，阶段 2 才开始在同一个 Python 环境中同时加载 ReasonerBackend 与 GeneratorBackend。
