# Gate 0.5：Edge 双引擎显存预算

本阶段仍然分别启动模型，不测试同 PID、并发或 Green Context。目标是先得到下一阶段可用的显存预算：服务 ready 后的驻留显存、一次固定请求期间的峰值，以及请求相对 ready 驻留的额外峰值。

Reasoner 使用 eager、`max_num_seqs=1`、`max_model_len=2048`，并通过 `--kv-cache-memory-bytes 1G` 固定 KV cache。不能用常规的 `gpu_memory_utilization` 自动规划，否则 Reasoner 会占用固定比例的整卡空闲显存，无法估计双引擎共存。

## 执行

容器内、GPU 0 空闲时：

```bash
cd /workspace/cosmos-framework
source .runtime/vllm_omni_tf517/venv/bin/activate

python -m profiling_tools.runners.measure_gate_0_5_memory reasoner
python -m profiling_tools.runners.measure_gate_0_5_memory generator

# 与 rich02 相同的代表性 T2V：832×480、121 帧、24 FPS、35 steps
python -m profiling_tools.runners.measure_gate_0_5_memory generator_t2v \
  --t2v-input outputs/local_agentic_t2v/rich02/inputs/t2v_upsampled.json \
  --video-size 832x480 \
  --num-frames 121 \
  --fps 24 \
  --steps 35
```

`generator` 只做 256×256、2-step T2I，用于测最小初始化与请求峰值；
`generator_t2v` 则复现 rich02 的 832×480、121 帧、24 FPS、35-step T2V，
用于补充代表性运行期峰值。

输出位于 `profiling_results/gate_0_5_memory/`，包括每个 workload 的 `summary.json`、`memory.csv`、`server.log` 和 `response.json`，Generator 还保存 `generator.png`。

脚本要求目标 GPU 启动前显存占用不超过 512 MiB，且退出时只终止自己启动的进程组。

## Gate 1 前的判断

最小必要预算不是两个权重日志的简单相加，而至少是：

```text
reasoner ready residency
+ generator ready residency
+ 两边运行期额外峰值中可能同时出现的部分
+ CUDA context / allocator / 后续 graph pool 的安全余量
```

Gate 0.5 通过只说明 32 GiB 上存在合理的共存空间；同一 PID 构造是否成功仍由 Gate 1 验证。

## 2026-10-20 实测结果

RTX 5090 32 GiB、GPU 启动占用 0 MiB：

| workload | ready median | request peak | peak above ready | request |
|---|---:|---:|---:|---:|
| Edge Reasoner eager，1 GiB KV，128-token decode | 10,048 MiB | 10,050 MiB | 2 MiB | 2.157 s |
| Edge Generator，256×256，2-step T2I | 9,846 MiB | 9,862 MiB | 16 MiB | 0.515 s |
| Edge Generator，832×480，121 帧，35-step T2V | 9,847 MiB | 15,755 MiB | 5,908 MiB | 34.456 s |

两边 ready 驻留之和为 19,894 MiB（约 19.43 GiB），相对该卡可见的
32,110 MiB 还剩 12,216 MiB（约 11.93 GiB）。因此最小配置具备进入 Gate 1
同 PID 串行构造的显存空间。

这个余量不能外推为代表性 T2V 的安全上界：当前 Generator 请求分辨率和 step
都刻意设得很小，也尚未包含两边 CUDA Graph pool。Gate 1 第一版继续保持 eager
和最小 Generator 请求；代表性 T2V 峰值在双引擎骨架跑通后补测。

代表性 T2V 已随后补测。Reasoner ready 与 T2V 请求峰值合计 25,803 MiB
（约 25.20 GiB），相对 32,110 MiB 可见显存还剩 6,307 MiB（约 6.16 GiB）。
这支持先进行 eager Gate 1，但仍需实测同进程 allocator 和初始化峰值，不能仅凭加法判定通过。
