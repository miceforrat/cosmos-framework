#!/usr/bin/env python3
"""Measure standalone Edge Reasoner or Generator GPU-memory residency/peak."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def gpu_used_mib(gpu_index: int) -> int:
    output = subprocess.check_output(
        ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def request_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def request_form(url: str, payload: dict, timeout: float) -> bytes:
    request = urllib.request.Request(
        url,
        data=urllib.parse.urlencode(payload).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def wait_ready(url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server did not become ready within {timeout:.0f}s")


def stop_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=("reasoner", "generator", "generator_t2v"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int)
    parser.add_argument("--result-dir", type=Path, default=Path("profiling_results/gate_0_5_memory"))
    parser.add_argument("--ready-seconds", type=float, default=3.0)
    parser.add_argument("--sample-period", type=float, default=0.1)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--image-size", default="256x256")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--video-size", default="832x480")
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--t2v-input",
        type=Path,
        default=Path("outputs/local_agentic_t2v/rich02/inputs/t2v_upsampled.json"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    vllm = root / ".runtime/vllm_omni_tf517/venv/bin/vllm"
    if not vllm.exists():
        raise FileNotFoundError(f"persistent runtime missing: {vllm}")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    port = args.port or (8010 if args.workload == "reasoner" else 8011)
    prefix = args.result_dir / args.workload
    log_path = prefix.with_suffix(".server.log")
    samples_path = prefix.with_suffix(".memory.csv")
    response_path = prefix.with_suffix(".response.json")
    summary_path = prefix.with_suffix(".summary.json")

    baseline_mib = gpu_used_mib(args.gpu)
    if baseline_mib > 512:
        raise RuntimeError(f"GPU {args.gpu} already uses {baseline_mib} MiB; use an idle GPU")

    common = [str(vllm), "serve", "nvidia/Cosmos3-Edge", "--host", "127.0.0.1", "--port", str(port)]
    if args.workload == "reasoner":
        command = common + [
            "--tensor-parallel-size", "1", "--max-num-seqs", "1", "--max-model-len", "2048",
            "--kv-cache-memory-bytes", "1G", "--no-enable-prefix-caching", "--enforce-eager",
        ]
        endpoint = "/v1/chat/completions"
        payload = {
            "model": "nvidia/Cosmos3-Edge",
            "messages": [{"role": "user", "content": "Describe a safe plan for a robot to move a cup onto a shelf."}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
    elif args.workload == "generator":
        command = common + ["--omni", "--no-guardrails", "--init-timeout", "1800", "--stage-init-timeout", "1800"]
        endpoint = "/v1/images/generations"
        payload = {
            "model": "nvidia/Cosmos3-Edge", "prompt": "A red robot repairs a bicycle in a bright workshop.",
            "n": 1, "size": args.image_size, "response_format": "b64_json", "seed": 0,
            "num_inference_steps": args.steps or 2,
        }
    else:
        command = common + ["--omni", "--no-guardrails", "--init-timeout", "1800", "--stage-init-timeout", "1800"]
        endpoint = "/v1/videos/sync"
        t2v_prompt = "A red robot repairs a bicycle in a bright workshop while the camera slowly moves around them."
        if args.t2v_input.exists():
            t2v_prompt = json.loads(args.t2v_input.read_text())["prompt"]
        payload = {
            "model": "nvidia/Cosmos3-Edge",
            "prompt": t2v_prompt,
            "size": args.video_size,
            "num_frames": str(args.num_frames),
            "fps": str(args.fps),
            "num_inference_steps": str(args.steps or 35),
            "guidance_scale": "6.0",
            "seed": "0",
        }

    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), LD_LIBRARY_PATH="", VLLM_USE_DEEP_GEMM="0")
    samples: list[tuple[float, int, str]] = []
    phase = "startup"
    stop_sampling = threading.Event()

    def sample_loop() -> None:
        while not stop_sampling.is_set():
            try:
                samples.append((time.time(), gpu_used_mib(args.gpu), phase))
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
            stop_sampling.wait(args.sample_period)

    response: dict = {}
    video_response: bytes | None = None
    request_start = request_end = 0.0
    with log_path.open("w") as log:
        log.write("command: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, text=True)
        sampler = threading.Thread(target=sample_loop, daemon=True)
        sampler.start()
        try:
            wait_ready(f"http://127.0.0.1:{port}/health", process, args.startup_timeout)
            phase = "ready"
            time.sleep(args.ready_seconds)
            phase = "request"
            request_start = time.time()
            if args.workload == "generator_t2v":
                video_response = request_form(
                    f"http://127.0.0.1:{port}{endpoint}", payload, args.startup_timeout
                )
            else:
                response = request_json(f"http://127.0.0.1:{port}{endpoint}", payload, args.startup_timeout)
            request_end = time.time()
            phase = "post_request"
            time.sleep(1.0)
        finally:
            stop_sampling.set()
            sampler.join(timeout=2)
            stop_process_group(process)

    if args.workload == "generator" and response.get("data"):
        encoded = response["data"][0].pop("b64_json", None)
        if encoded:
            prefix.with_suffix(".png").write_bytes(base64.b64decode(encoded))
    if video_response is not None:
        prefix.with_suffix(".mp4").write_bytes(video_response)
        response = {"content_type": "video/mp4", "bytes": len(video_response)}
    response_path.write_text(json.dumps(response, indent=2) + "\n")
    with samples_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("unix_seconds", "gpu_used_mib", "phase"))
        writer.writerows(samples)

    ready = [used for _, used, item_phase in samples if item_phase == "ready"]
    during = [used for _, used, item_phase in samples if item_phase == "request"]
    if not ready or not during:
        raise RuntimeError("memory sampler did not capture ready/request windows")
    ready_median = sorted(ready)[len(ready) // 2]
    summary = {
        "workload": args.workload, "gpu": args.gpu, "command": command, "baseline_mib": baseline_mib,
        "ready_mib_median": ready_median, "ready_mib_max": max(ready), "request_peak_mib": max(during),
        "request_peak_above_ready_mib": max(during) - ready_median,
        "request_wall_seconds": request_end - request_start, "sample_period_seconds": args.sample_period,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
