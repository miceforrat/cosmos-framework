#!/usr/bin/env python3
"""Gate 1: construct Edge Reasoner and Generator in one PID, then run serially."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import statistics
import os
import subprocess
import time
from pathlib import Path

# These must be set before importing vLLM. Gate 1 explicitly forbids EngineCore MP.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")


def cuda_context() -> str | None:
    driver = ctypes.CDLL("libcuda.so.1")
    if driver.cuInit(0) != 0:
        return None
    context = ctypes.c_void_p()
    if driver.cuCtxGetCurrent(ctypes.byref(context)) != 0 or not context.value:
        return None
    return hex(context.value)


def process_snapshot() -> dict:
    pid = os.getpid()
    ps = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,pgid=,stat=,comm=,args="], text=True
    ).splitlines()
    descendants: set[int] = set()
    rows: list[dict] = []
    changed = True
    while changed:
        changed = False
        parents = descendants | {pid}
        for line in ps:
            fields = line.strip().split(None, 5)
            if len(fields) < 6:
                continue
            child_pid, parent_pid = int(fields[0]), int(fields[1])
            if parent_pid in parents and child_pid not in descendants:
                descendants.add(child_pid)
                changed = True
    for line in ps:
        fields = line.strip().split(None, 5)
        if len(fields) == 6 and int(fields[0]) in descendants:
            rows.append(
                {"pid": int(fields[0]), "ppid": int(fields[1]), "pgid": int(fields[2]),
                 "stat": fields[3], "comm": fields[4], "args": fields[5]}
            )
    try:
        gpu_lines = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError:
        gpu_lines = []
    gpu_processes = []
    for line in gpu_lines:
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 3:
            gpu_processes.append({"pid": int(fields[0]), "name": fields[1], "used_mib": int(fields[2])})
    return {"pid": pid, "children": rows, "gpu_processes": gpu_processes, "cuda_context": cuda_context()}


def create_reasoner(
    *,
    borrow_omni_parallel_state: bool,
    max_model_len: int,
    execution_mode: str,
    gpu_memory_utilization: float = 0.70,
):
    if borrow_omni_parallel_state:
        from profiling_tools.core.omni_shared_parallel import install_reasoner_context_groups_from_omni

        install_reasoner_context_groups_from_omni()
    from vllm import LLM

    if execution_mode not in {"eager", "compile", "graph"}:
        raise ValueError(f"Unsupported Reasoner execution mode: {execution_mode}")
    if execution_mode == "eager":
        execution_kwargs = {"enforce_eager": True}
    elif execution_mode == "compile":
        execution_kwargs = {
            "enforce_eager": False,
            "compilation_config": {"mode": 3, "cudagraph_mode": "NONE"},
        }
    else:
        execution_kwargs = {
            "enforce_eager": False,
            "compilation_config": {"cudagraph_mode": "FULL_AND_PIECEWISE"},
        }

    return LLM(
        model="nvidia/Cosmos3-Edge",
        tensor_parallel_size=1,
        max_num_seqs=1,
        max_model_len=max_model_len,
        # vLLM performs this admission check before honoring the explicit
        # kv_cache_memory_bytes limit. Keep it below the free-memory ratio
        # when the Omni generator is already resident (23.59 / 31.36 GiB).
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_memory_bytes=1 << 30,
        enable_prefix_caching=False,
        disable_log_stats=True,
        **execution_kwargs,
    )


def create_generator(
    *, share_reasoner_parallel_state: bool, cache_mode: str, execution_mode: str
):
    if share_reasoner_parallel_state:
        from profiling_tools.core.omni_shared_parallel import install_single_rank_shared_parallel_state

        install_single_rank_shared_parallel_state()
    from vllm_omni import Omni

    if cache_mode not in {"on", "off"}:
        raise ValueError(f"Unsupported Generator cache mode: {cache_mode}")
    if execution_mode not in {"eager", "compile"}:
        raise ValueError(f"Unsupported Generator execution mode: {execution_mode}")
    cache_kwargs = (
        {
            "cache_backend": "sea_cache",
            "cache_config": {
                "sea_threshold": 0.25,
                "sea_residual_order": 1,
                "sea_max_consecutive_cached": 2,
                "sea_power_exp": 3.0,
            },
        }
        if cache_mode == "on"
        else {"cache_backend": "none"}
    )

    return Omni(
        model="nvidia/Cosmos3-Edge",
        model_config={"guardrails": False},
        max_num_seqs=1,
        enforce_eager=execution_mode == "eager",
        init_timeout=1800,
        stage_init_timeout=1800,
        log_stats=False,
        **cache_kwargs,
    )


def run_reasoner(
    reasoner,
    *,
    representative: bool,
    max_tokens: int,
    fixed_tokens: bool = False,
    seed: int = 0,
    prompt_override: str | None = None,
) -> dict:
    from vllm import SamplingParams

    if prompt_override is not None:
        workload = "prompt_override"
        messages = [{"role": "user", "content": prompt_override}]
    elif representative:
        from profiling_tools.core.official_upsampler_workload import (
            OFFICIAL_T2I_SYSTEM_PROMPT,
            render_official_t2i_upsampler_prompt,
        )

        workload = "official_hf_t2i_upsampler"
        messages = [
            {"role": "system", "content": OFFICIAL_T2I_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": render_official_t2i_upsampler_prompt(
                    "A red robot repairs a bicycle in a bright workshop while the camera slowly moves around them."
                ),
            },
        ]
    else:
        workload = "smoke"
        messages = [
            {"role": "user", "content": "Describe a safe plan for a robot to move a cup onto a shelf."}
        ]
    formatted_prompt = reasoner.get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_token_ids = reasoner.get_tokenizer().encode(formatted_prompt)
    prompt_token_count = len(prompt_token_ids)
    prompt_sha256 = hashlib.sha256(formatted_prompt.encode("utf-8")).hexdigest()
    prompt_token_ids_sha256 = hashlib.sha256(
        json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    start = time.perf_counter()
    outputs = reasoner.generate(
        [formatted_prompt],
        SamplingParams(
            temperature=0.0,
            seed=seed,
            max_tokens=max_tokens,
            min_tokens=max_tokens if fixed_tokens else 0,
            ignore_eos=fixed_tokens,
        ),
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - start
    text = outputs[0].outputs[0].text
    return {
        "workload": workload,
        "prompt_tokens": prompt_token_count,
        "formatted_prompt_sha256": prompt_sha256,
        "prompt_token_ids_sha256": prompt_token_ids_sha256,
        "seconds": elapsed,
        "text": text,
        "completion_tokens": len(outputs[0].outputs[0].token_ids),
        "decode_tokens_per_second": len(outputs[0].outputs[0].token_ids) / elapsed,
        "finish_reason": outputs[0].outputs[0].finish_reason,
        "fixed_tokens": fixed_tokens,
        "seed": seed,
    }



def run_reasoner_trials(
    reasoner,
    *,
    representative: bool,
    max_tokens: int,
    warmups: int,
    repeats: int,
) -> dict:
    if warmups < 0:
        raise ValueError("--reasoner-warmups must be non-negative")
    if repeats < 1:
        raise ValueError("--reasoner-repeats must be at least one")

    warmup_results = []
    measured_results = []
    for index in range(warmups):
        print(f"Reasoner warmup {index + 1}/{warmups}", flush=True)
        warmup_results.append(
            run_reasoner(reasoner, representative=representative, max_tokens=max_tokens)
        )
    for index in range(repeats):
        print(f"Reasoner measured {index + 1}/{repeats}", flush=True)
        measured_results.append(
            run_reasoner(reasoner, representative=representative, max_tokens=max_tokens)
        )

    seconds = [item["seconds"] for item in measured_results]
    result = dict(measured_results[-1])
    result.update(
        {
            "warmups": warmup_results,
            "measured": measured_results,
            "measured_seconds": seconds,
            "median_seconds": statistics.median(seconds),
        }
    )
    return result
def run_generator(generator, *, representative: bool, t2v_input: Path) -> dict:
    import torch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    prompt_text = "A red robot repairs a bicycle in a bright workshop."
    negative_prompt = ""
    height = width = 256
    num_frames = 1
    num_inference_steps = 2
    guidance_scale = 5.0
    flow_shift = None
    seed = 0
    native_sample_args_path = os.environ.get("COSMOS_MATCHED_GENERATOR_SAMPLE_ARGS")
    native_sample_args = None
    if representative:
        if native_sample_args_path:
            native_sample_args = json.loads(
                Path(native_sample_args_path).read_text(encoding="utf-8")
            )
            prompt_text = native_sample_args["prompt"]
            negative_prompt = native_sample_args["negative_prompt"]
            resolution_payload = json.loads(prompt_text).get("resolution", {})
            height = int(resolution_payload.get("H", 480))
            width = int(resolution_payload.get("W", 832))
            num_frames = int(native_sample_args["num_frames"])
            num_inference_steps = int(native_sample_args["num_steps"])
            guidance_scale = float(native_sample_args["guidance"])
            flow_shift = float(native_sample_args["shift"])
            seed = int(native_sample_args["seed"])
        else:
            if t2v_input.exists():
                prompt_text = json.loads(t2v_input.read_text(encoding="utf-8"))["prompt"]
            height, width = 480, 832
            num_frames = 121
            num_inference_steps = 35
            guidance_scale = 6.0

    canonical_workload = {
        "prompt": prompt_text,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "flow_shift": flow_shift,
        "seed": seed,
    }
    canonical_json = json.dumps(
        canonical_workload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    prompt = {
        "prompt": prompt_text,
        "negative_prompt": negative_prompt,
        "modalities": ["video"],
    }
    params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        extra_args=({"flow_shift": flow_shift} if flow_shift is not None else {}),
    )
    start = time.perf_counter()
    outputs = generator.generate(prompt, params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    first = outputs[0]
    return {
        "workload": (
            "matched_native_t2v_sample_args"
            if native_sample_args is not None
            else "representative_t2v"
            if representative
            else "smoke"
        ),
        "native_sample_args_path": native_sample_args_path,
        "canonical_workload": canonical_workload,
        "canonical_workload_sha256": hashlib.sha256(
            canonical_json.encode("utf-8")
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "negative_prompt_sha256": hashlib.sha256(
            negative_prompt.encode("utf-8")
        ).hexdigest(),
        "seconds": elapsed,
        "output_type": first.final_output_type,
        "num_images": len(first.images or []),
        "finished": first.finished,
    }



def run_generator_trials(
    generator,
    *,
    representative: bool,
    t2v_input: Path,
    warmups: int,
    repeats: int,
) -> dict:
    if warmups < 0:
        raise ValueError("--generator-warmups must be non-negative")
    if repeats < 1:
        raise ValueError("--generator-repeats must be at least one")

    warmup_results = []
    measured_results = []
    for index in range(warmups):
        print(f"Generator warmup {index + 1}/{warmups}", flush=True)
        warmup_results.append(
            run_generator(generator, representative=representative, t2v_input=t2v_input)
        )
    for index in range(repeats):
        print(f"Generator measured {index + 1}/{repeats}", flush=True)
        measured_results.append(
            run_generator(generator, representative=representative, t2v_input=t2v_input)
        )

    seconds = [item["seconds"] for item in measured_results]
    result = dict(measured_results[-1])
    result.update(
        {
            "warmups": warmup_results,
            "measured": measured_results,
            "measured_seconds": seconds,
            "median_seconds": statistics.median(seconds),
        }
    )
    return result
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator-mode",
        choices=("eager", "compile"),
        default="eager",
        help="Omni diffusion backend; compile enables its torch.compile path.",
    )
    parser.add_argument(
        "--generator-warmups",
        type=int,
        default=0,
        help="Full Generator requests excluded from measured statistics.",
    )
    parser.add_argument(
        "--generator-repeats",
        type=int,
        default=1,
        help="Full Generator requests included in measured statistics.",
    )
    parser.add_argument("--order", choices=("reasoner-first", "generator-first"), default="reasoner-first")
    parser.add_argument("--output", type=Path, default=Path("profiling_results/gate_1_single_process/result.json"))
    parser.add_argument(
        "--representative-workloads",
        action="store_true",
        help="Use the official HF T2I upsampler (2600-token default cap) and representative T2V.",
    )
    parser.add_argument("--reasoner-max-tokens", type=int, default=2600)
    parser.add_argument("--reasoner-warmups", type=int, default=0)
    parser.add_argument("--reasoner-repeats", type=int, default=1)
    parser.add_argument(
        "--reasoner-mode",
        choices=("eager", "graph"),
        default="eager",
        help="Reasoner execution backend; graph uses vLLM FULL_AND_PIECEWISE CUDA Graph.",
    )
    parser.add_argument(
        "--generator-cache",
        choices=("on", "off"),
        default="off",
        help="Enable Cosmos3 SeaCache with the original framework defaults.",
    )
    parser.add_argument(
        "--t2v-input",
        type=Path,
        default=Path("outputs/local_agentic_t2v/rich02/inputs/t2v_upsampled.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    reasoner = generator = None
    result = {
        "order": args.order,
        "representative_workloads": args.representative_workloads,
        "reasoner_mode": args.reasoner_mode,
        "generator_mode": args.generator_mode,
        "generator_cache": args.generator_cache,
        "snapshots": {"initial": process_snapshot()},
    }
    max_model_len = 6144 if args.representative_workloads else 2048
    try:
        if args.order == "reasoner-first":
            reasoner = create_reasoner(
                borrow_omni_parallel_state=False,
                max_model_len=max_model_len,
                execution_mode=args.reasoner_mode,
            )
            result["snapshots"]["after_reasoner"] = process_snapshot()
            generator = create_generator(
                share_reasoner_parallel_state=True, cache_mode=args.generator_cache, execution_mode=args.generator_mode
            )
            result["snapshots"]["after_both"] = process_snapshot()
        else:
            generator = create_generator(
                share_reasoner_parallel_state=False, cache_mode=args.generator_cache, execution_mode=args.generator_mode
            )
            result["snapshots"]["after_generator"] = process_snapshot()
            reasoner = create_reasoner(
                borrow_omni_parallel_state=True,
                max_model_len=max_model_len,
                execution_mode=args.reasoner_mode,
            )
            result["snapshots"]["after_both"] = process_snapshot()

        reasoner_max_tokens = args.reasoner_max_tokens if args.representative_workloads else 32
        result["reasoner"] = run_reasoner_trials(
            reasoner,
            representative=args.representative_workloads,
            max_tokens=reasoner_max_tokens,
            warmups=args.reasoner_warmups,
            repeats=args.reasoner_repeats,
        )
        result["snapshots"]["after_reasoner_inference"] = process_snapshot()
        result["generator"] = run_generator_trials(
            generator,
            representative=args.representative_workloads,
            t2v_input=args.t2v_input,
            warmups=args.generator_warmups,
            repeats=args.generator_repeats,
        )
        result["snapshots"]["after_generator_inference"] = process_snapshot()

        own_pid = os.getpid()
        gpu_pids = {item["pid"] for item in result["snapshots"]["after_both"]["gpu_processes"]}
        result["checks"] = {
            "same_host_pid": True,
            "only_own_gpu_pid_after_both": gpu_pids == {own_pid},
            "same_main_thread_cuda_context": len(
                {snap["cuda_context"] for snap in result["snapshots"].values() if snap["cuda_context"]}
            ) == 1,
            "reasoner_output_nonempty": bool(result["reasoner"]["text"]),
            "generator_output_nonempty": result["generator"]["num_images"] > 0,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if args.order == "generator-first":
            from profiling_tools.core.omni_shared_parallel import clear_reasoner_context_group_aliases

            # Omni created and owns the process-wide distributed environment in
            # this construction order.  Remove the Reasoner-only aliases, let
            # Omni destroy its groups first, then release the Reasoner model.
            # Reversing the two owner shutdowns invalidates Omni's _WORLD group
            # and makes generator.close() destroy the same process group twice.
            clear_reasoner_context_group_aliases()
            if generator is not None:
                generator.close()
            if reasoner is not None:
                reasoner.llm_engine.engine_core.shutdown()
        else:
            if generator is not None:
                generator.close()
            if reasoner is not None:
                reasoner.llm_engine.engine_core.shutdown()
        del generator, reasoner
        gc.collect()


if __name__ == "__main__":
    main()
