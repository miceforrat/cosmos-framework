#!/usr/bin/env python3
"""Gate 2: concurrent Reasoner and Generator in one PID without spatial sharing."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import functools
import gc
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path


from profiling_tools.runners.single_process_dual_engine import (
    create_generator,
    create_reasoner,
    process_snapshot,
    run_generator,
    run_reasoner,
)


REASONER_RUNS_PER_GENERATOR = 1
COLOCATE_REGION = "full"
REASONER_PROMPT_OVERRIDE: str | None = None



@dataclass
class GeneratorStageMeasurement:
    host_start: float
    host_end: float
    gpu_seconds: float


class GeneratorStageProbe:
    def __init__(self, stage: str, barrier: threading.Barrier | None = None) -> None:
        if stage not in {"vae_decode", "diffusion_sampling"}:
            raise ValueError(f"Unsupported generator stage: {stage}")
        self.stage = stage
        self.barrier = barrier
        self.measurements: list[GeneratorStageMeasurement] = []

    def invoke(self, original, *args, **kwargs):
        import torch
        if self.barrier is not None:
            self.barrier.wait(timeout=300.0)
        stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start_event.record(stream)
        with torch.cuda.nvtx.range(f"gate3.generator.{self.stage}"):
            result = original(*args, **kwargs)
        end_event.record(stream)
        end_event.synchronize()
        host_end = time.perf_counter()
        self.measurements.append(
            GeneratorStageMeasurement(
                host_start,
                host_end,
                start_event.elapsed_time(end_event) / 1000.0,
            )
        )
        return result

    def result(self) -> dict:
        if len(self.measurements) != 1:
            raise RuntimeError(
                f"Expected exactly one Cosmos3 {self.stage}, observed {len(self.measurements)}"
            )
        item = self.measurements[0]
        return {
            "host_start": item.host_start,
            "host_end": item.host_end,
            "host_seconds": item.host_end - item.host_start,
            "gpu_seconds": item.gpu_seconds,
        }


_ACTIVE_GENERATOR_STAGE_PROBE: GeneratorStageProbe | None = None


def install_cosmos3_generator_stage_probes() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
        Cosmos3OmniDiffusersPipeline,
    )

    for stage, attribute in (
        ("vae_decode", "_decode_latents"),
        ("diffusion_sampling", "diffuse"),
    ):
        original = getattr(Cosmos3OmniDiffusersPipeline, attribute)
        if getattr(original, "_cosmos_generator_stage_probe", False):
            continue

        def make_wrapper(original_function, expected_stage):
            @functools.wraps(original_function)
            def wrapped(self, *args, **kwargs):
                probe = _ACTIVE_GENERATOR_STAGE_PROBE
                if probe is None or probe.stage != expected_stage:
                    return original_function(self, *args, **kwargs)
                return probe.invoke(
                    functools.partial(original_function, self), *args, **kwargs
                )

            wrapped._cosmos_generator_stage_probe = True
            return wrapped

        setattr(
            Cosmos3OmniDiffusersPipeline,
            attribute,
            make_wrapper(original, stage),
        )


@contextmanager
def activate_generator_stage_probe(probe: GeneratorStageProbe | None):
    global _ACTIVE_GENERATOR_STAGE_PROBE
    if probe is None:
        yield
        return
    if _ACTIVE_GENERATOR_STAGE_PROBE is not None:
        raise RuntimeError("A Cosmos3 generator stage probe is already active")
    _ACTIVE_GENERATOR_STAGE_PROBE = probe
    try:
        yield
    finally:
        _ACTIVE_GENERATOR_STAGE_PROBE = None

def gpu_used_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.strip().splitlines()[0])


def timed_reasoner(
    reasoner,
    *,
    max_tokens: int,
    stream: torch.cuda.Stream,
    label: str,
    use_external_stream: bool,
    fixed_tokens: bool,
) -> dict:
    import torch
    start = time.perf_counter()
    runs = []
    for run_index in range(REASONER_RUNS_PER_GENERATOR):
        run_start = time.perf_counter()
        # vLLM owns the stream and event contract for CUDA Graph replay.
        stream_context = torch.cuda.stream(stream) if use_external_stream else nullcontext()
        with stream_context, torch.cuda.nvtx.range(f"{label}.request{run_index}"):
            result = run_reasoner(
                reasoner,
                representative=True,
                max_tokens=max_tokens,
                fixed_tokens=fixed_tokens,
                seed=0,
                prompt_override=REASONER_PROMPT_OVERRIDE,
            )
        run_end = time.perf_counter()
        if fixed_tokens and result["completion_tokens"] != max_tokens:
            raise RuntimeError(
                f"Reasoner request {run_index} produced "
                f"{result['completion_tokens']} tokens; expected {max_tokens}"
            )
        runs.append(
            {
                "host_start": run_start,
                "host_end": run_end,
                "host_seconds": run_end - run_start,
                "result": result,
            }
        )
    end = time.perf_counter()
    return {
        "host_start": start,
        "host_end": end,
        "host_seconds": end - start,
        "result": runs[-1]["result"],
        "runs": runs,
    }


def timed_generator(
    generator,
    *,
    t2v_input: Path,
    stream: torch.cuda.Stream,
    label: str,
    stage_probe: GeneratorStageProbe | None = None,
) -> dict:
    import torch
    if stage_probe is None and COLOCATE_REGION in {"vae_decode", "diffusion_sampling"}:
        stage_probe = GeneratorStageProbe(COLOCATE_REGION)
    start = time.perf_counter()
    with (
        activate_generator_stage_probe(stage_probe),
        torch.cuda.stream(stream),
        torch.cuda.nvtx.range(label),
    ):
        result = run_generator(generator, representative=True, t2v_input=t2v_input)
    end = time.perf_counter()
    output = {
        "host_start": start,
        "host_end": end,
        "host_seconds": end - start,
        "result": result,
    }
    if stage_probe is not None:
        output[stage_probe.stage] = stage_probe.result()
    return output


def concurrent_once(
    executor: ThreadPoolExecutor,
    *,
    reasoner,
    generator,
    reasoner_stream: torch.cuda.Stream,
    generator_stream: torch.cuda.Stream,
    max_tokens: int,
    t2v_input: Path,
    iteration: int,
    reasoner_use_external_stream: bool,
    reasoner_fixed_tokens: bool,
    nvtx_prefix: str,
    reasoner_start_delay_seconds: float,
) -> dict:
    start_barrier = threading.Barrier(3) if COLOCATE_REGION == "full" else None
    stage_barrier = (
        threading.Barrier(2)
        if COLOCATE_REGION in {"vae_decode", "diffusion_sampling"}
        else None
    )
    stage_probe = (
        GeneratorStageProbe(COLOCATE_REGION, stage_barrier)
        if stage_barrier is not None
        else None
    )

    def reasoner_worker() -> dict:
        if start_barrier is not None:
            start_barrier.wait(timeout=60.0)
        else:
            stage_barrier.wait(timeout=300.0)
        if reasoner_start_delay_seconds:
            time.sleep(reasoner_start_delay_seconds)
        return timed_reasoner(
            reasoner,
            max_tokens=max_tokens,
            stream=reasoner_stream,
            label=f"{nvtx_prefix}.reasoner.iter{iteration}",
            use_external_stream=reasoner_use_external_stream,
            fixed_tokens=reasoner_fixed_tokens,
        )

    def generator_worker() -> dict:
        if start_barrier is not None:
            start_barrier.wait(timeout=60.0)
        return timed_generator(
            generator,
            t2v_input=t2v_input,
            stream=generator_stream,
            label=f"{nvtx_prefix}.generator.iter{iteration}",
            stage_probe=stage_probe,
        )

    reasoner_future = executor.submit(reasoner_worker)
    generator_future = executor.submit(generator_worker)
    if start_barrier is not None:
        start_barrier.wait(timeout=60.0)
    wall_start = time.perf_counter()
    reasoner_result = reasoner_future.result()
    generator_result = generator_future.result()
    wall_end = time.perf_counter()

    overlap_start = max(reasoner_result["host_start"], generator_result["host_start"])
    overlap_end = min(reasoner_result["host_end"], generator_result["host_end"])
    host_overlap = max(0.0, overlap_end - overlap_start)
    shorter = min(reasoner_result["host_seconds"], generator_result["host_seconds"])
    stage_result = generator_result.get(COLOCATE_REGION)
    stage_overlap = 0.0
    if stage_result is not None:
        stage_overlap = max(
            0.0,
            min(reasoner_result["host_end"], stage_result["host_end"])
            - max(reasoner_result["host_start"], stage_result["host_start"]),
        )
    return {
        "reasoner": reasoner_result,
        "generator": generator_result,
        "configured_reasoner_start_delay_seconds": reasoner_start_delay_seconds,
        "observed_reasoner_start_offset_seconds": (
            reasoner_result["host_start"] - generator_result["host_start"]
        ),
        "observed_reasoner_start_offset_from_stage_seconds": (
            None
            if stage_result is None
            else reasoner_result["host_start"] - stage_result["host_start"]
        ),
        "reasoner_stage_overlap_seconds": stage_overlap,
        "host_wall_seconds": wall_end - wall_start,
        "host_overlap_seconds": host_overlap,
        "host_overlap_over_shorter": host_overlap / shorter if shorter else 0.0,
    }


def median(values: list[float]) -> float:
    return statistics.median(values)


def clear_vllm_cudagraphs() -> None:
    import torch
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    torch.cuda.synchronize()
    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()
    gc.collect()
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reasoner-mode", choices=("eager", "compile", "graph"), default="graph"
    )
    parser.add_argument("--generator-mode", choices=("eager", "compile"), default="compile")
    parser.add_argument("--generator-cache", choices=("on", "off"), default="on")
    parser.add_argument(
        "--colocate-region",
        choices=("full", "diffusion_sampling", "vae_decode"),
        default="full",
    )
    parser.add_argument("--reasoner-max-tokens", type=int, default=2600)
    parser.add_argument("--reasoner-input", type=Path, default=None, help="Use the exact prompt from a native Cosmos Reasoner JSON input.")
    parser.add_argument("--reasoner-runs-per-generator", type=int, default=1)
    parser.add_argument(
        "--generator-sm-count",
        type=int,
        default=None,
        help="Enable Gate 3 with this requested Generator GC size; Reasoner receives the remainder.",
    )
    parser.add_argument(
        "--reasoner-fixed-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force exactly --reasoner-max-tokens tokens for comparable benchmark work.",
    )
    parser.add_argument(
        "--isolate-ir-op-priority",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a profiling-only ContextVar patch for scoped vLLM IrOp priorities.",
    )
    parser.add_argument("--full-sm-warmups", type=int, default=1)
    parser.add_argument("--full-sm-repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--alone-repeats", type=int, default=1)
    parser.add_argument("--concurrent-repeats", type=int, default=1)
    parser.add_argument(
        "--reasoner-start-delay-seconds",
        type=float,
        default=0.0,
        help="Delay only the Reasoner in concurrent rounds; the delay remains part of makespan.",
    )
    parser.add_argument("--memory-sample-period", type=float, default=0.2)
    parser.add_argument(
        "--t2v-input",
        type=Path,
        default=Path("outputs/local_agentic_t2v/rich02/inputs/t2v_upsampled.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiling_results/gate_2_no_spatial/result.json"),
    )
    args = parser.parse_args()
    global REASONER_RUNS_PER_GENERATOR, COLOCATE_REGION, REASONER_PROMPT_OVERRIDE
    REASONER_RUNS_PER_GENERATOR = args.reasoner_runs_per_generator
    COLOCATE_REGION = args.colocate_region
    reasoner_input_data = None
    if args.reasoner_input is not None:
        reasoner_input_data = json.loads(args.reasoner_input.read_text(encoding="utf-8"))
        REASONER_PROMPT_OVERRIDE = reasoner_input_data["prompt"]
        input_max_tokens = int(reasoner_input_data.get("max_new_tokens", args.reasoner_max_tokens))
        if input_max_tokens != args.reasoner_max_tokens:
            raise ValueError(f"Reasoner input max_new_tokens={input_max_tokens} does not match --reasoner-max-tokens={args.reasoner_max_tokens}")
    if COLOCATE_REGION in {"vae_decode", "diffusion_sampling"}:
        install_cosmos3_generator_stage_probes()
    import torch
    if args.reasoner_runs_per_generator < 1:
        raise ValueError("--reasoner-runs-per-generator must be at least one")
    if args.warmups < 1 or args.full_sm_warmups < 1:
        raise ValueError("Gate 2 requires at least one serial warmup")
    if args.full_sm_repeats < 1 or args.alone_repeats < 1 or args.concurrent_repeats < 1:
        raise ValueError("repeat counts must be at least one")
    if args.reasoner_start_delay_seconds < 0:
        raise ValueError("--reasoner-start-delay-seconds must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    generator = reasoner = None
    gc_owner = None
    full_reasoner_gc_owner = full_generator_gc_owner = None
    generator_stream_patch = None
    reasoner_graph_stream_patch = None
    ir_priority_context_patch = None
    reasoner_stream = generator_stream = full_reasoner_stream = full_generator_stream = None
    stop_sampler = threading.Event()
    memory_samples: list[dict] = []
    phase = "initialization"

    def sample_memory() -> None:
        while not stop_sampler.is_set():
            try:
                memory_samples.append(
                    {"monotonic": time.perf_counter(), "phase": phase, "used_mib": gpu_used_mib()}
                )
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
            stop_sampler.wait(args.memory_sample_period)

    sampler = threading.Thread(target=sample_memory, name="gate2-memory-sampler", daemon=True)
    sampler.start()
    result = {
        "gate": (
            "gate_3_green_context"
            if args.generator_sm_count is not None
            else "gate_2_no_spatial_sharing"
        ),
        "reasoner_mode": args.reasoner_mode,
        "generator_mode": args.generator_mode,
        "generator_compile_policy": (
            None
            if args.generator_mode == "eager"
            else {
                "kind": "regional_torch_compile",
                "dynamic": True,
                "diffusion_cuda_graph": False,
                "kernel_stream_policy": "current_stream_at_invocation",
                "kernel_placement_requires_nsys": True,
            }
        ),
        "generator_cache": args.generator_cache,
        "colocate_region": args.colocate_region,
        "reasoner_max_tokens": args.reasoner_max_tokens,
        "reasoner_runs_per_generator": args.reasoner_runs_per_generator,
        "reasoner_fixed_tokens": args.reasoner_fixed_tokens,
        "isolate_ir_op_priority": args.isolate_ir_op_priority,
        "requested_generator_sm_count": args.generator_sm_count,
        "reasoner_input": (None if args.reasoner_input is None else str(args.reasoner_input)),
        "reasoner_prompt_sha256": (None if REASONER_PROMPT_OVERRIDE is None else __import__("hashlib").sha256(REASONER_PROMPT_OVERRIDE.encode("utf-8")).hexdigest()),
        "workload": {
            "reasoner": ("native_cosmos_prompt_json" if args.reasoner_input is not None else "official_hf_t2i_upsampler"),
            "generator": "t2v_832x480_f121_s35_cfg6",
        },
        "snapshots": {"initial": process_snapshot()},
    }

    try:
        if args.generator_sm_count is not None:
            # Cosmos3 Omni ``compile`` is regional torch.compile(dynamic=True),
            # not a captured diffusion CUDA Graph.  Its generated kernels use
            # the current stream at invocation time, so the same worker-level
            # stream patch used by eager mode can bind compiled execution to a
            # Generator Green Context.  Kernel placement remains an explicit
            # Nsys validation requirement below.
            from profiling_tools.core.green_context import GreenContextPair, GreenContextStream
            from profiling_tools.core.runtime_stream_patch import (
                bind_vllm_runner_stream,
                install_omni_diffusion_worker_stream_patch,
            )
            if args.isolate_ir_op_priority:
                from profiling_tools.core.vllm_thread_context_patch import (
                    install_vllm_ir_priority_context_patch,
                )

                ir_priority_context_patch = install_vllm_ir_priority_context_patch()

            gc_owner = GreenContextPair(args.generator_sm_count, device=0)
            generator_stream = torch.cuda.ExternalStream(gc_owner.first_stream_ptr, device=0)
            reasoner_stream = torch.cuda.ExternalStream(gc_owner.second_stream_ptr, device=0)
            primary_stream = torch.cuda.current_stream()
            if args.reasoner_mode != "graph":
                full_reasoner_gc_owner = GreenContextStream(gc_owner.total_sm_count, device=0)
                full_generator_gc_owner = GreenContextStream(gc_owner.total_sm_count, device=0)
                full_reasoner_stream = torch.cuda.ExternalStream(
                    full_reasoner_gc_owner.stream_ptr, device=0
                )
                full_generator_stream = torch.cuda.ExternalStream(
                    full_generator_gc_owner.stream_ptr, device=0
                )
                generator_stream_patch = install_omni_diffusion_worker_stream_patch(
                    full_generator_stream
                )
                torch.cuda.set_stream(full_generator_stream)
                try:
                    generator = create_generator(
                        share_reasoner_parallel_state=False,
                        cache_mode=args.generator_cache,
                        execution_mode=args.generator_mode,
                    )
                finally:
                    torch.cuda.set_stream(primary_stream)
            else:
                # A captured graph cannot migrate between Green Contexts. Use
                # ordinary full-GPU streams for the baseline, then recreate and
                # recapture only the Reasoner on its partition stream below.
                full_reasoner_stream = torch.cuda.Stream()
                full_generator_stream = torch.cuda.Stream()
                generator = create_generator(
                    share_reasoner_parallel_state=False,
                    cache_mode=args.generator_cache,
                    execution_mode=args.generator_mode,
                )
            result["snapshots"]["after_generator"] = process_snapshot()

            if args.reasoner_mode != "graph":
                torch.cuda.set_stream(full_reasoner_stream)
                try:
                    reasoner = create_reasoner(
                        borrow_omni_parallel_state=True,
                        max_model_len=6144,
                        execution_mode=args.reasoner_mode,
                    )
                finally:
                    torch.cuda.set_stream(primary_stream)
                full_reasoner_binding = bind_vllm_runner_stream(reasoner, full_reasoner_stream)
            else:
                reasoner = create_reasoner(
                    borrow_omni_parallel_state=True,
                    max_model_len=6144,
                    execution_mode=args.reasoner_mode,
                )
                full_reasoner_binding = {"policy": "vllm_owned_full_gpu_graph_stream"}
            result["green_context"] = {
                "requested_generator_sm_count": args.generator_sm_count,
                "actual_generator_sm_count": gc_owner.first_sm_count,
                "actual_reasoner_sm_count": gc_owner.second_sm_count,
                "total_sm_count": gc_owner.total_sm_count,
                "generator_is_first_partition": True,
                "ignore_sm_coscheduling": False,
                "full_reasoner_actual_sm_count": gc_owner.total_sm_count,
                "full_generator_actual_sm_count": gc_owner.total_sm_count,
            }
            result["stream_bindings"] = {
                "full_reasoner": full_reasoner_binding,
                "full_generator": int(full_generator_stream.cuda_stream),
            }
        else:
            generator = create_generator(
                share_reasoner_parallel_state=False,
                cache_mode=args.generator_cache,
                execution_mode=args.generator_mode,
            )
            result["snapshots"]["after_generator"] = process_snapshot()
            reasoner = create_reasoner(
                borrow_omni_parallel_state=True,
                max_model_len=6144,
                execution_mode=args.reasoner_mode,
            )
            reasoner_stream = torch.cuda.Stream()
            generator_stream = torch.cuda.Stream()
            result["green_context"] = None

        result["full_sm"] = None
        if args.generator_sm_count is not None:
            result["full_sm"] = {
                "warmups": [],
                "alone": {"reasoner": [], "generator": []},
                "serial_host_wall_seconds": [],
            }
            phase = "full_sm_warmup"
            for index in range(args.full_sm_warmups):
                print(
                    f"Gate 3 full-SM warmup {index + 1}/{args.full_sm_warmups}: Reasoner",
                    flush=True,
                )
                full_reasoner = timed_reasoner(
                    reasoner,
                    max_tokens=args.reasoner_max_tokens,
                    stream=full_reasoner_stream,
                    label=f"gate3.full_sm.warmup.reasoner.iter{index}",
                    use_external_stream=args.reasoner_mode != "graph",
                    fixed_tokens=args.reasoner_fixed_tokens,
                )
                print(
                    f"Gate 3 full-SM warmup {index + 1}/{args.full_sm_warmups}: Generator",
                    flush=True,
                )
                full_generator = timed_generator(
                    generator,
                    t2v_input=args.t2v_input,
                    stream=full_generator_stream,
                    label=f"gate3.full_sm.warmup.generator.iter{index}",
                )
                result["full_sm"]["warmups"].append(
                    {"reasoner": full_reasoner, "generator": full_generator}
                )

            phase = "full_sm_alone"
            for index in range(args.full_sm_repeats):
                serial_start = time.perf_counter()
                result["full_sm"]["alone"]["reasoner"].append(
                    timed_reasoner(
                        reasoner,
                        max_tokens=args.reasoner_max_tokens,
                        stream=full_reasoner_stream,
                        label=f"gate3.full_sm.alone.reasoner.iter{index}",
                        use_external_stream=args.reasoner_mode != "graph",
                        fixed_tokens=args.reasoner_fixed_tokens,
                    )
                )
                result["full_sm"]["alone"]["generator"].append(
                    timed_generator(
                        generator,
                        t2v_input=args.t2v_input,
                        stream=full_generator_stream,
                        label=f"gate3.full_sm.alone.generator.iter{index}",
                    )
                )
                result["full_sm"]["serial_host_wall_seconds"].append(
                    time.perf_counter() - serial_start
                )

            full_reasoner_stream.synchronize()
            full_generator_stream.synchronize()
            if args.reasoner_mode != "graph":
                result["stream_bindings"]["partition_reasoner"] = bind_vllm_runner_stream(
                    reasoner, reasoner_stream
                )
                result["stream_bindings"]["partition_generator"] = (
                    generator_stream_patch.switch_stream(generator_stream)
                )
            else:
                clear_vllm_cudagraphs()
                from profiling_tools.core.omni_shared_parallel import (
                    shutdown_reasoner_preserving_shared_parallel_state,
                )
                from profiling_tools.core.runtime_stream_patch import (
                    install_vllm_graph_capture_stream_patch,
                )

                shutdown_reasoner_preserving_shared_parallel_state(reasoner)
                reasoner = None
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                reasoner_graph_stream_patch = install_vllm_graph_capture_stream_patch(
                    reasoner_stream
                )
                torch.cuda.set_stream(reasoner_stream)
                try:
                    reasoner = create_reasoner(
                        borrow_omni_parallel_state=True,
                        max_model_len=6144,
                        execution_mode=args.reasoner_mode,
                        # Full-SM graph teardown leaves CUDA/runtime pools
                        # resident, while KV allocation remains explicitly
                        # fixed at 1 GiB. Lower only vLLM's admission threshold
                        # for the recapture; this does not change cache size.
                        gpu_memory_utilization=0.55,
                    )
                finally:
                    torch.cuda.set_stream(primary_stream)
                if reasoner_graph_stream_patch.calls == 0:
                    raise RuntimeError(
                        "vLLM graph mode did not enter the patched Green Context "
                        "graph-capture context"
                    )
                if (
                    reasoner_graph_stream_patch.manager_capture_stream_calls
                    + reasoner_graph_stream_patch.compiled_capture_stream_calls
                    == 0
                ):
                    raise RuntimeError(
                        "vLLM graph mode did not request either patched inner "
                        "Green Context capture stream"
                    )
                generator_stream_patch = install_omni_diffusion_worker_stream_patch(
                    generator_stream
                )
                result["stream_bindings"]["partition_reasoner"] = {
                    "policy": "recaptured_vllm_graph_on_green_context",
                    "stream": int(reasoner_stream.cuda_stream),
                    "gpu_memory_utilization_admission": 0.55,
                    "kv_cache_memory_bytes": 1 << 30,
                }
                result["stream_bindings"]["partition_generator"] = {
                    "new_stream": int(generator_stream.cuda_stream)
                }

        result["snapshots"]["after_both"] = process_snapshot()
        result["streams"] = {
            "reasoner": int(reasoner_stream.cuda_stream),
            "generator": int(generator_stream.cuda_stream),
            "distinct": reasoner_stream.cuda_stream != generator_stream.cuda_stream,
            "reasoner_policy": (
                "external_green_context"
                if args.generator_sm_count is not None and args.reasoner_mode != "graph"
                else "vllm_graph_captured_on_green_context"
                if args.generator_sm_count is not None
                else "external_ordinary"
                if args.reasoner_mode != "graph"
                else "vllm_owned_graph_replay"
            ),
            "generator_policy": (
                "external_green_context"
                if args.generator_sm_count is not None
                else "external_ordinary"
            ),
        }
        reasoner_use_external_stream = (
            args.reasoner_mode != "graph" or args.generator_sm_count is not None
        )
        nvtx_prefix = "gate3" if args.generator_sm_count is not None else "gate2"
        gate_display = "Gate 3" if args.generator_sm_count is not None else "Gate 2"

        result["warmups"] = []
        phase = "warmup"
        for index in range(args.warmups):
            print(f"{gate_display} serial warmup {index + 1}/{args.warmups}: Reasoner", flush=True)
            reasoner_warmup = timed_reasoner(
                reasoner,
                max_tokens=args.reasoner_max_tokens,
                stream=reasoner_stream,
                label=f"{nvtx_prefix}.warmup.reasoner.iter{index}",
                use_external_stream=reasoner_use_external_stream,
                fixed_tokens=args.reasoner_fixed_tokens,
            )
            print(f"{gate_display} serial warmup {index + 1}/{args.warmups}: Generator", flush=True)
            generator_warmup = timed_generator(
                generator,
                t2v_input=args.t2v_input,
                stream=generator_stream,
                label=f"{nvtx_prefix}.warmup.generator.iter{index}",
            )
            result["warmups"].append(
                {"reasoner": reasoner_warmup, "generator": generator_warmup}
            )

        nsys_capture_enabled = os.environ.get("COSMOS_NSYS_CAPTURE") == "1"
        if nsys_capture_enabled:
            import torch

            torch.cuda.cudart().cudaProfilerStart()

        phase = "alone"
        result["alone"] = {"reasoner": [], "generator": []}
        result["partitioned_serial_host_wall_seconds"] = []
        for index in range(args.alone_repeats):
            serial_start = time.perf_counter()
            print(f"{gate_display} alone {index + 1}/{args.alone_repeats}: Reasoner", flush=True)
            result["alone"]["reasoner"].append(
                timed_reasoner(
                    reasoner,
                    max_tokens=args.reasoner_max_tokens,
                    stream=reasoner_stream,
                    label=f"{nvtx_prefix}.alone.reasoner.iter{index}",
                    use_external_stream=reasoner_use_external_stream,
                    fixed_tokens=args.reasoner_fixed_tokens,
                )
            )
            print(f"{gate_display} alone {index + 1}/{args.alone_repeats}: Generator", flush=True)
            result["alone"]["generator"].append(
                timed_generator(
                    generator,
                    t2v_input=args.t2v_input,
                    stream=generator_stream,
                    label=f"{nvtx_prefix}.alone.generator.iter{index}",
                )
            )
            result["partitioned_serial_host_wall_seconds"].append(
                time.perf_counter() - serial_start
            )

        phase = "concurrent"
        result["concurrent"] = []
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gate2") as executor:
            for index in range(args.concurrent_repeats):
                print(f"{gate_display} concurrent {index + 1}/{args.concurrent_repeats}", flush=True)
                result["concurrent"].append(
                    concurrent_once(
                        executor,
                        reasoner=reasoner,
                        generator=generator,
                        reasoner_stream=reasoner_stream,
                        generator_stream=generator_stream,
                        max_tokens=args.reasoner_max_tokens,
                        t2v_input=args.t2v_input,
                        iteration=index,
                        reasoner_use_external_stream=reasoner_use_external_stream,
                        reasoner_fixed_tokens=args.reasoner_fixed_tokens,
                        nvtx_prefix=nvtx_prefix,
                        reasoner_start_delay_seconds=args.reasoner_start_delay_seconds,
                    )
                )

        if nsys_capture_enabled:
            torch.cuda.cudart().cudaProfilerStop()

        alone_reasoner = [x["host_seconds"] for x in result["alone"]["reasoner"]]
        alone_generator = [x["host_seconds"] for x in result["alone"]["generator"]]
        coloc_reasoner = [x["reasoner"]["host_seconds"] for x in result["concurrent"]]
        coloc_generator = [x["generator"]["host_seconds"] for x in result["concurrent"]]
        coloc_wall = [x["host_wall_seconds"] for x in result["concurrent"]]
        stage_name = args.colocate_region if args.colocate_region != "full" else None
        partition_stage = (
            [x[stage_name]["gpu_seconds"] for x in result["alone"]["generator"]]
            if stage_name is not None
            else []
        )
        coloc_stage = (
            [x["generator"][stage_name]["gpu_seconds"] for x in result["concurrent"]]
            if stage_name is not None
            else []
        )
        if result["full_sm"] is not None:
            full_reasoner = [
                x["host_seconds"] for x in result["full_sm"]["alone"]["reasoner"]
            ]
            full_generator = [
                x["host_seconds"] for x in result["full_sm"]["alone"]["generator"]
            ]
        else:
            full_reasoner = alone_reasoner
            full_generator = alone_generator
        reasoner_token_counts = {
            "full_sm_warmups": (
                []
                if result["full_sm"] is None
                else [
                    x["reasoner"]["result"]["completion_tokens"]
                    for x in result["full_sm"]["warmups"]
                ]
            ),
            "full_sm_alone": (
                []
                if result["full_sm"] is None
                else [
                    x["result"]["completion_tokens"]
                    for x in result["full_sm"]["alone"]["reasoner"]
                ]
            ),
            "warmups": [x["reasoner"]["result"]["completion_tokens"] for x in result["warmups"]],
            "alone": [x["result"]["completion_tokens"] for x in result["alone"]["reasoner"]],
            "concurrent": [
                x["reasoner"]["result"]["completion_tokens"] for x in result["concurrent"]
            ],
        }
        all_reasoner_token_counts = [
            count for counts in reasoner_token_counts.values() for count in counts
        ]
        expected_tokens = args.reasoner_max_tokens if args.reasoner_fixed_tokens else None
        token_workload_valid = len(set(all_reasoner_token_counts)) == 1 and (
            expected_tokens is None or all_reasoner_token_counts[0] == expected_tokens
        )
        result["reasoner_token_counts"] = reasoner_token_counts
        if not token_workload_valid:
            raise RuntimeError(f"Reasoner token workload mismatch: {reasoner_token_counts}")

        def collect_values(node, key: str) -> list:
            values = []
            if isinstance(node, dict):
                if key in node:
                    values.append(node[key])
                for value in node.values():
                    values.extend(collect_values(value, key))
            elif isinstance(node, list):
                for value in node:
                    values.extend(collect_values(value, key))
            return values

        reasoner_prompt_token_hashes = set(
            collect_values(result, "prompt_token_ids_sha256")
        )
        generator_workload_hashes = set(
            collect_values(result, "canonical_workload_sha256")
        )
        if len(reasoner_prompt_token_hashes) != 1:
            raise RuntimeError(
                "Reasoner prompt token IDs changed across runs: "
                f"{sorted(reasoner_prompt_token_hashes)}"
            )
        if len(generator_workload_hashes) != 1:
            raise RuntimeError(
                "Generator canonical workload changed across runs: "
                f"{sorted(generator_workload_hashes)}"
            )
        result["matched_workload_identity"] = {
            "reasoner_prompt_token_ids_sha256": next(
                iter(reasoner_prompt_token_hashes)
            ),
            "reasoner_fixed_completion_tokens_per_request": expected_tokens,
            "reasoner_requests_per_generator": args.reasoner_runs_per_generator,
            "generator_canonical_workload_sha256": next(
                iter(generator_workload_hashes)
            ),
            "native_generator_sample_args": os.environ.get(
                "COSMOS_MATCHED_GENERATOR_SAMPLE_ARGS"
            ),
        }
        partitioned_serial_wall = median(result["partitioned_serial_host_wall_seconds"])
        full_sm_serial_wall = (
            median(result["full_sm"]["serial_host_wall_seconds"])
            if result["full_sm"] is not None
            else partitioned_serial_wall
        )
        result["summary"] = {
            "reasoner_full_sm_alone_seconds": median(full_reasoner),
            "generator_full_sm_alone_seconds": median(full_generator),
            "reasoner_alone_seconds": median(alone_reasoner),
            "generator_alone_seconds": median(alone_generator),
            "reasoner_sm_restriction_slowdown": median(alone_reasoner) / median(full_reasoner),
            "generator_sm_restriction_slowdown": median(alone_generator) / median(full_generator),
            "serial_host_wall_seconds": full_sm_serial_wall,
            "partitioned_serial_host_wall_seconds": partitioned_serial_wall,
            "reasoner_concurrent_seconds": median(coloc_reasoner),
            "generator_concurrent_seconds": median(coloc_generator),
            "concurrent_host_wall_seconds": median(coloc_wall),
            "reasoner_slowdown": median(coloc_reasoner) / median(alone_reasoner),
            "generator_slowdown": median(coloc_generator) / median(alone_generator),
            "measured_generator_stage": stage_name,
            "generator_stage_alone_seconds": (
                median(partition_stage) if partition_stage else None
            ),
            "generator_stage_concurrent_seconds": (
                median(coloc_stage) if coloc_stage else None
            ),
            "generator_stage_slowdown": (
                median(coloc_stage) / median(partition_stage)
                if partition_stage and coloc_stage
                else None
            ),
            "vae_decode_alone_seconds": (
                median(partition_stage) if stage_name == "vae_decode" else None
            ),
            "vae_decode_concurrent_seconds": (
                median(coloc_stage) if stage_name == "vae_decode" else None
            ),
            "vae_decode_slowdown": (
                median(coloc_stage) / median(partition_stage)
                if stage_name == "vae_decode" and partition_stage and coloc_stage
                else None
            ),
            "diffusion_sampling_alone_seconds": (
                median(partition_stage)
                if stage_name == "diffusion_sampling"
                else None
            ),
            "diffusion_sampling_concurrent_seconds": (
                median(coloc_stage)
                if stage_name == "diffusion_sampling"
                else None
            ),
            "diffusion_sampling_slowdown": (
                median(coloc_stage) / median(partition_stage)
                if stage_name == "diffusion_sampling" and partition_stage and coloc_stage
                else None
            ),
            "throughput_gain_host_wall": full_sm_serial_wall / median(coloc_wall),
            "throughput_gain_partitioned_reference": partitioned_serial_wall / median(coloc_wall),
            "host_overlap_seconds": median([x["host_overlap_seconds"] for x in result["concurrent"]]),
            "host_overlap_over_shorter": median(
                [x["host_overlap_over_shorter"] for x in result["concurrent"]]
            ),
            "configured_reasoner_start_delay_seconds": args.reasoner_start_delay_seconds,
            "observed_reasoner_start_offset_seconds": median(
                [x["observed_reasoner_start_offset_seconds"] for x in result["concurrent"]]
            ),
            "observed_reasoner_start_offset_from_stage_seconds": (
                median(
                    [
                        x["observed_reasoner_start_offset_from_stage_seconds"]
                        for x in result["concurrent"]
                    ]
                )
                if stage_name is not None
                else None
            ),
            "reasoner_stage_overlap_seconds": (
                median([x["reasoner_stage_overlap_seconds"] for x in result["concurrent"]])
                if stage_name is not None
                else None
            ),
            "gpu_kernel_overlap_verified": False,
            "spatial_sharing_enabled": args.generator_sm_count is not None,
            "reasoner_token_workload_valid": token_workload_valid,
        }
        result["snapshots"]["after_concurrent"] = process_snapshot()
        result["checks"] = {
            "same_host_pid": True,
            "only_own_gpu_pid": {
                item["pid"] for item in result["snapshots"]["after_both"]["gpu_processes"]
            }
            == {result["snapshots"]["after_both"]["pid"]},
            "same_main_thread_cuda_context": len(
                {
                    snap["cuda_context"]
                    for snap in result["snapshots"].values()
                    if snap["cuda_context"]
                }
            )
            == 1,
            "distinct_requested_streams": result["streams"]["distinct"],
            "both_workloads_externally_stream_wrapped": reasoner_use_external_stream,
            "reasoner_graph_gc_binding_requested": (
                args.generator_sm_count is not None and args.reasoner_mode == "graph"
            ),
            "green_context_kernel_placement_verified": False,
            "generator_worker_stream_patch_installed": generator_stream_patch is not None,
            "generator_compile_gc_binding_requested": (
                args.generator_sm_count is not None and args.generator_mode == "compile"
            ),
            # This stays false until the resulting Nsys trace is inspected or
            # an offline SQLite checker is wired into the runner.  A successful
            # request alone does not prove that compiled kernels stayed in GC.
            "generator_compile_kernel_placement_verified": False,
            "generator_worker_stream_patch_calls": (
                generator_stream_patch.calls if generator_stream_patch is not None else 0
            ),
            "generator_output_postprocess_patch_calls": (
                generator_stream_patch.postprocess_calls
                if generator_stream_patch is not None
                else 0
            ),
            "reasoner_graph_stream_patch_installed": reasoner_graph_stream_patch is not None,
            "reasoner_graph_capture_context_calls": (
                reasoner_graph_stream_patch.calls
                if reasoner_graph_stream_patch is not None
                else 0
            ),
            "reasoner_graph_manager_capture_stream_calls": (
                reasoner_graph_stream_patch.manager_capture_stream_calls
                if reasoner_graph_stream_patch is not None
                else 0
            ),
            "reasoner_graph_compiled_capture_stream_calls": (
                reasoner_graph_stream_patch.compiled_capture_stream_calls
                if reasoner_graph_stream_patch is not None
                else 0
            ),
            "ir_priority_context_patch_installed": ir_priority_context_patch is not None,
            "ir_priority_context_dispatch_calls": (
                ir_priority_context_patch.dispatch_calls
                if ir_priority_context_patch is not None
                else 0
            ),
            "ir_priority_context_override_entries": (
                ir_priority_context_patch.override_entries
                if ir_priority_context_patch is not None
                else 0
            ),
        }
    finally:
        phase = "shutdown"
        stop_sampler.set()
        sampler.join(timeout=2.0)
        result["memory_samples"] = memory_samples
        if memory_samples:
            result["peak_memory_mib"] = max(x["used_mib"] for x in memory_samples)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if gc_owner is not None and reasoner_stream is not None and generator_stream is not None:
            reasoner_stream.synchronize()
            generator_stream.synchronize()
            full_reasoner_stream.synchronize()
            full_generator_stream.synchronize()
        if generator_stream_patch is not None:
            generator_stream_patch.restore()
        if reasoner_graph_stream_patch is not None:
            reasoner_graph_stream_patch.restore()
        if generator is not None or reasoner is not None:
            from profiling_tools.core.omni_shared_parallel import (
                clear_reasoner_context_group_aliases,
            )

            clear_reasoner_context_group_aliases()
            if generator is not None:
                generator.close()
            if reasoner is not None:
                reasoner.llm_engine.engine_core.shutdown()
        if ir_priority_context_patch is not None:
            ir_priority_context_patch.restore()
        del generator, reasoner
        gc.collect()
        if gc_owner is not None:
            del reasoner_stream, generator_stream, full_reasoner_stream, full_generator_stream
            gc.collect()
            gc_owner.close()
            if full_generator_gc_owner is not None:
                full_generator_gc_owner.close()
            if full_reasoner_gc_owner is not None:
                full_reasoner_gc_owner.close()

    print(json.dumps(result["summary"], indent=2))
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
