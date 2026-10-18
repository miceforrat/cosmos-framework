"""Capture Cosmos diffusion samplers and benchmark pairwise Green Context colocation."""
from __future__ import annotations

import argparse
import contextvars
import json
import math
import runpy
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from profiling_tools.core.green_context import GreenContextPair

TASK_FROM_MODE = {"text2image": "t2i", "text2video": "t2v", "wam": "action"}
ACTIVE_TASK: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cosmos_diffusion_capture_task", default=None
)
CAPTURED: dict[str, list["CapturedCall"]] = {task: [] for task in TASK_FROM_MODE.values()}


@dataclass(frozen=True)
class CapturedCall:
    task: str
    function: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def invoke(self) -> Any:
        return self.function(*clone_tensors(self.args), **clone_tensors(self.kwargs))

    def prepare(self) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]:
        return self.function, clone_tensors(self.args), clone_tensors(self.kwargs)


@dataclass(frozen=True)
class PairCase:
    first_task: str
    second_task: str
    first_sms: int

    @property
    def name(self) -> str:
        return f"{self.first_task}_{self.second_task}"


def clone_tensors(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_tensors(item) for item in value)
    if isinstance(value, list):
        return [clone_tensors(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tensors(item) for key, item in value.items()}
    return value


def task_from_batch(data_batch: dict[str, Any]) -> str | None:
    mode = data_batch.get("mode")
    while isinstance(mode, (list, tuple)) and mode:
        mode = mode[0]
    mode = getattr(mode, "value", mode)
    return TASK_FROM_MODE.get(str(mode))


def install_capture() -> None:
    from cosmos_framework.model.generator.diffusion.samplers.edm import EDMSampler
    from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
    from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler
    from cosmos_framework.inference.inference import OmniInference
    from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModel
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    original_generate_batch = OmniInference.generate_batch

    def generate_batch_with_task(self: Any, sample_args_list: Any, *args: Any, **kwargs: Any) -> Any:
        mode = sample_args_list[0].model_mode
        task = TASK_FROM_MODE.get(str(getattr(mode, "value", mode)))
        token = ACTIVE_TASK.set(task)
        try:
            return original_generate_batch(self, sample_args_list, *args, **kwargs)
        finally:
            ACTIVE_TASK.reset(token)

    OmniInference.generate_batch = generate_batch_with_task

    for model_type in (OmniMoTModel, OmniMoTCausalModel):
        original_generate = model_type.generate_samples_from_batch

        def generate_with_task(self: Any, data_batch: dict[str, Any], *args: Any,
                               __original: Callable[..., Any] = original_generate,
                               **kwargs: Any) -> Any:
            task = task_from_batch(data_batch)
            token = ACTIVE_TASK.set(task or ACTIVE_TASK.get())
            try:
                return __original(self, data_batch, *args, **kwargs)
            finally:
                ACTIVE_TASK.reset(token)

        model_type.generate_samples_from_batch = generate_with_task

    for sampler_type in (UniPCSampler, EDMSampler, FixedStepSampler):
        original_call = sampler_type.__call__

        def call_and_capture(*args: Any,
                             __original: Callable[..., Any] = original_call,
                             **kwargs: Any) -> Any:
            task = ACTIVE_TASK.get()
            if task is not None:
                CAPTURED[task].append(
                    CapturedCall(task=task, function=__original, args=args, kwargs=dict(kwargs))
                )
            return __original(*args, **kwargs)

        sampler_type.__call__ = call_and_capture


def capture_real_workloads(args: argparse.Namespace) -> None:
    install_capture()
    capture_output = args.capture_output
    if capture_output is None:
        capture_output = Path("outputs/diffusion_colocation") / f"capture_{int(time.time())}"
    inference_argv = [
        "cosmos_framework.scripts.inference",
        "--parallelism-preset=latency",
        "--no-guardrails",
        "--no-diffusion-cache",
        "-i",
        str(args.t2v_input),
        str(args.t2i_input),
        str(args.action_input),
        "-o",
        str(capture_output),
        "--checkpoint-path",
        args.checkpoint_path,
        "--seed=0",
    ]
    previous_argv = sys.argv
    sys.argv = inference_argv
    try:
        runpy.run_module("cosmos_framework.scripts.inference", run_name="__main__")
    finally:
        sys.argv = previous_argv
    counts = {task: len(calls) for task, calls in CAPTURED.items()}
    if any(count != 1 for count in counts.values()):
        raise RuntimeError(f"Expected exactly one sampler call per task; captured {counts}")


def install_replay_attention_dispatch() -> list[tuple[torch.nn.Module, object]]:
    from cosmos_framework.model.generator.mot.inference_text_kv_memory import (
        install_inference_memory_attention_dispatch,
    )

    velocity_fn = CAPTURED["t2v"][0].args[1]
    closure = dict(zip(velocity_fn.__code__.co_freevars, velocity_fn.__closure__ or ()))
    model = closure["self"].cell_contents
    net_cell = closure.get("net")
    net = net_cell.cell_contents if net_cell is not None else None
    return install_inference_memory_attention_dispatch(net or model.net)


def run_workload(task: str, stream: torch.cuda.Stream, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode(), torch.cuda.stream(stream), torch.cuda.nvtx.range(
        f"cosmos.colocation.{task}.repeat{repeats}"
    ):
        prepared = [CAPTURED[task][0].prepare() for _ in range(repeats)]
        start.record(stream)
        for function, args, kwargs in prepared:
            result = function(*args, **kwargs)
            del result
        end.record(stream)
    end.synchronize()
    return start.elapsed_time(end)


def average_workload(task: str, stream: torch.cuda.Stream, repeats: int,
                     iterations: int) -> tuple[float, list[float]]:
    samples = [run_workload(task, stream, repeats) for _ in range(iterations)]
    return statistics.fmean(samples), samples


def balance_repeats(first_ms: float, second_ms: float, max_repeats: int) -> tuple[int, int]:
    if first_ms >= second_ms:
        return 1, min(max_repeats, max(1, round(first_ms / second_ms)))
    return min(max_repeats, max(1, round(second_ms / first_ms))), 1


def benchmark_case(
    case: PairCase,
    full_alone_ms: dict[str, float],
    warmup_iters: int,
    profile_iters: int,
    max_repeats: int,
    ignore_sm_coscheduling: bool,
) -> dict[str, Any]:
    owner = GreenContextPair(
        case.first_sms,
        torch.cuda.current_device(),
        ignore_sm_coscheduling=ignore_sm_coscheduling,
    )
    first_stream = torch.cuda.ExternalStream(owner.first_stream_ptr, device=owner.device)
    second_stream = torch.cuda.ExternalStream(owner.second_stream_ptr, device=owner.device)
    try:
        for _ in range(warmup_iters):
            run_workload(case.first_task, first_stream, 1)
            run_workload(case.second_task, second_stream, 1)

        first_single, first_single_samples = average_workload(
            case.first_task, first_stream, 1, profile_iters
        )
        second_single, second_single_samples = average_workload(
            case.second_task, second_stream, 1, profile_iters
        )
        first_repeats, second_repeats = balance_repeats(
            first_single, second_single, max_repeats
        )
        first_partition, first_partition_samples = average_workload(
            case.first_task, first_stream, first_repeats, profile_iters
        )
        second_partition, second_partition_samples = average_workload(
            case.second_task, second_stream, second_repeats, profile_iters
        )

        barrier = threading.Barrier(3)

        def worker(task: str, stream: torch.cuda.Stream, repeats: int) -> float:
            torch.cuda.set_device(owner.device)
            barrier.wait()
            return run_workload(task, stream, repeats)

        def concurrent_once(executor: ThreadPoolExecutor) -> tuple[float, float, float]:
            future_first = executor.submit(
                worker, case.first_task, first_stream, first_repeats
            )
            future_second = executor.submit(
                worker, case.second_task, second_stream, second_repeats
            )
            barrier.wait()
            wall_start = time.perf_counter()
            first_ms = future_first.result()
            second_ms = future_second.result()
            return first_ms, second_ms, (time.perf_counter() - wall_start) * 1000.0

        with ThreadPoolExecutor(max_workers=2) as executor:
            for _ in range(warmup_iters):
                concurrent_once(executor)
            colocated_samples = [
                concurrent_once(executor) for _ in range(profile_iters)
            ]

        colocated_first = statistics.fmean(x[0] for x in colocated_samples)
        colocated_second = statistics.fmean(x[1] for x in colocated_samples)
        colocated_wall = statistics.fmean(x[2] for x in colocated_samples)
        colocated_gpu_makespan = statistics.fmean(max(x[0], x[1]) for x in colocated_samples)
        full_serial = (
            full_alone_ms[case.first_task] * first_repeats
            + full_alone_ms[case.second_task] * second_repeats
        )
        partition_serial = first_partition + second_partition
        return {
            "case": case.name,
            "requested_first_sm_count": case.first_sms,
            "sm_partition": {
                case.first_task: owner.first_sm_count,
                case.second_task: owner.second_sm_count,
                "total": owner.total_sm_count,
            },
            "workload_repeats": {
                case.first_task: first_repeats,
                case.second_task: second_repeats,
            },
            "full_sm_alone_ms_per_request": {
                case.first_task: full_alone_ms[case.first_task],
                case.second_task: full_alone_ms[case.second_task],
            },
            "partition_alone_ms_per_request": {
                case.first_task: first_single,
                case.second_task: second_single,
            },
            "partition_alone_ms_balanced_workload": {
                case.first_task: first_partition,
                case.second_task: second_partition,
            },
            "colocated_ms_balanced_workload": {
                case.first_task: colocated_first,
                case.second_task: colocated_second,
            },
            "interference_slowdown_pct": {
                case.first_task: 100.0 * (colocated_first / first_partition - 1.0),
                case.second_task: 100.0 * (colocated_second / second_partition - 1.0),
            },
            "full_sm_pure_serial_ms": full_serial,
            "partition_pure_serial_ms": partition_serial,
            "colocated_gpu_makespan_ms": colocated_gpu_makespan,
            "colocated_host_wall_ms": colocated_wall,
            "speedup_vs_full_sm_serial": full_serial / colocated_gpu_makespan,
            "speedup_vs_partition_serial": partition_serial / colocated_gpu_makespan,
            "balanced_tail_ms": abs(colocated_first - colocated_second),
            "samples": {
                "partition_single": {
                    case.first_task: first_single_samples,
                    case.second_task: second_single_samples,
                },
                "partition_balanced": {
                    case.first_task: first_partition_samples,
                    case.second_task: second_partition_samples,
                },
                "colocated": [
                    {
                        case.first_task: first,
                        case.second_task: second,
                        "host_wall_ms": wall,
                    }
                    for first, second, wall in colocated_samples
                ],
            },
        }
    finally:
        torch.cuda.synchronize()
        del first_stream, second_stream
        owner.close()


def parse_case(value: str) -> PairCase:
    try:
        tasks, sm_text = value.rsplit("@", 1)
        first_task, second_task = tasks.split("+", 1)
        first_sms = int(sm_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case must look like t2v+t2i@128 (first task receives the stated SM count)"
        ) from error
    valid = set(CAPTURED)
    if first_task not in valid or second_task not in valid or first_task == second_task:
        raise argparse.ArgumentTypeError(f"tasks must be two distinct values from {sorted(valid)}")
    return PairCase(first_task, second_task, first_sms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", default="Cosmos3-Edge")
    parser.add_argument("--t2v-input", type=Path, default=Path("inputs/omni/t2v.json"))
    parser.add_argument("--t2i-input", type=Path, default=Path("inputs/omni/t2i.json"))
    parser.add_argument(
        "--action-input", type=Path, default=Path("inputs/omni/action_policy_robot.json")
    )
    parser.add_argument("--capture-output", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiling_results/colocation/diffusion_pairwise.json"),
    )
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--max-repeats", type=int, default=32)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        dest="cases",
        help="Pair and first partition size, e.g. t2v+t2i@128. Repeatable.",
    )
    parser.add_argument("--ignore-sm-coscheduling", action="store_true")
    args = parser.parse_args()
    if args.warmup_iters < 0 or args.profile_iters <= 0 or args.max_repeats <= 0:
        parser.error("warmup must be non-negative; profile/max-repeats must be positive")
    if not args.cases:
        args.cases = [
            PairCase("t2v", "t2i", 128),
            PairCase("t2v", "t2i", 136),
            PairCase("t2v", "t2i", 142),
            PairCase("t2v", "action", 128),
            PairCase("t2v", "action", 136),
            PairCase("t2v", "action", 142),
            PairCase("t2i", "action", 80),
        ]
    return args


def main() -> None:
    args = parse_args()
    capture_real_workloads(args)
    previous_dispatch = install_replay_attention_dispatch()
    default_stream = torch.cuda.default_stream()
    full_alone_ms = {}
    full_alone_samples = {}
    for task in ("t2v", "t2i", "action"):
        for _ in range(args.warmup_iters):
            run_workload(task, default_stream, 1)
        full_alone_ms[task], full_alone_samples[task] = average_workload(
            task, default_stream, 1, args.profile_iters
        )
    cases = [
        benchmark_case(
            case,
            full_alone_ms,
            args.warmup_iters,
            args.profile_iters,
            args.max_repeats,
            args.ignore_sm_coscheduling,
        )
        for case in args.cases
    ]
    from cosmos_framework.model.generator.mot.inference_text_kv_memory import (
        restore_inference_attention_dispatch,
    )
    restore_inference_attention_dispatch(previous_dispatch)
    result = {
        "experiment": "cosmos_diffusion_pairwise_green_context_colocation",
        "checkpoint_path": args.checkpoint_path,
        "capture_count": {task: len(calls) for task, calls in CAPTURED.items()},
        "full_sm_alone_ms": full_alone_ms,
        "full_sm_alone_samples": full_alone_samples,
        "warmup_iters": args.warmup_iters,
        "profile_iters": args.profile_iters,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2)
    args.output.write_text(serialized + "\n")
    print(serialized, flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
