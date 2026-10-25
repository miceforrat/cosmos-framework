#!/usr/bin/env python3
"""Validate one eager or regionally compiled Omni Generator across GC streams."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import traceback
from pathlib import Path

import torch

from profiling_tools.core.green_context import GreenContextStream
from profiling_tools.core.runtime_stream_patch import install_omni_diffusion_worker_stream_patch
from profiling_tools.runners.concurrent_reasoner_generator import timed_generator
from profiling_tools.runners.single_process_dual_engine import create_generator, process_snapshot


def run_region(generator, *, stream, name: str, t2v_input: Path, repeats: int) -> list[dict]:
    values = []
    for index in range(repeats):
        print(f"{name} {index + 1}/{repeats}", flush=True)
        item = timed_generator(
            generator,
            t2v_input=t2v_input,
            stream=stream,
            label=f"gate3.generator_migration.{name}.iter{index}",
        )
        stream.synchronize()
        values.append(item)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm-a", type=int, default=136)
    parser.add_argument("--sm-b", type=int, default=104)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--generator-cache", choices=("on", "off"), default="on")
    parser.add_argument(
        "--generator-mode",
        choices=("eager", "compile"),
        default="eager",
        help="compile uses Omni regional torch.compile(dynamic=True), not a diffusion CUDA Graph.",
    )
    parser.add_argument(
        "--t2v-input",
        type=Path,
        default=Path("outputs/local_agentic_t2v/rich02/inputs/t2v_upsampled.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiling_results/gate_3_green_context/generator_gc_stream_migration.json"),
    )
    args = parser.parse_args()
    if args.sm_a == args.sm_b:
        raise ValueError("--sm-a and --sm-b must differ")
    if args.warmups < 1 or args.repeats < 1:
        raise ValueError("--warmups and --repeats must be at least one")
    if not args.t2v_input.exists():
        raise FileNotFoundError(args.t2v_input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    owner_a = GreenContextStream(args.sm_a, device=0)
    owner_b = GreenContextStream(args.sm_b, device=0)
    stream_a = torch.cuda.ExternalStream(owner_a.stream_ptr, device=0)
    stream_b = torch.cuda.ExternalStream(owner_b.stream_ptr, device=0)
    patch = install_omni_diffusion_worker_stream_patch(stream_a)
    generator = None
    result = {
        "experiment": f"generator_{args.generator_mode}_gc_stream_migration",
        "execution_mode": args.generator_mode,
        "compile_policy": (
            None
            if args.generator_mode == "eager"
            else {
                "kind": "regional_torch_compile",
                "dynamic": True,
                "diffusion_cuda_graph": False,
                "kernel_stream_policy": "current_stream_at_invocation",
            }
        ),
        "generator_cache": args.generator_cache,
        "streams": {
            "a": {"requested_sm_count": args.sm_a, "actual_sm_count": owner_a.actual_sm_count, "stream": int(stream_a.cuda_stream)},
            "b": {"requested_sm_count": args.sm_b, "actual_sm_count": owner_b.actual_sm_count, "stream": int(stream_b.cuda_stream)},
        },
        "workload": "t2v_832x480_f121_s35_cfg6",
        "snapshots": {"initial": process_snapshot()},
        "stream_bindings": [{"region": "initialization", "new_stream": int(stream_a.cuda_stream)}],
        "warmups_a": [],
        "a_before": [],
        "b_after_migration": [],
        "a_after_return": [],
        "phase": "create_generator_on_a",
        "failure": None,
    }
    try:
        primary = torch.cuda.current_stream()
        torch.cuda.set_stream(stream_a)
        try:
            generator = create_generator(
                share_reasoner_parallel_state=False,
                cache_mode=args.generator_cache,
                execution_mode=args.generator_mode,
            )
        finally:
            torch.cuda.set_stream(primary)
        result["snapshots"]["after_generator"] = process_snapshot()

        result["phase"] = "warmup_a"
        result["warmups_a"] = run_region(generator, stream=stream_a, name="warmup_a", t2v_input=args.t2v_input, repeats=args.warmups)
        result["phase"] = "a_before"
        result["a_before"] = run_region(generator, stream=stream_a, name="a_before", t2v_input=args.t2v_input, repeats=args.repeats)

        stream_a.synchronize()
        result["stream_bindings"].append({"region": "b_after_migration", **patch.switch_stream(stream_b)})
        result["phase"] = "b_after_migration"
        result["b_after_migration"] = run_region(generator, stream=stream_b, name="b_after_migration", t2v_input=args.t2v_input, repeats=args.repeats)

        stream_b.synchronize()
        result["stream_bindings"].append({"region": "a_after_return", **patch.switch_stream(stream_a)})
        result["phase"] = "a_after_return"
        result["a_after_return"] = run_region(generator, stream=stream_a, name="a_after_return", t2v_input=args.t2v_input, repeats=args.repeats)

        for key in ("warmups_a", "a_before", "b_after_migration", "a_after_return"):
            if not all(x["result"]["finished"] for x in result[key]):
                raise RuntimeError(f"Unfinished Generator output in {key}")

        def med(key: str) -> float:
            return statistics.median(x["host_seconds"] for x in result[key])

        result["summary"] = {
            "a_before_seconds": med("a_before"),
            "b_after_migration_seconds": med("b_after_migration"),
            "a_after_return_seconds": med("a_after_return"),
            "a_return_over_a_before": med("a_after_return") / med("a_before"),
            "worker_patch_calls": patch.calls,
            "output_pack_patch_calls": patch.pack_calls,
            "output_postprocess_patch_calls": patch.postprocess_calls,
            "migration_completed": True,
            "kernel_stream_placement_verified": False,
            "kernel_stream_placement_requires_nsys": True,
        }
        result["phase"] = "complete"
    except BaseException as exc:
        result["failure"] = {"phase": result["phase"], "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        patch.restore()
        if generator is not None:
            try:
                generator.close()
            except BaseException as exc:
                result["cleanup_failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        del generator
        gc.collect()
        stream_a.synchronize()
        stream_b.synchronize()
        del stream_a, stream_b
        gc.collect()
        owner_b.close()
        owner_a.close()

    print(json.dumps(result["summary"], indent=2))
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
