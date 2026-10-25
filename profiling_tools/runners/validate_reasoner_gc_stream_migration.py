#!/usr/bin/env python3
"""Validate one eager Reasoner migrating between Green Context streams."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import traceback
from pathlib import Path

import torch

from profiling_tools.core.green_context import GreenContextStream
from profiling_tools.core.runtime_stream_patch import bind_vllm_runner_stream
from profiling_tools.runners.concurrent_reasoner_generator import timed_reasoner
from profiling_tools.runners.single_process_dual_engine import create_reasoner, process_snapshot


def run_region(reasoner, *, stream, name: str, max_tokens: int, repeats: int) -> list[dict]:
    results = []
    for index in range(repeats):
        print(f"{name} {index + 1}/{repeats}", flush=True)
        item = timed_reasoner(
            reasoner,
            max_tokens=max_tokens,
            stream=stream,
            label=f"gate3.migration.{name}.iter{index}",
            use_external_stream=True,
            fixed_tokens=True,
        )
        stream.synchronize()
        results.append(item)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm-a", type=int, default=34)
    parser.add_argument("--sm-b", type=int, default=66)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiling_results/gate_3_green_context/reasoner_gc_stream_migration.json"),
    )
    args = parser.parse_args()
    if args.sm_a == args.sm_b:
        raise ValueError("--sm-a and --sm-b must differ")
    if args.warmups < 1 or args.repeats < 1:
        raise ValueError("--warmups and --repeats must be at least one")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    owner_a = GreenContextStream(args.sm_a, device=0)
    owner_b = GreenContextStream(args.sm_b, device=0)
    stream_a = torch.cuda.ExternalStream(owner_a.stream_ptr, device=0)
    stream_b = torch.cuda.ExternalStream(owner_b.stream_ptr, device=0)
    reasoner = None
    result = {
        "experiment": "reasoner_eager_gc_stream_migration",
        "execution_mode": "eager",
        "fixed_tokens": True,
        "max_tokens": args.max_tokens,
        "streams": {
            "a": {"requested_sm_count": args.sm_a, "actual_sm_count": owner_a.actual_sm_count, "stream": int(stream_a.cuda_stream)},
            "b": {"requested_sm_count": args.sm_b, "actual_sm_count": owner_b.actual_sm_count, "stream": int(stream_b.cuda_stream)},
        },
        "snapshots": {"initial": process_snapshot()},
        "warmups_a": [],
        "a_before": [],
        "b_after_migration": [],
        "a_after_return": [],
        "phase": "create_reasoner_on_a",
        "failure": None,
        "stream_bindings": [],
    }
    try:
        primary_stream = torch.cuda.current_stream()
        torch.cuda.set_stream(stream_a)
        try:
            reasoner = create_reasoner(
                borrow_omni_parallel_state=False,
                max_model_len=6144,
                execution_mode="eager",
            )
        finally:
            torch.cuda.set_stream(primary_stream)
        result["snapshots"]["after_reasoner"] = process_snapshot()

        result["stream_bindings"].append(
            {"region": "warmup_a", **bind_vllm_runner_stream(reasoner, stream_a)}
        )
        result["phase"] = "warmup_a"
        result["warmups_a"] = run_region(reasoner, stream=stream_a, name="warmup_a", max_tokens=args.max_tokens, repeats=args.warmups)
        result["phase"] = "a_before"
        result["a_before"] = run_region(reasoner, stream=stream_a, name="a_before", max_tokens=args.max_tokens, repeats=args.repeats)
        stream_a.synchronize()
        result["stream_bindings"].append(
            {
                "region": "b_after_migration",
                **bind_vllm_runner_stream(reasoner, stream_b),
            }
        )
        result["phase"] = "b_after_migration"
        result["b_after_migration"] = run_region(reasoner, stream=stream_b, name="b_after_migration", max_tokens=args.max_tokens, repeats=args.repeats)
        stream_b.synchronize()
        result["stream_bindings"].append(
            {"region": "a_after_return", **bind_vllm_runner_stream(reasoner, stream_a)}
        )
        result["phase"] = "a_after_return"
        result["a_after_return"] = run_region(reasoner, stream=stream_a, name="a_after_return", max_tokens=args.max_tokens, repeats=args.repeats)

        token_counts = {
            key: [x["result"]["completion_tokens"] for x in result[key]]
            for key in ("warmups_a", "a_before", "b_after_migration", "a_after_return")
        }
        if {count for values in token_counts.values() for count in values} != {args.max_tokens}:
            raise RuntimeError(f"Reasoner token workload mismatch: {token_counts}")

        def med(key: str) -> float:
            return statistics.median(x["host_seconds"] for x in result[key])

        result["token_counts"] = token_counts
        result["summary"] = {
            "a_before_seconds": med("a_before"),
            "b_after_migration_seconds": med("b_after_migration"),
            "a_after_return_seconds": med("a_after_return"),
            "a_return_over_a_before": med("a_after_return") / med("a_before"),
            "migration_completed": True,
            "kernel_stream_placement_verified": False,
        }
        result["phase"] = "complete"
    except BaseException as exc:
        result["failure"] = {
            "phase": result["phase"],
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if reasoner is not None:
            try:
                reasoner.llm_engine.engine_core.shutdown()
            except BaseException as cleanup_exc:
                result["cleanup_failure"] = {
                    "type": type(cleanup_exc).__name__,
                    "message": str(cleanup_exc),
                    "traceback": traceback.format_exc(),
                }
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        del reasoner
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
