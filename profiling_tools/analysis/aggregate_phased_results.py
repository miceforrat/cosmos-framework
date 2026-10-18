"""Aggregate phased component SM sweeps without mixing cold and warm passes."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

PATTERN = re.compile(
    r"^(?P<task>t2i|t2v|action|reasoner_video|agentic_upsampler|agentic_generator|agentic_video_upsampler|agentic_video_generator)_"
    r"(?P<component>.+)_sm(?P<sm>\d{3})_run(?P<run>\d+)\.json$"
)
PHASES = ("cold", "measured")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("profiling_results/components_phased_on"))
    parser.add_argument("--output", type=Path, default=Path("profiling_results/sm_scaling_phased_on.csv"))
    args = parser.parse_args()

    groups: dict[tuple[str, str, int, str], list[tuple[float, int, int | None, str]]] = defaultdict(list)
    for path in sorted(args.input_dir.glob("*_sm*_run*.json")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        component = match.group("component")
        per_pass: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for record in json.loads(path.read_text())["records"]:
            if record["component"] != component or record.get("gpu_ms") is None:
                continue
            phase = record.get("phase")
            pass_index = record.get("pass_index")
            if phase in PHASES and pass_index is not None:
                per_pass[(phase, int(pass_index))].append(record)
        requested = int(match.group("sm"))
        for (phase, pass_index), records in per_pass.items():
            actual_values = {r.get("actual_sm_count") for r in records if r.get("actual_sm_count") is not None}
            actual = next(iter(actual_values)) if len(actual_values) == 1 else None
            groups[(match.group("task"), component, requested, phase)].append(
                (
                    sum(float(r["gpu_ms"]) for r in records),
                    len(records),
                    actual,
                    f"{path}#pass{pass_index}",
                )
            )

    rows = []
    for (task, component, requested, phase), samples in sorted(groups.items()):
        totals = [sample[0] for sample in samples]
        actual_values = {sample[2] for sample in samples if sample[2] is not None}
        rows.append(
            {
                "task": task,
                "component": component,
                "phase": phase,
                "requested_sm_count": requested,
                "actual_sm_count": next(iter(actual_values)) if len(actual_values) == 1 else "",
                "samples": len(totals),
                "calls_per_sample_median": statistics.median(sample[1] for sample in samples),
                "gpu_total_median_ms": statistics.median(totals),
                "gpu_total_mean_ms": statistics.mean(totals),
                "gpu_total_stdev_ms": statistics.stdev(totals) if len(totals) > 1 else 0.0,
                "gpu_total_min_ms": min(totals),
                "gpu_total_max_ms": max(totals),
                "sources": ";".join(sample[3] for sample in samples),
            }
        )
    if not rows:
        raise SystemExit(f"No phased SM sweep records found under {args.input_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
