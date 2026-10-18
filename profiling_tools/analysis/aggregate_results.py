"""Aggregate repeated component runs into one row per task/component/SM count."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

PATTERN = re.compile(
    r"^(?P<task>t2i|t2v|action|reasoner_video|agentic_upsampler|agentic_generator)_"
    r"(?P<component>.+)_sm(?P<sm>\d{3})(?:_run(?P<run>\d+))?\.json$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("profiling_results/components"))
    parser.add_argument("--output", type=Path, default=Path("profiling_results/sm_scaling.csv"))
    args = parser.parse_args()

    groups: dict[tuple[str, str, int], list[tuple[float, int, int | None, str]]] = defaultdict(list)
    for path in sorted(args.input_dir.glob("*_sm*.json")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        component = match.group("component")
        records = [
            record
            for record in json.loads(path.read_text())["records"]
            if record["component"] == component and record["gpu_ms"] is not None
        ]
        if not records:
            print(f"WARNING: no {component} records in {path}")
            continue
        requested = int(match.group("sm"))
        actual_values = {record["actual_sm_count"] for record in records if record["actual_sm_count"] is not None}
        actual = next(iter(actual_values)) if len(actual_values) == 1 else None
        groups[(match.group("task"), component, requested)].append(
            (sum(float(record["gpu_ms"]) for record in records), len(records), actual, str(path))
        )

    rows = []
    for (task, component, requested), runs in sorted(groups.items()):
        totals = [run[0] for run in runs]
        actual_values = {run[2] for run in runs if run[2] is not None}
        rows.append(
            {
                "task": task,
                "component": component,
                "requested_sm_count": requested,
                "actual_sm_count": next(iter(actual_values)) if len(actual_values) == 1 else "",
                "repetitions": len(runs),
                "calls_per_run_median": statistics.median(run[1] for run in runs),
                "gpu_total_ms": statistics.median(totals),
                "gpu_total_mean_ms": statistics.mean(totals),
                "gpu_total_stdev_ms": statistics.stdev(totals) if len(totals) > 1 else 0.0,
                "gpu_total_min_ms": min(totals),
                "gpu_total_max_ms": max(totals),
                "sources": ";".join(run[3] for run in runs),
            }
        )

    if not rows:
        raise SystemExit("No component sweep JSON files found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} aggregated rows to {args.output}")


if __name__ == "__main__":
    main()
