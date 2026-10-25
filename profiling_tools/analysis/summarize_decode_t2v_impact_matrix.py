#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PARTITION_RE = re.compile(
    r"GC partition:\s+requested_generator=(\d+)\s+actual_generator=(\d+)\s+"
    r"actual_reasoner_remainder=(\d+)\s+total=(\d+)"
)
FIELD_RE = re.compile(r"^\s{2}([a-z0-9_]+)=([0-9.]+)(?: s|x)\s*$", re.MULTILINE)
GENERATOR_REPEATS_RE = re.compile(r"generator_repeats=(\d+)")
GENERATOR_WORKLOAD_RE = re.compile(r"generator_workload=([a-z0-9_]+)")
COLOCATED_HOST_WALL_RE = re.compile(
    r"Colocation result.*?\n\s{2}host_wall=([0-9.]+) s", re.DOTALL
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("profiling_results/decode_t2v_impact_matrix"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("profiling_results/decode_t2v_impact_matrix/summary.csv"),
    )
    parser.add_argument(
        "--pattern",
        default="*.log",
        help="Log filename glob within --input-dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(args.input_dir.glob(args.pattern)):
        if any(tag in path.name for tag in ("_calibration_", "_smoke", "_quick")):
            continue
        text = path.read_text(errors="replace")
        partition = PARTITION_RE.search(text)
        if partition is None or "Balanced interference summary:" not in text:
            print(f"Skipping incomplete log: {path}")
            continue
        fields = dict(FIELD_RE.findall(text[text.rfind("Balanced interference summary:"):]))
        colocated_host_wall_matches = COLOCATED_HOST_WALL_RE.findall(text)
        if not colocated_host_wall_matches:
            raise ValueError(f"{path}: missing colocated host_wall")
        # New logs use workload-neutral generator_* names. Accept historical
        # T2V logs as aliases so existing experiments remain summarizable.
        for old_name, new_name in (
            ("t2v_alone", "generator_alone"),
            ("t2v_colocated", "generator_colocated"),
            ("t2v_slowdown", "generator_slowdown"),
        ):
            if new_name not in fields and old_name in fields:
                fields[new_name] = fields[old_name]
        # Normalize renamed fields while retaining compatibility with historical logs.
        if "partition_serial_host_wall" in fields and "serial_host_wall" not in fields:
            fields["serial_host_wall"] = fields["partition_serial_host_wall"]
        if "partition_overlap_efficiency_host_wall" in fields and "throughput_gain_host_wall" not in fields:
            fields["throughput_gain_host_wall"] = fields["partition_overlap_efficiency_host_wall"]
        required = {
            "decode_alone",
            "decode_colocated",
            "decode_slowdown",
            "generator_alone",
            "generator_colocated",
            "generator_slowdown",
            "serial_gpu_sum",
            "serial_host_wall",
            "colocated_makespan",
            "throughput_gain_gpu_sum",
            "throughput_gain_host_wall",
        }
        missing = required - fields.keys()
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        requested_generator, actual_generator, actual_reasoner, total = map(
            int, partition.groups()
        )
        implementation = next(
            (prefix for prefix in ("static_eager", "eager", "graph") if path.name.startswith(prefix + "_")),
            path.name.split("_", 1)[0],
        )
        row: dict[str, object] = {
            "implementation": implementation,
            "generator_workload": (
                GENERATOR_WORKLOAD_RE.search(text).group(1)
                if GENERATOR_WORKLOAD_RE.search(text)
                else "t2v"
            ),
            "generator_repeats": (
                int(GENERATOR_REPEATS_RE.search(text).group(1))
                if GENERATOR_REPEATS_RE.search(text)
                else 1
            ),
            "requested_generator_sm": requested_generator,
            "actual_reasoner_sm": actual_reasoner,
            "actual_generator_sm": actual_generator,
            "total_sm": total,
        }
        row.update({name: float(fields[name]) for name in sorted(required)})
        row["colocated_host_wall"] = float(colocated_host_wall_matches[-1])
        row["alone_time_gap_s"] = abs(
            float(fields["decode_alone"]) - float(fields["generator_alone"])
        )
        row["balance_ratio_decode_over_t2v"] = (
            float(fields["decode_alone"]) / float(fields["generator_alone"])
        )
        row["source"] = str(path)
        rows.append(row)

    if not rows:
        raise SystemExit(f"No completed logs found in {args.input_dir}")
    implementation_order = {"eager": 0, "static_eager": 1, "graph": 2}
    rows.sort(key=lambda row: (implementation_order.get(str(row["implementation"]), 99), int(row["actual_reasoner_sm"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} matrix cells)")

    for row in rows:
        print(
            (
                "{implementation:5s} "
                "R{actual_reasoner_sm:>3}/G{actual_generator_sm:>3} "
                "alone={decode_alone:.3f}/{generator_alone:.3f}s "
                "slowdown={decode_slowdown:.3f}/{generator_slowdown:.3f}x "
            ).format(**row)
        )


if __name__ == "__main__":
    main()
