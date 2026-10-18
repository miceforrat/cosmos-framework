"""Select representative kernels for every profiled Cosmos task/component."""
from __future__ import annotations

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path

DEFAULT_TASKS = ("t2i", "t2v", "action", "reasoner_video")
PREFIX = "cosmos.component."


def find_column(names: list[str], prefix: str) -> str:
    for name in names:
        if name.strip().lower().startswith(prefix.lower()):
            return name
    raise ValueError(f"No CSV column begins with {prefix!r}: {names}")


def read_nsys_csv(path: Path) -> tuple[list[dict[str, str]], str, str, str]:
    lines = path.read_text().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("NVTX Range,"))
    except StopIteration:
        raise ValueError(f"No NVTX Range CSV header in {path}") from None
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    rows, names = list(reader), reader.fieldnames or []
    return (rows, find_column(names, "nvtx range"),
            find_column(names, "kernel name"), find_column(names, "total time"))


def clean_range(value: str) -> str:
    return value.strip().lstrip(":")


def select(rows: list[dict[str, str]], range_col: str, kernel_col: str,
           total_col: str, nvtx_range: str, coverage: float) -> list[dict[str, object]]:
    # Aggregate defensively in case the same name appears under multiple PID/TID rows.
    times: dict[str, float] = defaultdict(float)
    for row in rows:
        if clean_range(row[range_col]) == nvtx_range:
            times[row[kernel_col]] += float(row[total_col].replace(",", "").strip())
    ranked = sorted(times.items(), key=lambda item: item[1], reverse=True)
    total = sum(value for _, value in ranked)
    cumulative = 0.0
    result: list[dict[str, object]] = []
    for rank, (kernel, value) in enumerate(ranked, 1):
        cumulative += value
        result.append({
            "rank": rank,
            "nvtx_range": nvtx_range,
            "kernel_name": kernel,
            "total_time_ns": value,
            "share": value / total if total else 0.0,
            "cumulative_share": cumulative / total if total else 0.0,
        })
        if total and cumulative / total >= coverage:
            break
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--discovery-dir", type=Path,
                        default=Path("profiling_results/kernel_discovery"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("profiling_results/kernel_selection"))
    args = parser.parse_args()
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be in (0, 1]")

    manifest: list[dict[str, object]] = []
    for task in args.tasks:
        path = args.discovery_dir / f"{task}_nvtx_kernels.csv"
        if not path.is_file():
            print(f"WARNING: skipping missing {path}")
            continue
        try:
            rows, range_col, kernel_col, total_col = read_nsys_csv(path)
        except ValueError as error:
            print(f"WARNING: skipping {path}: {error}")
            continue
        ranges = sorted({clean_range(row[range_col]) for row in rows
                         if clean_range(row[range_col]).startswith(PREFIX)})
        if not ranges:
            print(f"WARNING: no {PREFIX} ranges in {path}")
            continue
        for nvtx_range in ranges:
            selected = select(rows, range_col, kernel_col, total_col, nvtx_range, args.coverage)
            if not selected:
                continue
            component = nvtx_range.removeprefix(PREFIX)
            output = args.output_dir / f"{task}_{component}_selected_kernels.csv"
            write_csv(output, selected)
            for row in selected:
                manifest.append({"task": task, **row})
            print(f"{task:16s} {nvtx_range:44s} {len(selected):3d} kernels -> {output}")

    if not manifest:
        raise SystemExit("No component kernels were selected")
    manifest_path = args.output_dir / "all_selected_kernels.csv"
    write_csv(manifest_path, manifest)
    print(f"Wrote combined manifest: {manifest_path}")


if __name__ == "__main__":
    main()
