"""Select kernels covering a target fraction of an NVTX component's GPU time."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

DEFAULT_DIR = Path("profiling_results/kernel_discovery")


def column(names: list[str], prefix: str) -> str:
    for name in names:
        if name.strip().lower().startswith(prefix.lower()):
            return name
    raise SystemExit(f"No CSV column begins with {prefix!r}: {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--task", help="Derive standard paths from a task name, e.g. t2v.")
    source.add_argument("--input", type=Path, help="NVTX kernel-summary CSV from nsys stats.")
    parser.add_argument("--component", required=True, help="Component name or full NVTX range.")
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--output", type=Path, help="Optional output override.")
    parser.add_argument("--discovery-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be in (0, 1]")

    nvtx_range = (args.component if args.component.startswith("cosmos.component.")
                  else f"cosmos.component.{args.component}")
    component = nvtx_range.removeprefix("cosmos.component.")
    if args.task:
        input_path = args.discovery_dir / f"{args.task}_nvtx_kernels.csv"
        output_path = args.output or args.discovery_dir / f"{args.task}_{component}_selected_kernels.csv"
    else:
        input_path, output_path = args.input, args.output
    assert input_path is not None
    if not input_path.is_file():
        raise SystemExit(f"Input CSV does not exist: {input_path}")

    lines = input_path.read_text().splitlines()
    try:
        header_index = next(i for i, line in enumerate(lines) if line.startswith("NVTX Range,"))
    except StopIteration:
        raise SystemExit(f"Could not find the NVTX Range CSV header in {input_path}") from None
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    rows, names = list(reader), reader.fieldnames or []
    if not rows:
        raise SystemExit(f"Input CSV is empty: {input_path}")
    range_col = column(names, "nvtx range")
    kernel_col = column(names, "kernel name")
    total_col = column(names, "total time")
    matching = [row for row in rows if row[range_col].strip().lstrip(":") == nvtx_range]
    if not matching:
        available = sorted({row[range_col].strip() for row in rows if row[range_col].strip()})
        raise SystemExit(f"No rows matched {nvtx_range!r}. Available: {', '.join(available)}")

    ranked = sorted(
        ((row, float(row[total_col].replace(",", "").strip())) for row in matching),
        key=lambda item: item[1], reverse=True,
    )
    total = sum(value for _, value in ranked)
    cumulative = 0.0
    selected = []
    print(f"Input: {input_path}\nRange: {nvtx_range}\nTotal kernel time: {total:.3f}")
    for rank, (row, value) in enumerate(ranked, 1):
        cumulative += value
        fraction = cumulative / total if total else 0.0
        selected.append({"rank": rank, "nvtx_range": nvtx_range,
                         "kernel_name": row[kernel_col], "total_time": value,
                         "share": value / total if total else 0.0,
                         "cumulative_share": fraction})
        print(f"{rank:3d} time={value:14.3f} cumulative={fraction:7.2%} kernel={row[kernel_col]}")
        if fraction >= args.coverage:
            break

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        print(f"Wrote {len(selected)} selected kernels to {output_path}")


if __name__ == "__main__":
    main()
