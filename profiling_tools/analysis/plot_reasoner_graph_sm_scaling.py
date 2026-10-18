#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

GC_RE = re.compile(
    r"GC decode baseline:\s+requested_generator=(\d+)\s+actual_generator=(\d+)\s+"
    r"actual_reasoner=(\d+)\s+seconds=([0-9.]+)"
)
FULL_RE = re.compile(r"Full-SM decode baseline:\s*([0-9.]+)\s*s")
HASH_RE = re.compile(r"validation .*?sha256=([0-9a-f]{64})")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("profiling_results/reasoner_graph_sm_scaling"))
    parser.add_argument("--output-dir", type=Path, default=Path("profiling_results/reasoner_graph_sm_scaling/plots"))
    parser.add_argument("--total-sms", type=int, default=170)
    parser.add_argument(
        "--implementation",
        choices=("graph", "static_eager"),
        default="graph",
        help="Controls plot labels and output filenames; log parsing is identical.",
    )
    parser.add_argument(
        "--expected-points",
        type=int,
        default=21,
        help="Required number of completed restricted-SM points; use 0 to disable the check.",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    prefix = "reasoner_graph" if args.implementation == "graph" else "reasoner_static_eager"
    implementation_title = "persistent CUDA Graph" if args.implementation == "graph" else "static-KV eager"
    full_text = (args.input_dir / "full_sm.log").read_text(errors="replace")
    full_match = FULL_RE.search(full_text)
    if full_match is None:
        raise ValueError("full_sm.log does not contain a completed baseline")
    full_s = float(full_match.group(1))
    hashes = set(HASH_RE.findall(full_text))
    rows = []
    for path in sorted(args.input_dir.glob("requested_reasoner_sm*.log")):
        if path.stat().st_size == 0:
            print(f"Skipping empty stale log: {path}")
            continue
        text = path.read_text(errors="replace")
        match = GC_RE.search(text)
        if match is None:
            raise ValueError(f"Incomplete log: {path}")
        requested_match = re.search(r"sm(\d+)", path.stem)
        assert requested_match is not None
        requested_reasoner = int(requested_match.group(1))
        requested_generator, actual_generator, actual_reasoner = map(int, match.groups()[:3])
        seconds = float(match.group(4))
        file_hashes = set(HASH_RE.findall(text))
        if len(file_hashes) != 1:
            raise ValueError(f"Expected one token hash in {path}, found {file_hashes}")
        hashes.update(file_hashes)
        rows.append({
            "requested_reasoner_sm": requested_reasoner,
            "requested_generator_sm": requested_generator,
            "actual_reasoner_sm": actual_reasoner,
            "actual_generator_sm": actual_generator,
            "sm_ratio": actual_reasoner / args.total_sms,
            "decode_seconds": seconds,
            "speedup_tfull_over_tsm": full_s / seconds,
            "slowdown_tsm_over_tfull": seconds / full_s,
            "token_sha256": next(iter(file_hashes)),
        })
    if args.expected_points and len(rows) != args.expected_points:
        raise ValueError(
            f"Expected {args.expected_points} completed restricted-SM points, found {len(rows)}"
        )
    if len(hashes) != 1:
        raise ValueError(f"Token correctness failure: {sorted(hashes)}")
    rows.sort(key=lambda row: row["actual_reasoner_sm"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{prefix}_sm_scaling.csv"
    fields = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "requested_reasoner_sm": args.total_sms,
            "requested_generator_sm": 0,
            "actual_reasoner_sm": args.total_sms,
            "actual_generator_sm": 0,
            "sm_ratio": 1.0,
            "decode_seconds": full_s,
            "speedup_tfull_over_tsm": 1.0,
            "slowdown_tsm_over_tfull": 1.0,
            "token_sha256": next(iter(hashes)),
        })

    sms = [row["actual_reasoner_sm"] for row in rows] + [args.total_sms]
    ratios = [row["sm_ratio"] for row in rows] + [1.0]
    times = [row["decode_seconds"] for row in rows] + [full_s]
    speedup = [row["speedup_tfull_over_tsm"] for row in rows] + [1.0]

    # Match one subplot of the 14 x 5.2 two-panel agentic T2V figure.
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.plot(sms, times, "o-", color="#c83e4d", linewidth=2, markersize=4)
    ax.axhline(full_s, color="0.35", linestyle="--", linewidth=1.4, label=f"Full SM: {full_s:.3f} s")
    ax.set(xlabel="Actual Reasoner SM count", ylabel="Decode time (s)",
           title=f"Reasoner {implementation_title} decode time (log scale)")
    ax.set_yscale("log")
    ax.set_xlim(0, 174)
    ax.grid(alpha=0.25)
    for index in (0, 4, len(sms) - 2, len(sms) - 1):
        ax.annotate(
            f"{times[index]:.2f}s",
            (sms[index], times[index]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{prefix}_decode_time_vs_sm.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.plot(ratios, speedup, "o-", color="#2369a1", linewidth=2, markersize=4, label=args.implementation)
    ax.plot([0, 1], [0, 1], "--", color="0.35", linewidth=1.4, label="ideal linear")
    ax.set(xlabel="SM ratio (actual Reasoner SM / 170)",
           ylabel="Scaling = T(full SM) / T(restricted SM)",
           title="Reasoner decode")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{prefix}_speedup_vs_sm_ratio.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Validated token SHA-256: {next(iter(hashes))}")
    print(f"Full-SM decode: {full_s:.3f} s")
    print(f"Wrote {csv_path}")
    print(f"Wrote two PNG files in {args.output_dir}")


if __name__ == "__main__":
    main()
