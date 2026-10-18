"""Plot fine-grained local agentic Upsampler/Generator SM scaling curves."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = ("agentic_upsampler", "agentic_generator")
LABELS = {
    "agentic_upsampler": "Local prompt upsampler",
    "agentic_generator": "Upsampled T2I generator",
}
PHASE_LABELS = {"cold": "Cold", "measured": "Warm"}

COMPONENTS = {
    "agentic_upsampler": ("reasoner_prefill", "reasoner_decode"),
    "agentic_generator": ("condition_prepare", "diffusion_sampling", "vae_decode"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("profiling_results/sm_scaling.csv"))
    parser.add_argument("--components-dir", type=Path, default=Path("profiling_results/components"))
    parser.add_argument("--output-dir", type=Path, default=Path("profiling_results/plots_local_agentic"))
    parser.add_argument("--full-sm-count", type=int, default=170)
    parser.add_argument("--phase", choices=("cold", "measured"), default="measured")
    parser.add_argument("--pies-only", action="store_true", help="Plot full-SM shares without loading sweep CSV.")
    return parser.parse_args()


def load_sweep(path: Path, phase: str) -> dict[tuple[str, str], list[dict[str, float]]]:
    curves: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["task"] not in TASKS:
                continue
            if "phase" in row and row["phase"] != phase:
                continue
            curves[(row["task"], row["component"])].append(
                {
                    "sm": float(row["actual_sm_count"] or row["requested_sm_count"]),
                    "median": float(row.get("gpu_total_median_ms") or row["gpu_total_ms"]),
                    "min": float(row["gpu_total_min_ms"]),
                    "max": float(row["gpu_total_max_ms"]),
                }
            )
    for points in curves.values():
        points.sort(key=lambda point: point["sm"])
    return curves


def load_baselines(directory: Path, phase: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for task in TASKS:
        for path in sorted(directory.glob(f"{task}_full_run*.json")):
            per_pass: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
            for record in json.loads(path.read_text(encoding="utf-8"))["records"]:
                component = record["component"]
                record_phase = record.get("phase")
                pass_index = record.get("pass_index")
                if (
                    component in COMPONENTS[task]
                    and record["gpu_ms"] is not None
                    and record_phase == phase
                    and pass_index is not None
                ):
                    per_pass[(record_phase, int(pass_index))][component] += float(record["gpu_ms"])
            for totals in per_pass.values():
                for component, total in totals.items():
                    values[(task, component)].append(total)
    return {key: statistics.median(totals) for key, totals in values.items()}


def draw_pies(
    baselines: dict[tuple[str, str], float],
    output_dir: Path,
    phase: str,
) -> None:
    """Plot full-SM component shares with percentages and absolute times."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    palette = plt.get_cmap("tab10")
    for axis, task in zip(axes, TASKS):
        values = [(component, baselines.get((task, component), 0.0)) for component in COMPONENTS[task]]
        values = [(component, value) for component, value in values if value > 0]
        if not values:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            axis.axis("off")
            continue
        names, times = zip(*values)
        total = sum(times)

        def percentage_label(percent: float) -> str:
            return f"{percent:.1f}%" if percent >= 5.0 else ""

        wedges, _texts, autotexts = axis.pie(
            times,
            labels=None,
            autopct=percentage_label,
            pctdistance=0.68,
            startangle=90,
            counterclock=False,
            colors=[palette(index) for index in range(len(names))],
            radius=0.86,
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )
        for text in autotexts:
            text.set_fontsize(10)
            text.set_fontweight("semibold")

        def legend_label(name: str, time: float) -> str:
            time_text = f"{time / 1000:.3f} s" if time >= 1000 else f"{time:.3f} ms"
            return f"{name} — {time_text} ({time / total:.1%})"

        axis.legend(
            wedges,
            [legend_label(name, time) for name, time in values],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
            fontsize=9.5,
        )
        axis.set_title(f"{LABELS[task]}\nTotal: {total / 1000:.3f} s", fontsize=13)
    figure.suptitle(f"Local agentic T2I full-SM component shares ({PHASE_LABELS[phase]})", fontsize=16)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        destination = output_dir / f"local_agentic_component_time_breakdown_{phase}.{extension}"
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        print(f"Wrote {destination}")
    plt.close(figure)


def draw(
    curves: dict[tuple[str, str], list[dict[str, float]]],
    baselines: dict[tuple[str, str], float],
    output_dir: Path,
    full_sm_count: int,
    *,
    ratio_axis: bool,
    phase: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for axis, task in zip(axes, TASKS):
        for component in COMPONENTS[task]:
            points = curves.get((task, component), [])
            baseline = baselines.get((task, component))
            if not points or baseline is None:
                continue
            x = [point["sm"] / full_sm_count if ratio_axis else point["sm"] for point in points]
            y = [baseline / point["median"] for point in points]
            low = [baseline / point["max"] for point in points]
            high = [baseline / point["min"] for point in points]
            axis.plot(x, y, marker="o", linewidth=2, markersize=4, label=component)
            if any(upper > lower for lower, upper in zip(low, high)):
                axis.fill_between(x, low, high, alpha=0.15)
        if ratio_axis:
            axis.plot([0, 1], [0, 1], "--", color="0.35", linewidth=1.4, label="ideal linear")
            axis.set_xlim(0, 1.02)
            axis.set_xlabel("SM ratio (actual SM / full SM)")
        else:
            axis.axhline(1.0, linestyle="--", color="0.35", linewidth=1.4)
            axis.set_xlim(0, full_sm_count + 4)
            axis.set_xlabel("SM count")
        axis.set_title(LABELS[task])
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Scaling = T(full SM) / T(restricted SM)")
    figure.suptitle(f"Local agentic T2I fine-grained component SM scaling ({phase})")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = ("local_agentic_speedup_vs_sm_ratio" if ratio_axis else "local_agentic_speedup_vs_sm_count") + f"_{phase}"
    for extension in ("png", "pdf"):
        destination = output_dir / f"{stem}.{extension}"
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        print(f"Wrote {destination}")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    baselines = load_baselines(args.components_dir, args.phase)
    draw_pies(baselines, args.output_dir, args.phase)
    if args.pies_only:
        return
    curves = load_sweep(args.csv, args.phase)
    missing = [
        f"{task}/{component}" for task in TASKS for component in COMPONENTS[task] if (task, component) not in baselines
    ]
    if missing:
        raise SystemExit(f"Missing full-SM baselines: {', '.join(missing)}")
    for ratio_axis in (False, True):
        draw(curves, baselines, args.output_dir, args.full_sm_count, ratio_axis=ratio_axis, phase=args.phase)


if __name__ == "__main__":
    main()
