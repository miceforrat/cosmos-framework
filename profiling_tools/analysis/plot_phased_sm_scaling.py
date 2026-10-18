"""Plot cold/warm component shares and SM speedup from phased profiles."""

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


TASKS = ("t2i", "t2v", "action", "reasoner_video")
LABELS = {"t2i": "T2I", "t2v": "T2V", "action": "Action Policy", "reasoner_video": "Video-to-Text"}
PHASES = ("cold", "measured")
PHASE_LABELS = {"cold": "Cold", "measured": "Warm"}
OVERLAPPING_COMPONENTS = {"reasoner_e2e"}
COMPONENT_ORDER = {
    "condition_prepare": 0,
    "diffusion_sampling": 1,
    "vae_decode": 2,
    "action_postprocess": 3,
    "vision_encoder": 0,
    "reasoner_prefill": 1,
    "reasoner_decode": 2,
}


def component_sort_key(component: str) -> tuple[int, str]:
    return COMPONENT_ORDER.get(component, len(COMPONENT_ORDER)), component


def load_curves(path: Path) -> dict:
    curves = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sm = int(row["actual_sm_count"] or row["requested_sm_count"])
            curves[(row["phase"], row["task"], row["component"])].append({
                "sm": sm,
                "median": float(row["gpu_total_median_ms"]),
                "min": float(row["gpu_total_min_ms"]),
                "max": float(row["gpu_total_max_ms"]),
            })
    for points in curves.values():
        points.sort(key=lambda point: point["sm"])
    return curves


def load_baselines(directory: Path) -> tuple[dict, dict]:
    samples = defaultdict(list)
    for task in TASKS:
        for path in sorted(directory.glob(f"{task}_full_run*.json")):
            per_pass = defaultdict(float)
            for record in json.loads(path.read_text())["records"]:
                phase = record.get("phase")
                pass_index = record.get("pass_index")
                if phase in PHASES and pass_index is not None and record.get("gpu_ms") is not None:
                    per_pass[(phase, int(pass_index), record["component"])] += float(record["gpu_ms"])
            for (phase, _pass_index, component), total in per_pass.items():
                samples[(phase, task, component)].append(total)
    medians = {key: statistics.median(values) for key, values in samples.items()}
    ranges = {key: (min(values), max(values)) for key, values in samples.items()}
    return medians, ranges


def save(figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)


def plot_shares(phase: str, curves: dict, baselines: dict, output_dir: Path, dpi: int) -> None:
    figure = plt.figure(figsize=(16, 8), constrained_layout=True)
    figure.suptitle(
        f"Cosmos3-Edge {PHASE_LABELS[phase].lower()} full-SM component shares",
        fontsize=17,
        fontweight="semibold",
    )
    outer = figure.add_gridspec(2, 2, wspace=0.06, hspace=0.16)
    palette = plt.get_cmap("tab10")
    for index, task in enumerate(TASKS):
        cell = outer[index // 2, index % 2].subgridspec(
            1, 2, width_ratios=(1.0, 1.35), wspace=0.02
        )
        pie_axis = figure.add_subplot(cell[0, 0])
        legend_axis = figure.add_subplot(cell[0, 1])
        legend_axis.axis("off")

        # Full-SM shares can be plotted before the SM sweep finishes.
        components = sorted({component for candidate_phase, candidate_task, component in baselines
                             if candidate_phase == phase and candidate_task == task
                             and component not in OVERLAPPING_COMPONENTS}, key=component_sort_key)
        values = [(component, baselines.get((phase, task, component), 0.0)) for component in components]
        values = [(component, value) for component, value in values if value > 0]
        if not values:
            pie_axis.text(0.5, 0.5, "No data", ha="center", va="center")
            pie_axis.axis("off")
            continue
        names, times = zip(*values)
        total = sum(times)

        def percentage_label(percent: float) -> str:
            return f"{percent:.1f}%" if percent >= 6.0 else ""

        wedges, _texts, autotexts = pie_axis.pie(
            times,
            labels=None,
            autopct=percentage_label,
            pctdistance=0.67,
            startangle=90,
            counterclock=False,
            colors=[palette(i % 10) for i in range(len(names))],
            radius=0.92,
            wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        )
        for text in autotexts:
            text.set_fontsize(10)
            text.set_fontweight("semibold")

        def legend_label(name: str, time: float) -> str:
            time_text = f"{time:.3f} ms" if time < 1.0 else f"{time:.1f} ms"
            share = time / total
            share_text = "<0.1%" if share < 0.001 else f"{share:.1%}"
            return f"{name}  —  {time_text} ({share_text})"

        legend_axis.legend(
            wedges,
            [legend_label(name, time) for name, time in values],
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            frameon=False,
            fontsize=9.5,
            handlelength=1.2,
            labelspacing=0.55,
        )
        pie_axis.set_title(LABELS[task], fontsize=13, pad=6)
    save(figure, output_dir, f"component_time_breakdown_{phase}", dpi)


def plot_speedup(phase: str, curves: dict, baselines: dict, output_dir: Path, dpi: int,
                 full_sm_count: int) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)
    figure.suptitle(f"Cosmos3-Edge {PHASE_LABELS[phase].lower()} component SM scaling", fontsize=16)
    palette = plt.get_cmap("tab10")
    for axis, task in zip(axes.flat, TASKS):
        components = sorted({component for candidate_phase, candidate_task, component in curves
                             if candidate_phase == phase and candidate_task == task}, key=component_sort_key)
        for index, component in enumerate(components):
            baseline = baselines.get((phase, task, component))
            if baseline is None:
                continue
            points = curves[(phase, task, component)]
            xs = [point["sm"] for point in points] + [full_sm_count]
            ys = [baseline / point["median"] for point in points] + [1.0]
            low = [baseline / point["max"] for point in points] + [1.0]
            high = [baseline / point["min"] for point in points] + [1.0]
            color = palette(index % 10)
            axis.plot(xs, ys, marker="o", linewidth=1.8, markersize=4.5, label=component, color=color)
            axis.fill_between(xs, low, high, color=color, alpha=0.15)
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        axis.set_title(LABELS[task])
        axis.set_xlabel("Actual SM count")
        axis.set_ylabel("Speedup = full-SM time / restricted-SM time")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    save(figure, output_dir, f"speedup_vs_sm_count_{phase}", dpi)



def plot_speedup_ratio(
    phase: str,
    curves: dict,
    baselines: dict,
    output_dir: Path,
    dpi: int,
    full_sm_count: int,
) -> None:
    """Plot phased speedup against the allocated fraction of full-GPU SMs."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=False)
    figure.suptitle(
        f"Cosmos3-Edge {PHASE_LABELS[phase].lower()} component SM-ratio scaling",
        fontsize=16,
    )
    palette = plt.get_cmap("tab10")
    for axis, task in zip(axes.flat, TASKS):
        components = sorted({
            component
            for candidate_phase, candidate_task, component in curves
            if candidate_phase == phase and candidate_task == task
        }, key=component_sort_key)
        plotted = False
        for index, component in enumerate(components):
            baseline = baselines.get((phase, task, component))
            if baseline is None:
                continue
            points = curves[(phase, task, component)]
            xs = [point["sm"] / full_sm_count for point in points] + [1.0]
            ys = [baseline / point["median"] for point in points] + [1.0]
            low = [baseline / point["max"] for point in points] + [1.0]
            high = [baseline / point["min"] for point in points] + [1.0]
            color = palette(index % 10)
            axis.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=component,
                color=color,
            )
            axis.fill_between(xs, low, high, color=color, alpha=0.15)
            plotted = True

        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            label="ideal proportional scaling",
        )
        axis.set_title(LABELS[task])
        axis.set_xlabel(f"SM ratio = actual SMs / {full_sm_count}")
        axis.set_ylabel("Speedup = full-SM time / restricted-SM time")
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([index / 10 for index in range(11)])
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=8)

    figure.text(
        0.5,
        0.012,
        "Shading is the min–max range; the dashed diagonal is ideal proportional scaling.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    save(figure, output_dir, f"speedup_vs_sm_ratio_{phase}", dpi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("profiling_results/sm_scaling_phased_on.csv"))
    parser.add_argument("--components-dir", type=Path, default=Path("profiling_results/components_phased_on"))
    parser.add_argument("--output-dir", type=Path, default=Path("profiling_results/plots_phased_on"))
    parser.add_argument("--full-sm-count", type=int, default=170)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    curves = load_curves(args.csv)
    baselines, _ranges = load_baselines(args.components_dir)
    for phase in PHASES:
        plot_shares(phase, curves, baselines, args.output_dir, args.dpi)
        plot_speedup(phase, curves, baselines, args.output_dir, args.dpi, args.full_sm_count)
        plot_speedup_ratio(phase, curves, baselines, args.output_dir, args.dpi, args.full_sm_count)


if __name__ == "__main__":
    main()
