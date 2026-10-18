"""Create the two summary figures for Cosmos3 component SM scaling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TASKS = ("t2i", "t2v", "action", "reasoner_video")
TASK_LABELS = {
    "t2i": "T2I",
    "t2v": "T2V",
    "action": "Action Policy",
    "reasoner_video": "Video-to-Text",
}
FULL_FILES = {
    "t2i": "t2i_full.json",
    "t2v": "t2v_full.json",
    "action": "action_full.json",
    "reasoner_video": "reasoner_video_full.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("profiling_results/sm_scaling.csv"))
    parser.add_argument("--components-dir", type=Path, default=Path("profiling_results/components"))
    parser.add_argument("--output-dir", type=Path, default=Path("profiling_results/plots"))
    parser.add_argument("--full-sm-count", type=int, default=170)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_sweep(path: Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    curves: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]
            component = row["component"]
            sm_count = int(row["actual_sm_count"] or row["requested_sm_count"])
            # Total stage latency is required because reasoner_decode contains
            # one timing record per generated token.
            latency_ms = float(row["gpu_total_ms"])
            curves[(task, component)].append((sm_count, latency_ms))
    for points in curves.values():
        points.sort()
    return curves


def load_baselines(directory: Path) -> dict[tuple[str, str], float]:
    baselines: dict[tuple[str, str], float] = {}
    for task, filename in FULL_FILES.items():
        stem = Path(filename).stem
        paths = sorted(directory.glob(f"{stem}*.json"))
        if not paths:
            print(f"WARNING: missing full-GPU baseline matching: {directory / (stem + '*.json')}")
            continue
        totals_by_component: dict[str, list[float]] = defaultdict(list)
        for path in paths:
            run_totals: dict[str, float] = defaultdict(float)
            for record in json.loads(path.read_text())["records"]:
                if record["gpu_ms"] is not None:
                    run_totals[record["component"]] += float(record["gpu_ms"])
            for component, latency_ms in run_totals.items():
                totals_by_component[component].append(latency_ms)
        for component, run_totals in totals_by_component.items():
            baselines[(task, component)] = statistics.median(run_totals)
    return baselines


def component_colors(curves: dict[tuple[str, str], list[tuple[int, float]]]) -> dict[str, object]:
    components = sorted({component for _, component in curves})
    palette = plt.get_cmap("tab10")
    return {component: palette(index % 10) for index, component in enumerate(components)}


def save_figure(figure: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    for extension in ("png", "pdf"):
        destination = output_dir / f"{stem}.{extension}"
        figure.savefig(destination, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {destination}")
    plt.close(figure)


def plot_component_pies(
    curves: dict[tuple[str, str], list[tuple[int, float]]],
    baselines: dict[tuple[str, str], float],
    colors: dict[str, object],
    output_dir: Path,
    dpi: int,
) -> None:
    """One 2x2 figure showing major-component shares for all four tasks."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    figure.suptitle("Cosmos3-Edge full-GPU major-component time breakdown", fontsize=16)

    for axis, task in zip(axes.flat, TASKS):
        # Restrict the pie to components that are independently SM-scanned.
        # This excludes overlapping container ranges such as reasoner_e2e.
        components = sorted(component for candidate, component in curves if candidate == task)
        values = [baselines.get((task, component), 0.0) for component in components]
        valid = [(component, value) for component, value in zip(components, values) if value > 0]
        if not valid:
            axis.text(0.5, 0.5, "No full-GPU component data", ha="center", va="center")
            axis.set_title(TASK_LABELS[task])
            axis.axis("off")
            continue
        components, values = zip(*valid)
        labels = [f"{component}\n{value:.1f} ms" for component, value in valid]
        axis.pie(
            values,
            labels=labels,
            colors=[colors[component] for component in components],
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            textprops={"fontsize": 8.5},
        )
        axis.set_title(TASK_LABELS[task])

    figure.text(
        0.5,
        0.015,
        "Shares are normalized over independently profiled major components; overlapping e2e ranges are excluded.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(figure, output_dir, "component_time_breakdown", dpi)


def plot_speedup(
    curves: dict[tuple[str, str], list[tuple[int, float]]],
    baselines: dict[tuple[str, str], float],
    colors: dict[str, object],
    output_dir: Path,
    full_sm_count: int,
    dpi: int,
) -> None:
    """One 2x2 figure of T_full / T_restricted versus provisioned SMs."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    figure.suptitle("Cosmos3-Edge component speedup versus Green Context SM count", fontsize=16)

    for axis, task in zip(axes.flat, TASKS):
        components = sorted(component for candidate, component in curves if candidate == task)
        plotted = False
        for component in components:
            baseline = baselines.get((task, component))
            if baseline is None or baseline <= 0:
                print(f"WARNING: missing baseline for {task}/{component}")
                continue
            points = list(curves[(task, component)]) + [(full_sm_count, baseline)]
            points.sort()
            sms = [sm for sm, _ in points]
            # User-requested traditional speedup definition.
            speedups = [baseline / restricted_ms for _, restricted_ms in points]
            axis.plot(
                sms,
                speedups,
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                label=component,
                color=colors[component],
            )
            plotted = True
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
        axis.set_title(TASK_LABELS[task])
        axis.set_xlabel("SM count")
        axis.set_ylabel("Speedup = T(full 170 SM) / T(restricted)")
        axis.set_xlim(0, full_sm_count + 6)
        axis.set_xticks(list(range(8, 169, 16)) + [full_sm_count])
        axis.grid(True, alpha=0.3)
        if plotted:
            axis.legend(fontsize=8)

    figure.text(
        0.5,
        0.015,
        "1.0 is full-GPU performance; values below 1.0 mean the restricted-SM run is slower.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(figure, output_dir, "speedup_vs_sm_count", dpi)



def plot_speedup_ratio(
    curves: dict[tuple[str, str], list[tuple[int, float]]],
    baselines: dict[tuple[str, str], float],
    colors: dict[str, object],
    output_dir: Path,
    full_sm_count: int,
    dpi: int,
) -> None:
    """Plot T_full / T_restricted against the provisioned fraction of full SMs."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    figure.suptitle("Cosmos3-Edge component speedup versus Green Context SM ratio", fontsize=16)

    for axis, task in zip(axes.flat, TASKS):
        components = sorted(component for candidate, component in curves if candidate == task)
        plotted = False
        for component in components:
            baseline = baselines.get((task, component))
            if baseline is None or baseline <= 0:
                continue
            points = list(curves[(task, component)]) + [(full_sm_count, baseline)]
            points.sort()
            ratios = [sm_count / full_sm_count for sm_count, _ in points]
            speedups = [baseline / restricted_ms for _, restricted_ms in points]
            axis.plot(
                ratios,
                speedups,
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                label=component,
                color=colors[component],
            )
            plotted = True

        # Ideal proportional scaling: r of the SMs provides r of full-GPU throughput.
        axis.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
            label="ideal proportional scaling",
        )
        axis.set_title(TASK_LABELS[task])
        axis.set_xlabel(f"SM ratio = allocated SMs / {full_sm_count}")
        axis.set_ylabel("Speedup = T(full) / T(restricted)")
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([index / 10 for index in range(11)])
        axis.grid(True, alpha=0.3)
        if plotted:
            axis.legend(fontsize=8)

    figure.text(
        0.5,
        0.015,
        "The dashed diagonal is ideal proportional scaling from (0, 0) to (1, 1).",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(figure, output_dir, "speedup_vs_sm_ratio", dpi)


def write_plot_data(
    curves: dict[tuple[str, str], list[tuple[int, float]]],
    baselines: dict[tuple[str, str], float],
    output: Path,
    full_sm_count: int,
) -> None:
    rows = []
    for (task, component), points in sorted(curves.items()):
        baseline = baselines.get((task, component))
        all_points = list(points)
        if baseline is not None:
            all_points.append((full_sm_count, baseline))
        for sm_count, restricted_ms in sorted(all_points):
            rows.append({
                "task": task,
                "component": component,
                "sm_count": sm_count,
                "restricted_gpu_stage_ms": restricted_ms,
                "full_gpu_stage_ms": baseline if baseline is not None else math.nan,
                "speedup_full_over_restricted": (
                    baseline / restricted_ms if baseline is not None and restricted_ms > 0 else math.nan
                ),
                "is_full_gpu": sm_count == full_sm_count,
            })
    if not rows:
        raise ValueError("No SM-scaling rows were found")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = load_sweep(args.csv)
    baselines = load_baselines(args.components_dir)
    colors = component_colors(curves)
    plot_component_pies(curves, baselines, colors, args.output_dir, args.dpi)
    plot_speedup(curves, baselines, colors, args.output_dir, args.full_sm_count, args.dpi)
    plot_speedup_ratio(curves, baselines, colors, args.output_dir, args.full_sm_count, args.dpi)
    write_plot_data(curves, baselines, args.output_dir / "plot_data.csv", args.full_sm_count)


if __name__ == "__main__":
    main()
