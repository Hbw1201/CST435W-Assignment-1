"""Join ordinary timings with per-process CUDA overlaps and draw the batch comparison."""
import argparse
import csv
import json
from math import fsum, isclose
from pathlib import Path
from statistics import fmean, median, quantiles, stdev

from analyze_gpu_trace import save


def read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def quartiles(values):
    if len(values) == 1:
        return values[0], values[0]
    lower, _, upper = quantiles(values, n=4, method="inclusive")
    return lower, upper


def summarize(root):
    timed = read(root / "timed/runs.csv")
    traced = read(root / "traced/runs.csv")
    devices = {(int(r["batch"]), int(r["repeat"]), int(r["cuda_device"])): r
               for r in read(root / "kernel_timeline.csv")}
    config = [json.loads((root / tag / "metadata.json").read_text()) for tag in ("timed", "traced")]
    batches = sorted(config[0]["batches"])
    if batches != sorted(config[1]["batches"]):
        raise ValueError("Different batch grids in the two passes")
    for meta, rows in zip(config, [timed, traced]):
        expected = {(b, i) for b in batches for i in range(1, meta["repeats"] + 1)}
        if {(int(r["batch"]), int(r["repeat"])) for r in rows} != expected or len(rows) != len(expected):
            raise ValueError("Incomplete or duplicated requests")
        if any(int(r["mismatch_vs_cpu"]) or int(r["correct"]) != meta["correct_per_request"]
               or int(r["n_test"]) != meta["n_test"] for r in rows):
            raise ValueError("Unexpected sample or prediction count")
    if len(devices) != 2 * len(traced):
        raise ValueError("Expected two per-GPU traces for each traced request")
    concurrent = []
    for r in traced:
        batch, repeat = int(r["batch"]), int(r["repeat"])
        a, b = [devices[batch, repeat, i] for i in (0, 1)]
        wall, both = float(a["trace_wall_ms"]), float(a["concurrent_kernel_ms"])
        if (a["trace_wall_ms"] != b["trace_wall_ms"] or a["concurrent_kernel_ms"] != b["concurrent_kernel_ms"]
                or not 0 <= both <= min(float(x["kernel_union_ms"]) for x in (a, b)) + 1e-9):
            raise ValueError("Inconsistent GPU intersection")
        concurrent.append(dict(batch=batch, repeat=repeat, position=int(r["position"]),
                               trace_wall_ms=wall, both_kernel_ms=both, overlap_pct=100 * both / wall,
                               gpu0_kernel_ms=float(a["kernel_union_ms"]), gpu1_kernel_ms=float(b["kernel_union_ms"])))
    summary = []
    for batch in batches:
        group = [r for r in timed if int(r["batch"]) == batch]
        profiles = [r for r in concurrent if r["batch"] == batch]
        times = [float(r["wall_ms"]) for r in group]
        mean_ms = fmean(times)
        sample_sd_ms = stdev(times) if len(times) > 1 else None
        percentages = [r["overlap_pct"] for r in profiles]
        wall = fsum(r["trace_wall_ms"] for r in profiles)
        overlap = fsum(r["both_kernel_ms"] for r in profiles)
        tq1, tq3 = quartiles(times)
        oq1, oq3 = quartiles(percentages)
        summary.append(dict(batch=batch, chunks_per_gpu=int(group[0]["chunks_per_gpu"]),
                            timed_runs=len(times), traced_runs=len(profiles), median_wall_ms=median(times),
                            mean_wall_ms=mean_ms, sample_sd_wall_ms=sample_sd_ms,
                            cv_wall_pct=100 * sample_sd_ms / mean_ms if sample_sd_ms is not None else None,
                            wall_p25_ms=tq1, wall_p75_ms=tq3, wall_min_ms=min(times), wall_max_ms=max(times),
                            images_per_s=1000 * int(group[0]["n_test"]) / median(times),
                            both_kernel_span_pct=100 * overlap / wall,
                            mean_request_overlap_pct=fmean(percentages),
                            sample_sd_request_overlap_pp=stdev(percentages) if len(percentages) > 1 else None,
                            overlap_p25_pct=oq1, overlap_p75_pct=oq3,
                            overlap_min_pct=min(percentages), overlap_max_pct=max(percentages),
                            mean_traced_wall_ms=wall / len(profiles), mean_both_kernel_ms=overlap / len(profiles),
                            gpu0_kernel_span_pct=100 * fsum(r["gpu0_kernel_ms"] for r in profiles) / wall,
                            gpu1_kernel_span_pct=100 * fsum(r["gpu1_kernel_ms"] for r in profiles) / wall,
                            gpu0_peak_alloc_mib=max(float(r["gpu0_peak_alloc_mib"]) for r in group),
                            gpu1_peak_alloc_mib=max(float(r["gpu1_peak_alloc_mib"]) for r in group),
                            correct=int(group[0]["correct"]), mismatch_vs_cpu=0))
    by_batch = {r["batch"]: r for r in summary}
    for r in read(root / "kernel_summary.csv"):
        if not isclose(float(r["both_gpus_kernel_span_pct"]),
                       by_batch[int(r["batch"])]["both_kernel_span_pct"], rel_tol=1e-12):
            raise ValueError("Batch and device aggregates disagree")
    save(root / "overlap_runs.csv", concurrent)
    save(root / "summary.csv", summary)
    return summary


def draw(rows, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, (time_ax, overlap_ax) = plt.subplots(1, 2, figsize=(8.4, 4.4), sharey=True)
    for ax, center, low, high, color, marker, title, xlabel in [
        (time_ax, "median_wall_ms", "wall_p25_ms", "wall_p75_ms", "#08666a", "o",
         "End-to-end time", "Milliseconds per 10,000 images"),
        (overlap_ax, "both_kernel_span_pct", "overlap_p25_pct", "overlap_p75_pct", "#a05224", "D",
         "Both GPUs executing KNN kernels", "Share of request time (%)"),
    ]:
        for y, row in enumerate(rows):
            ax.hlines(y, row[low], row[high], colors=color, linewidth=3, alpha=0.55, zorder=2)
            ax.scatter(row[center], y, color=color, marker=marker, s=32, zorder=3)
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=9)
        ax.set_xlabel(xlabel)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#d9dfe2", linewidth=0.7)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=8)
    time_ax.set_xlim(0, 1.10 * max(max(r["wall_p75_ms"], r["median_wall_ms"]) for r in rows))
    overlap_ax.set_xlim(0, 100)
    overlap_ax.set_xticks([0, 25, 50, 75, 100])
    time_ax.set_yticks(range(len(rows)), [f"{r['batch']:,}" for r in rows])
    time_ax.invert_yaxis()
    time_ax.set_ylim(len(rows) - 0.5, -0.5)
    time_ax.set_ylabel("Batch size per GPU (images)")
    fastest = min(rows, key=lambda r: r["median_wall_ms"])
    busiest = max(rows, key=lambda r: r["both_kernel_span_pct"])
    headline = (f"Lowest median time at batch {fastest['batch']:,}. "
                f"Highest kernel overlap at batch {busiest['batch']:,}.")
    fig.text(0.04, 0.96, headline, fontsize=12, fontweight="bold", va="top")
    fig.text(0.04, 0.10, "MNIST exact 5-NN on two H100s. Each GPU holds the training set and processes 5,000 queries.", fontsize=9)
    fig.text(0.04, 0.05, f"Points show median time ({rows[0]['timed_runs']} runs) and pooled kernel overlap "
             f"({rows[0]['traced_runs']} traces). Bars show the middle 50% of requests.", fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.25, top=0.79, wspace=0.22)
    fig.savefig(root / "batch_sweep.svg", bbox_inches="tight", pad_inches=0.13)
    fig.savefig(root / "batch_sweep.png", dpi=170, bbox_inches="tight", pad_inches=0.13)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    draw(summarize(args.results), args.results)
