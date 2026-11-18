"""Plot independent-session timing estimates and a separately profiled CUDA request."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plot_results import BLUE, SKY, VIOLET, LIME, INK, MUTED, RULE, WIDTH, PRINT_STYLE, read, sheet


def figure(folder, trace):
    sessions = read(folder, "session_summary.csv")
    summary = read(folder, "summary.csv")
    scaling = read(folder, "scaling.csv")
    ratios = read(folder, "session_scaling.csv")
    segments = read(trace, "segments.csv")
    sizes = [100, 1000, 10000]
    height = 288
    fig, art = sheet(height)
    for x, y, letter, title in [(1, 10, "a", "Request latency"),
                               (242, 10, "b", "Scaling to two H100s"),
                               (1, 190, "c", "Where device time goes")]:
        art.text(x, y, letter, va="center", weight="bold", fontsize=11)
        art.text(x + 13, y, title, va="center", weight="bold", fontsize=9.5)
    a = fig.add_axes([52/WIDTH, (height-140)/height, 164/WIDTH, 94/height])
    styles = [("cpu", "CPU", BLUE, "o"), ("gpu1", "1 H100", SKY, "s"),
              ("gpu2", "2 H100s", VIOLET, "^")]
    for mode, label, colour, marker in styles:
        means = []
        for x, size in enumerate(sizes):
            values = [1000 * float(r["mean_wall_s"]) for r in sessions
                      if int(r["n_test"]) == size and r["mode"] == mode]
            row = next(r for r in summary if int(r["n_test"]) == size and r["mode"] == mode)
            if len(values) != 3 or not np.isclose(np.mean(values), 1000*float(row["mean_wall_s"])):
                raise ValueError("Session means disagree with the summary")
            means.append(np.mean(values))
            a.scatter(x + np.array([-.065, 0, .065]), values, s=12, marker=marker,
                      facecolor="white", edgecolor=colour, linewidth=.75, zorder=4)
        a.plot(range(3), means, color=colour, marker=marker, ms=4, lw=1, label=label, zorder=5)
    a.set(yscale="log", ylim=(25, 32000), xlim=(-.18, 2.18), xticks=range(3),
          xticklabels=["100", "1,000", "10,000"], xlabel="Queries per request",
          ylabel="Wall time (ms)", yticks=[100, 1000, 10000], yticklabels=["100", "1,000", "10,000"])
    a.legend(loc="lower left", bbox_to_anchor=(-.10, 1.04), ncol=3, frameon=False,
             handlelength=1.1, handletextpad=.35, columnspacing=.9, borderaxespad=0, fontsize=8)
    a.minorticks_off()
    a.grid(axis="y", color=RULE, lw=.5)
    b = fig.add_axes([283/WIDTH, (height-140)/height, 163/WIDTH, 94/height])
    for x, label in [(1, "Parity"), (2, "Ideal")]:
        b.axvline(x, color=RULE, linestyle=(0, (3, 3)), lw=.8, zorder=0)
        b.text(x, 2.78, label, ha="center", fontsize=8, color=MUTED, clip_on=False)
    for y, size in zip([2, 1, 0], sizes):
        row = next(r for r in scaling if int(r["n_test"]) == size)
        value, low, high = [float(row["gpu2_speedup" + suffix]) for suffix in ("_mean", "_ci95_low", "_ci95_high")]
        points = [float(r["gpu2_speedup"]) for r in ratios if int(r["n_test"]) == size]
        b.scatter(points, y+np.array([-.12, 0, .12]), marker="o", s=17,
                  facecolor="white", edgecolor=BLUE, linewidth=.8, zorder=3)
        b.errorbar(value, y, xerr=[[value-low], [high-value]], fmt="D", color=INK,
                   ms=3.3, capsize=2.3, lw=.85, zorder=4)
        b.text(1.72, y, f"{float(row['gpu2_efficiency_pct_mean']):.1f}%", va="center",
               ha="center", fontsize=8.5, color=VIOLET)
    b.text(1.72, 2.78, "Efficiency", ha="center", fontsize=8, color=MUTED, clip_on=False)
    b.set(xlim=(.89, 2.05), ylim=(-.5, 2.5), xticks=[1, 1.5, 2],
          xticklabels=["1.0", "1.5", "2.0"], yticks=[0, 1, 2],
          yticklabels=["10,000", "1,000", "100"], xlabel="One-card / two-card time")
    b.tick_params(axis="y", length=0)

    c = fig.add_axes([83/WIDTH, (height-258)/height, 363/WIDTH, 52/height])
    activities = [("Host-to-Device", "Upload", SKY), ("Kernel", "KNN kernels", VIOLET),
                  ("Device-to-Host", "Return", LIME)]
    selected = [r for r in segments if r["placement"] == "per_request" and int(r["repeat"]) == 2]
    if not selected:
        raise ValueError("The preselected second traced request is missing")
    maximum = 0
    for y, count, device in [(2, 1, 0), (1, 2, 0), (0, 2, 1)]:
        rows = [r for r in selected if int(r["gpu_count"]) == count and int(r["cuda_device"]) == device]
        walls = {float(r["trace_wall_ms"]) for r in rows}
        if len(walls) != 1:
            raise ValueError("Trace lane has inconsistent request boundaries")
        wall = walls.pop()
        maximum = max(maximum, wall)
        c.barh(y, wall, height=.52, color="#F0F2F3", edgecolor=RULE, lw=.35, zorder=0)
        for key, label, colour in activities:
            intervals = [(float(r["start_ms"]), float(r["end_ms"])-float(r["start_ms"]))
                         for r in rows if r["activity"] == key]
            if not intervals:
                raise ValueError("Missing traced activity: " + key)
            c.broken_barh(intervals, (y-.26, .52), facecolors=colour, linewidth=0, zorder=2)
        # Return transfers are only microseconds long. A small tick locates them without widening the interval.
        ends = [float(r["start_ms"]) for r in rows if r["activity"] == "Device-to-Host"]
        c.scatter(ends, [y]*len(ends), marker="|", c=LIME, s=48, linewidth=1, zorder=4)
    c.set(xlim=(0, np.ceil(maximum/20)*20), ylim=(-.55, 2.55), yticks=[2, 1, 0],
          yticklabels=["1 card: GPU 0", "2 cards: GPU 0", "2 cards: GPU 1"],
          xlabel="Elapsed from request start (ms)")
    c.tick_params(axis="y", length=0, labelsize=8)
    c.tick_params(axis="x", labelsize=8)
    legend = [Patch(facecolor=col, label=label) if key != "Device-to-Host" else
              Line2D([], [], marker="|", color=col, ls="", label=label)
              for key, label, col in activities]
    art.legend(handles=legend, loc="center right", bbox_to_anchor=(.99, 1-190/height),
               frameon=False, ncol=3, handlelength=.9, handletextpad=.35,
               columnspacing=.9, fontsize=8)
    for ax in [a, b, c]:
        ax.spines[["top", "right"]].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color(RULE)
            spine.set_linewidth(.6)
        ax.tick_params(width=.55, length=3)
    return fig


def plots(folder, trace, out):
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PRINT_STYLE):
        fig = figure(folder, trace)
        for extension in ["pdf", "svg", "png"]:
            metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else {"Date": None}
            fig.savefig(out / ("performance-sessions." + extension), dpi=300,
                        metadata=metadata if extension != "png" else None)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--trace", type=Path, default=Path("results/trace_repeat"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    args = p.parse_args()
    plots(args.results, args.trace, args.out)
