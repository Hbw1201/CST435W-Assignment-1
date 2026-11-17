"""Rebuild the report figures from the measurement CSV files."""
import argparse
import csv
from math import ceil, floor, log10, log1p
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, NullLocator


WIDTH = 162 * 72 / 25.4  # Printed text width in points
INK, BLUE, VIOLET, TAN, GOLD = "#102449", "#2A5572", "#3D2372", "#DCB281", "#D1B92B"
TEXT, RULE = "#172334", "#C9D2D8"
HARDWARE = {
    "cpu": ("1 CPU thread", "s", TAN),
    "gpu1": ("1 H100", "o", BLUE),
    "gpu2": ("2 H100s", "D", VIOLET),
}
PRINT_STYLE = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT, "pdf.fonttype": 42,
    "svg.fonttype": "none", "svg.hashsalt": "cst435w-report", "savefig.facecolor": "white",
}


def read(folder, name):
    with (folder/name).open(newline="") as f:
        return list(csv.DictReader(f))


def panel(fig, x, top, letter, title):
    height = fig.get_figheight() * 72
    for left, label, size in ((x, letter, 11), (x + 13, title, 10)):
        fig.text(left / WIDTH, 1 - top / height, label, ha="left", va="top",
                 fontsize=size, weight="bold", color=INK)


def bottom_axis(ax):
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(.65)
    ax.yaxis.set_visible(False)
    ax.tick_params(axis="x", length=3, width=.65, pad=3, labelsize=8)


def run_offsets(values, med, points_per_ms):
    """Stack only coincident marks. Their measured time coordinates stay exact."""
    placed = []
    offsets = [0.0] * len(values)
    gap = 4.6
    order = sorted(range(len(values)), key=lambda j: (abs(values[j] - med), j))
    for j in order:
        x = values[j] * points_per_ms
        for step in range(2 * len(values) + 1):
            level = (step + 1) // 2
            y = level * gap if step % 2 else -level * gap
            if all((x - px) ** 2 + (y - py) ** 2 >= gap ** 2 - 1e-8 for px, py in placed):
                placed.append((x, y))
                offsets[j] = y
                break
    return offsets


def performance_figure(folder, rows, sizes):
    summary = {(int(r["n_test"]), r["mode"]): r for r in rows}
    runs = {}
    if (folder / "runs.csv").is_file():
        for r in read(folder, "runs.csv"):
            key = int(r["n_test"]), r["mode"]
            cell = runs.setdefault(key, {})
            repeat = int(r["run"])
            if repeat in cell:
                raise ValueError(f"Duplicated timing for {key}, run {repeat}")
            cell[repeat] = 1000 * float(r["wall_s"])
        for key, row in summary.items():
            values = list(runs.get(key, {}).values())
            expected = [float(row[k]) * 1000 for k in ("min_wall_s", "median_wall_s", "max_wall_s")]
            if (len(values) != int(row["repeats"]) or
                    not np.allclose([min(values), median(values), max(values)], expected)):
                raise ValueError(f"Timed requests disagree with the summary for {key}")

    row_height, bottom = 47, 23
    plot_height = row_height * len(sizes)
    height = plot_height + bottom + 38
    fig = plt.figure(figsize=(WIDTH / 72, height / 72), facecolor="white")
    time_left, time_width = 44, 201
    rate_left, rate_width = 272, WIDTH - 278
    time_ax = fig.add_axes([time_left / WIDTH, bottom / height, time_width / WIDTH, plot_height / height])
    rate_ax = fig.add_axes([rate_left / WIDTH, bottom / height, rate_width / WIDTH, plot_height / height])
    panel(fig, 1, 3, "a", "GPU request time (ms)")
    panel(fig, rate_left - 6, 3, "b", "Throughput (images/s, log scale)")
    key_y = 1 - 24 / height
    note = "Filled symbols mark medians." if runs else "Lines show ranges. Fill marks medians."
    fig.text(121 / WIDTH, key_y, note, va="center", fontsize=8)
    keys = [(4, "gpu1", False), (62, "gpu2", False),
            (rate_left - 3, "cpu", True), (rate_left + 73, "gpu1", True), (rate_left + 132, "gpu2", True)]
    for x, mode, filled in keys:
        label, shape, colour = HARDWARE[mode]
        marker = plt.Line2D([x / WIDTH], [key_y], transform=fig.transFigure, linestyle="none",
                            marker=shape, markersize=4.7, markerfacecolor=colour if filled else "white",
                            markeredgecolor=INK if filled else colour, markeredgewidth=.7)
        fig.add_artist(marker)
        fig.text((x + 6) / WIDTH, key_y, label, va="center", fontsize=8)
    fig.text(1 / WIDTH, (plot_height + bottom) / height, "Images",
             va="center", fontsize=8, weight="bold")

    lo = min(float(r["min_wall_s"]) * 1000 for r in rows if r["mode"] != "cpu")
    hi = max(float(r["max_wall_s"]) * 1000 for r in rows if r["mode"] != "cpu")
    padding = max((hi - lo) * .09, hi * .015, .001)
    low, high = max(0., lo - padding), hi + padding
    time_ax.set_xlim(low, high)
    time_ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    rates = {(n, m): n / float(summary[n, m]["median_wall_s"]) for n in sizes for m in HARDWARE}
    first = floor(log10(min(rates.values())))
    last = ceil(log10(max(rates.values())))
    rate_ax.set_xscale("log")
    rate_ax.set_xlim(10 ** first, max(10 ** (first + 1), max(rates.values()) * 1.38))
    powers = [10. ** k for k in range(first, last + 1) if 10. ** k <= rate_ax.get_xlim()[1]]
    rate_ax.set_xticks(powers, [f"{x/1000:g}k" if x >= 1000 else f"{x:g}" for x in powers])
    rate_ax.xaxis.set_minor_locator(NullLocator())
    for ax in (time_ax, rate_ax):
        ax.set_ylim(0, len(sizes))
        bottom_axis(ax)
        for y in range(1, len(sizes)):
            ax.axhline(y, color=RULE, linewidth=.4, zorder=0)

    for i, n in enumerate(sizes):
        center = len(sizes) - i - .5
        fig.text(1 / WIDTH, (bottom + center * row_height) / height, f"{n:,}",
                 va="center", fontsize=9.2, weight="bold")
        for mode, lane in (("gpu1", center + .23), ("gpu2", center - .23)):
            _, shape, colour = HARDWARE[mode]
            row = summary[n, mode]
            med = float(row["median_wall_s"]) * 1000
            left, right = (float(row[k]) * 1000 for k in ("min_wall_s", "max_wall_s"))
            time_ax.hlines(lane, left, right, color=colour, linewidth=.7, zorder=1)
            if runs:
                repeats = sorted(runs[n, mode].items())
                values = [value for _, value in repeats]
                offsets = run_offsets(values, med, time_width / (high - low))
                for (repeat, value), offset in zip(repeats, offsets):
                    time_ax.plot(value, lane + offset / row_height, marker=shape, markersize=3.6,
                                 color=colour, markerfacecolor="white", markeredgewidth=.8,
                                 gid=f"time-{n}-{mode}-run-{repeat}", zorder=3)
            time_ax.plot(med, lane, marker=shape, markersize=4.2, color=colour, markeredgewidth=.6,
                         gid=f"time-{n}-{mode}-median", zorder=4)
            before = right > high - .24 * (high - low)
            time_ax.annotate(f"{med:,.2f}", (left if before else right, lane),
                             xytext=(-5 if before else 5, 0), textcoords="offset points",
                             ha="right" if before else "left", va="center", fontsize=8)

        positions = [center + .24, center, center - .24]
        row_rates = [rates[n, mode] for mode in HARDWARE]
        rate_ax.hlines(center, min(row_rates), max(row_rates), color="#B5C3CB", linewidth=.8)
        for (mode, (_, shape, colour)), y in zip(HARDWARE.items(), positions):
            rate = rates[n, mode]
            rate_ax.vlines(rate, min(y, center), max(y, center), color="#B5C3CB", linewidth=.8)
            rate_ax.plot(rate, y, marker=shape, markersize=4.6, markerfacecolor=colour,
                         markeredgecolor=INK, markeredgewidth=.55, zorder=3, gid=f"rate-{n}-{mode}")
            fraction = log10(rate / rate_ax.get_xlim()[0]) / log10(rate_ax.get_xlim()[1] / rate_ax.get_xlim()[0])
            before = fraction > .79
            rate_ax.annotate(f"{rate:,.0f}", (rate, y), xytext=(-5 if before else 5, 0),
                             textcoords="offset points", ha="right" if before else "left",
                             va="center", fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=.15))
    return fig


def confusion_figure(folder, size, expected_correct):
    rows = [r for r in read(folder, "confusion.csv") if int(r["n_test"]) == size]
    counts = {(int(r["true_label"]), int(r["predicted_label"])): int(r["count"]) for r in rows}
    digits = range(10)
    if len(rows) != 100 or set(counts) != {(i, j) for i in digits for j in digits}:
        raise ValueError("Expected exactly one count for each true and predicted digit pair")
    matrix = np.array([[counts[i, j] for j in digits] for i in digits])
    correct = int(matrix.trace())
    if matrix.min() < 0 or matrix.sum() != size or correct != expected_correct:
        raise ValueError("Confusion counts disagree with sample size or accuracy")

    height, bottom, grid_height = 212, 28, 149
    fig = plt.figure(figsize=(WIDTH / 72, height / 72), facecolor="white")
    grid = fig.add_axes([28 / WIDTH, bottom / height, 218 / WIDTH, grid_height / height])
    bars = fig.add_axes([345 / WIDTH, bottom / height, 111 / WIDTH, grid_height / height])
    panel(fig, 1, 3, "a", "Confusion matrix")
    panel(fig, 262, 3, "b", "Errors by true digit")
    fig.text(246 / WIDTH, 1 - 5 / height, f"{size:,} images · {100 * correct / size:.2f}% correct",
             va="top", ha="right", fontsize=8)
    fig.text(137 / WIDTH, 1 - 20 / height, "Predicted digit", va="center", ha="center")
    fig.text(268 / WIDTH, 1 - 26 / height, "Digit", va="center", ha="center")
    fig.text(328 / WIDTH, 1 - 26 / height, "Errors / total", va="center", ha="right")
    fig.text(345 / WIDTH, 1 - 26 / height, "Error rate (%)", va="center", ha="left")
    grid.set(xlim=(-.5, 9.5), ylim=(9.5, -.5), xticks=digits, yticks=digits, ylabel="True digit")
    grid.xaxis.tick_top()
    grid.tick_params(axis="both", labelsize=8.2, pad=2, length=2, width=.5)
    grid.yaxis.labelpad = 4
    for spine in grid.spines.values():
        spine.set_color("#8D9CA8")
        spine.set_linewidth(.5)
    palette = LinearSegmentedColormap.from_list("ink_counts", ["#FFFFFF", "#DCE8EE", "#A8C2D0",
                                                                 "#648BA2", BLUE, INK])
    peak = max(1, int(matrix.max()))
    def fill(count):
        return palette(log1p(count) / log1p(peak))

    off_diagonal = sorted((c for (i, j), c in counts.items() if i != j and c > 0), reverse=True)
    boundary = off_diagonal[min(1, len(off_diagonal) - 1)] if off_diagonal else peak + 1
    for (i, j), count in np.ndenumerate(matrix):
        shade = fill(count)
        grid.add_patch(Rectangle((j - .5, i - .5), 1, 1, edgecolor="#C5D0D7", linewidth=.35,
                                 facecolor=shade, gid=f"cell-{i}-{j}"))
        if i != j and count >= boundary:
            grid.add_patch(Rectangle((j - .475, i - .465), .95, .93, facecolor="none",
                                     edgecolor=INK, linewidth=1.7))
            grid.add_patch(Rectangle((j - .475, i - .465), .95, .93, facecolor="none",
                                     edgecolor=GOLD, linewidth=.95, gid=f"top-error-{i}-{j}"))
        luminance = np.dot([c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4
                            for c in shade[:3]], [.2126, .7152, .0722])
        text_colour = "white" if luminance < .18 else ("#52606B" if count == 0 else TEXT)
        grid.text(j, i, str(count), ha="center", va="center", color=text_colour, fontsize=8.2,
                  weight="bold" if i == j or count and count >= boundary else "normal",
                  gid=f"count-{i}-{j}-{count}")
    examples = sorted({0, peak, *(v for v in (1, 5, 20, 100) if v < peak)})
    fig.text(1 / WIDTH, 9.5 / height, "Images per cell", va="center", fontsize=8)
    for x, value in zip(np.linspace(67, 217, len(examples)), examples):
        fig.add_artist(Rectangle((x / WIDTH, 6.5 / height), 6 / WIDTH, 6 / height, transform=fig.transFigure,
                                 facecolor=fill(value), edgecolor="#8996A1", linewidth=.4))
        fig.text((x + 8) / WIDTH, 9.5 / height, f"{value:,}", va="center", fontsize=7.7)

    totals = matrix.sum(axis=1)
    errors = totals - matrix.diagonal()
    rates = np.divide(100. * errors, totals, out=np.zeros(10), where=totals != 0)
    peak_rate = float(max(rates))
    scale = max(1., peak_rate)
    bars.set(xlim=(0, 1.32 * scale), ylim=(9.5, -.5))
    bottom_axis(bars)
    tick_values = MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]).tick_values(0, scale)
    bars.set_xticks([t for t in tick_values if t <= scale * 1.01])
    bars.spines["bottom"].set_bounds(0, scale)
    bars.vlines(0, -.5, 9.5, color="#9AA8AF", linewidth=.55)
    for i, (n, missed, rate) in enumerate(zip(totals, errors, rates)):
        y = (bottom + (9.5 - i) * grid_height / 10) / height
        maximum = bool(n and missed and rate == peak_rate)
        fig.text(268 / WIDTH, y, str(i), ha="center", va="center",
                 weight="bold" if maximum else "normal", fontsize=8)
        fig.text(328 / WIDTH, y, f"{missed:,} / {n:,}", ha="right", va="center",
                 weight="bold" if maximum else "normal", fontsize=8)
        bars.barh(i, rate, height=.43, color=VIOLET if maximum else "#86A6B8",
                  edgecolor=VIOLET if maximum else BLUE, linewidth=.45, gid=f"errors-{i}-{missed}")
        bars.annotate(f"{rate:.2f}%" if n else "n/a", (rate, i), xytext=(3, 0),
                      textcoords="offset points", va="center", ha="left", fontsize=8,
                      weight="bold" if maximum else "normal", clip_on=False)
    return fig


def plots(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, scale = read(folder, "summary.csv"), read(folder, "scaling.csv")
    sizes = sorted(int(r["n_test"]) for r in scale)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.edgecolor":"#727982","axes.labelcolor":"#262F35",
                         "axes.titleweight":"bold","legend.frameon":False,
                         "figure.facecolor":"white","savefig.facecolor":"white"})
    def save(fig,name):
        fig.savefig(out/(name+".png"),dpi=220,bbox_inches="tight",pad_inches=.12)
        fig.savefig(out/(name+".svg"),bbox_inches="tight",pad_inches=.12)
        plt.close(fig)
    prof = read(folder,"device_profile.csv")
    fig, (a,b) = plt.subplots(1,2,figsize=(9,3.8),gridspec_kw={"width_ratios":[1.08,1]})
    vals=[float(next(r["gpu2_speedup"] for r in scale if int(r["n_test"])==n)) for n in sizes]
    a.plot(range(len(sizes)),vals,"o-",color="#077477",lw=2.2,ms=7)
    a.axhline(1,color="#858D94",linestyle="--",lw=1); a.axhline(2,color="#858D94",linestyle=":",lw=1)
    for i,v in enumerate(vals):
        a.annotate(f"{v:.2f}×\n{v/2:.0%} eff.",(i,v),xytext=(0,8),textcoords="offset points",
                   ha="center",fontsize=9,color="#07585C")
    a.set(xticks=range(len(sizes)),xticklabels=[f"{s:,}" for s in sizes],xlabel="Test images per job",
          ylabel="Speedup over one H100",ylim=(0,max(2.24,max(vals)+.52)),xlim=(-.4,len(sizes)-.6))
    a.set_title("A   Adding the second GPU",loc="left",fontsize=12,pad=9)
    prows=sorted((r for r in prof if int(r["n_test"])==sizes[-1]),key=lambda r:(r["mode"],int(r["device"])))
    bottom=np.zeros(len(prows))
    for column,label,color in [("h2d_ms","Host to GPU","#58728D"),("compute_ms","KNN compute","#077477"),("d2h_ms","GPU to host","#CC8248")]:
        value=np.array([float(r[column]) for r in prows]); b.bar(range(len(prows)),value,bottom=bottom,color=color,label=label,width=.58); bottom+=value
    b.set(xticks=range(len(prows)),xticklabels=["1 GPU\ncard 0","2 GPUs\ncard 0","2 GPUs\ncard 1"],
          ylabel="Milliseconds on each device")
    b.set_title(f"B   Device profile at {sizes[-1]:,} images",loc="left",fontsize=12,pad=9)
    b.grid(axis="y",alpha=.2); b.set_axisbelow(True)
    b.legend(loc="upper center",bbox_to_anchor=(.50,-.19),ncol=3,fontsize=8,columnspacing=.8)
    fig.subplots_adjust(left=.09,right=.99,bottom=.24,top=.89,wspace=.38)
    save(fig,"scaling")

    size = sizes[-1]
    correct = next(int(r["correct"]) for r in rows if int(r["n_test"]) == size and r["mode"] == "cpu")
    with plt.rc_context(PRINT_STYLE):
        for name, figure in (("performance", performance_figure(folder, rows, sizes)),
                             ("confusion", confusion_figure(folder, size, correct))):
            for suffix in ("pdf", "svg", "png"):
                figure.savefig(out / f"{name}.{suffix}", dpi=300)
            plt.close(figure)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path); p.add_argument("--out",type=Path,default=Path("figures"))
    args=p.parse_args(); plots(args.results,args.out)
