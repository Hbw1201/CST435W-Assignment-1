"""Rebuild the coursework figures from the archived measurements and MNIST example."""

import argparse
import base64
import csv
import hashlib
import json
import zlib
from math import ceil, floor, log10
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, NullLocator

WIDTH = 162 * 72 / 25.4  # The report's 162 mm text column, in typographic points.
BLUE, SKY, VIOLET, LIME, CREAM = ("#6990B3", "#A1C9E9", "#895FC3", "#B3DC5B", "#EEE09B")
INK, MUTED, RULE = "#202B34", "#475662", "#D4DBDF"
PRINT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "cst435w-figures",
    "savefig.facecolor": "white",
}


def read(folder, name):
    with (folder / name).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sheet(height):
    """A typeset canvas in print points, with a top-left origin for diagram elements."""
    fig = plt.figure(figsize=(WIDTH / 72, height / 72), facecolor="white")
    art = fig.add_axes([0, 0, 1, 1], frameon=False)
    art.set(xlim=(0, WIDTH), ylim=(height, 0))
    art.set_axis_off()
    return fig, art


def heading(art, x, letter, title):
    art.text(x, 10, letter, va="center", fontsize=11, fontweight="bold")
    art.text(x + 13, 10, title, va="center", fontsize=9.5, fontweight="bold")


def tint(colour, fraction):
    return np.asarray(to_rgb(colour)) * fraction + np.ones(3) * (1 - fraction)


def vote_figure(example):
    info = json.loads(example.read_text(encoding="utf-8"))
    neighbours = sorted(info["neighbors"], key=lambda row: row["rank"])
    if len(neighbours) != 5 or [r["rank"] for r in neighbours] != list(range(1, 6)):
        raise ValueError("The voting example needs exactly five ranked neighbours")
    labels = [int(r["label"]) for r in neighbours]
    votes = np.bincount(labels, minlength=10)
    prediction = int(votes.argmax())
    if prediction != info["prediction"]:
        raise ValueError("Example prediction disagrees with the five recorded labels")
    height = 106
    fig, art = sheet(height)
    for x, letter, title in (
        (1, "a", "Test image"),
        (83, "b", "Five nearest training images"),
        (335, "c", "Vote count"),
    ):
        heading(art, x, letter, title)
    for x in (73, 325):
        art.plot([x, x], [25, 99], color=RULE, lw=0.55)

    def image_at(row, cx, cy, size):
        pixels = zlib.decompress(base64.b64decode(row["pixels_zlib"]))
        if len(pixels) != 784 or hashlib.sha256(pixels).hexdigest() != row["sha256"]:
            raise ValueError("MNIST example pixels failed their recorded checksum")
        ax = fig.add_axes(
            [(cx - size / 2) / WIDTH, 1 - (cy + size / 2) / height, size / WIDTH, size / height]
        )
        ax.imshow(
            np.frombuffer(pixels, dtype=np.uint8).reshape(28, 28),
            cmap="gray_r",
            vmin=0,
            vmax=255,
            interpolation="none",
        )
        ax.set_axis_off()

    image_at(info["query"], 33, 56, 48)
    art.text(33, 93, f"True label  {info['truth']}", ha="center", va="center", weight="bold")
    xs = np.linspace(102, 303, 5)
    for row, x in zip(neighbours, xs):
        art.text(x, 31, f"Rank {row['rank']}", ha="center", va="center", color=MUTED, fontsize=8)
        image_at(row, x, 59, 36)
        art.text(x, 92, f"Digit {row['label']}", ha="center", va="center", fontsize=8.5)
    classes = sorted(set(labels), key=lambda cls: (-votes[cls], cls))
    for i, cls in enumerate(classes):
        y = 37 + i * min(22, 41 / max(len(classes) - 1, 1))
        colour = VIOLET if cls == prediction else BLUE
        art.text(335, y, f"Digit {cls}", va="center")
        for k in range(votes[cls]):
            art.plot(
                375 + 11 * k,
                y,
                marker="o" if cls == prediction else "s",
                ms=6.5,
                mfc=colour,
                mec=INK,
                mew=0.45,
            )
        art.text(451, y, str(votes[cls]), ha="right", va="center", weight="bold")
    art.plot([335, 451], [79, 79], color=RULE, lw=0.55)
    art.text(335, 94, f"Predicted digit  {prediction}", va="center", weight="bold")
    return fig


def partition_figure(n_test, n_train):
    first = (n_test + 1) // 2
    spans = [(0, first, BLUE), (first, n_test, VIOLET)]
    fig, art = sheet(144)
    heading(art, 1, "a", "Query-by-reference comparisons")
    heading(art, 249, "b", "Reassembling the labels")
    left, right, top, bottom = 40, 225, 48, 127
    art.text(
        (left + right) / 2, 31, f"All {n_train:,} training images", va="center", ha="center", weight="bold"
    )
    art.plot([left, left, right, right], [43, 39, 39, 43], color=MUTED, lw=0.65)
    start, end = 251, WIDTH - 3
    for gpu, (lo, hi, colour) in enumerate(spans):
        y0 = top + (bottom - top) * lo / n_test
        y1 = top + (bottom - top) * hi / n_test
        art.add_patch(
            Rectangle((left, y0), right - left, y1 - y0, fc=tint(colour, 0.24), ec="white", lw=0.8)
        )
        art.add_patch(Rectangle((left, y0), 3, y1 - y0, fc=colour, ec="none"))
        yc = (y0 + y1) / 2
        art.text(1, yc, f"GPU {gpu}", va="center", fontsize=8.5, weight="bold")
        art.text(left + 10, yc - 7, f"Test IDs  {lo:,}–{hi - 1:,}", va="center", weight="bold")
        art.text(
            left + 10, yc + 7, f"{hi - lo:,} queries × {n_train:,} references", va="center", fontsize=8
        )
        x0 = start + (end - start) * lo / n_test
        x1 = start + (end - start) * hi / n_test
        xc = (x0 + x1) / 2
        art.text(xc, 34, f"GPU {gpu}", va="center", ha="center", weight="bold")
        art.text(xc, 47, f"{hi - lo:,} labels", va="center", ha="center", color=MUTED)
        art.annotate("", (xc, 72), (xc, 54), arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.3})
        art.add_patch(Rectangle((x0, 75), x1 - x0, 24, fc=tint(colour, 0.28), ec="white", lw=0.8))
        art.text(xc, 87, f"{lo:,}–{hi - 1:,}", va="center", ha="center", weight="bold")
    art.plot([start, start, end, end], [104, 108, 108, 104], color=MUTED, lw=0.65)
    art.text(
        (start + end) / 2,
        123,
        f"{n_test:,} predictions in test order",
        va="center",
        ha="center",
        weight="bold",
    )
    return fig


def performance_figure(folder, rows, sizes):
    summary = {(int(r["n_test"]), r["mode"]): r for r in rows}
    runs = {}
    if (folder / "runs.csv").is_file():
        for r in read(folder, "runs.csv"):
            trial = int(r["run"])
            group = runs.setdefault((int(r["n_test"]), r["mode"]), {})
            if trial in group:
                raise ValueError("Duplicate measured repeat")
            group[trial] = 1000 * float(r["wall_s"])
    styles = {
        "cpu": ("1 CPU thread", "s", CREAM, MUTED, "-."),
        "gpu1": ("1 H100", "o", BLUE, BLUE, "-"),
        "gpu2": ("2 H100s", "D", VIOLET, VIOLET, "--"),
    }
    height = 194
    fig, art = sheet(height)
    heading(art, 1, "a", "Request time (ms, log scale)")
    heading(art, 253, "b", "Scaling from one to two H100s")
    for x, (label, marker, face, line, _) in zip((34, 111, 164), styles.values()):
        art.plot(x, 28, marker=marker, mfc=face, mec=INK, mew=0.55, ms=5, linestyle="none")
        art.text(x + 6, 28, label, va="center", fontsize=8)
    ax = fig.add_axes([34 / WIDTH, (height - 155) / height, 200 / WIDTH, 113 / height])
    ax.set_yscale("log")
    xpos = np.arange(len(sizes), dtype=float)
    minimum = min(float(r["min_wall_s"]) * 1000 for r in rows)
    maximum = max(float(r["max_wall_s"]) * 1000 for r in rows)
    low, high = minimum / 1.65, maximum * 1.8
    ax.set(ylim=(low, high), xlim=(-0.18, len(sizes) - 1 + 0.72))
    tks = [10.0**j for j in range(floor(log10(low)), ceil(log10(high)) + 1) if low <= 10.0**j <= high]
    if not tks:
        tks = [minimum]
    ax.set_yticks(tks, [f"{v:,.0f}" if v >= 1 else f"{v:g}" for v in tks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks(xpos, [f"{v:,}" for v in sizes])
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set(color=MUTED, linewidth=0.6)
    ax.tick_params(length=2.8, width=0.6, labelsize=8, pad=3)
    ax.grid(axis="y", color=RULE, lw=0.5)
    ax.set_axisbelow(True)
    for offset, (mode, (_, marker, face, line, ls)) in zip((-0.04, 0.0, 0.04), styles.items()):
        medians, spread = [], []
        for n in sizes:
            row = summary[n, mode]
            mn, md, mx = [1000 * float(row[k]) for k in ("min_wall_s", "median_wall_s", "max_wall_s")]
            if runs:
                values = list(runs[n, mode].values())
                if len(values) != int(row["repeats"]) or not np.allclose(
                    (mn, md, mx), (min(values), median(values), max(values))
                ):
                    raise ValueError(f"Raw timings disagree with {mode}/{n}")
            medians.append(md)
            spread.append([md - mn, mx - md])
        ax.errorbar(
            xpos + offset,
            medians,
            yerr=np.transpose(spread),
            color=line,
            linestyle=ls,
            marker=marker,
            mfc=face,
            mec=INK,
            mew=0.6,
            ms=5,
            lw=1.25,
            elinewidth=0.8,
            capsize=2.5,
        )
        med = medians[-1]
        label = f"{med / 1000:.2f} s" if med >= 1000 else f"{med:.1f} ms"
        ax.annotate(
            label,
            (xpos[-1] + offset, med),
            (5, {"cpu": 0, "gpu1": 4, "gpu2": -6}[mode]),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.1,
            bbox={"facecolor": "white", "ec": "none", "pad": 0.2},
        )
    art.text(129, 185, "Test images per request", ha="center", va="center")
    x0, x1 = 293, 404
    ratios = [
        float(summary[n, "gpu1"]["median_wall_s"]) / float(summary[n, "gpu2"]["median_wall_s"])
        for n in sizes
    ]
    scale_lo, scale_hi = min(1, floor(min(ratios) * 2) / 2), max(2, ceil(max(ratios) * 2) / 2)

    def project(speedup):
        return x0 + (speedup - scale_lo) / (scale_hi - scale_lo) * (x1 - x0)

    art.text(253, 29, "Test images", va="center", color=MUTED, fontsize=8)
    art.text(WIDTH - 1, 29, "Efficiency", va="center", ha="right", color=MUTED, fontsize=8)
    for ref in (1, 2):
        art.plot(
            [project(ref)] * 2,
            [38, 155],
            ls="--" if ref == 2 else "-",
            color=MUTED if ref == 1 else "#97A3AA",
            lw=0.6,
        )
    ys = np.linspace(52, 137, len(sizes)) if len(sizes) > 1 else [94]
    for n, y, sp in zip(sizes, ys, ratios):
        art.text(253, y, f"{n:,}", va="center", fontsize=8.5)
        art.plot([x0, x1], [y, y], color=RULE, lw=0.6)
        art.plot([project(1), project(sp)], [y, y], color=VIOLET, lw=2.1, solid_capstyle="butt")
        art.plot(project(sp), y, marker="D", mfc=VIOLET, mec=INK, mew=0.55, ms=5)
        art.text(project(sp) + 5, y - 6, f"{sp:.2f}×", ha="left", va="bottom", fontsize=8)
        art.text(WIDTH - 1, y, f"{50 * sp:.1f}%", ha="right", va="center", weight="bold")
    art.plot([x0, x1], [155, 155], color=MUTED, lw=0.6)
    for ref in (scale_lo, (scale_lo + scale_hi) / 2, scale_hi):
        x = project(ref)
        art.plot([x, x], [155, 158], color=MUTED, lw=0.6)
        tick = {1: "1× (parity)", 2: "2× (ideal)"}.get(ref, f"{ref:g}×")
        art.text(x, 167, tick, ha="center", va="center", fontsize=8)
    art.text((x0 + x1) / 2, 185, "Speedup over one H100", ha="center", va="center")
    return fig


def confusion_figure(folder, size, expected_correct):
    rows = [r for r in read(folder, "confusion.csv") if int(r["n_test"]) == size]
    counts = {(int(r["true_label"]), int(r["predicted_label"])): int(r["count"]) for r in rows}
    if len(rows) != 100 or set(counts) != {(i, j) for i in range(10) for j in range(10)}:
        raise ValueError("Expected one count for every true/predicted digit pair")
    mat = np.array([[counts[i, j] for j in range(10)] for i in range(10)])
    if mat.min() < 0 or mat.sum() != size or mat.trace() != expected_correct:
        raise ValueError("Confusion counts disagree with accuracy")
    nrows = mat.sum(axis=1)
    errors = nrows - mat.diagonal()
    rates = np.divide(100.0 * errors, nrows, out=np.zeros(10), where=nrows != 0)
    off = sorted((v for (i, j), v in counts.items() if i != j and v), reverse=True)
    vmax = max(off, default=1)
    threshold = off[min(1, len(off) - 1)] if off else vmax + 1
    selected = int(np.argmax(rates))
    cmap = LinearSegmentedColormap.from_list("misclassified", ["white", tint(SKY, 0.70), BLUE])
    fig, art = sheet(243)
    heading(art, 1, "a", "Confusion matrix")
    heading(art, 245, "b", "Errors by true digit")
    x0, y0, cell, cell_width = 32, 43, 16.9, 19.5
    art.text(x0 + 5 * cell_width, 24, "Predicted digit", ha="center", va="center", fontsize=8)
    art.text(6, y0 + 5 * cell, "True digit", ha="center", va="center", rotation=90, fontsize=8)
    for i in range(10):
        art.text(x0 + (i + 0.5) * cell_width, y0 - 7, str(i), ha="center", va="center", fontsize=8)
        art.text(x0 - 9, y0 + (i + 0.5) * cell, str(i), ha="center", va="center", fontsize=8)
    for (i, j), v in np.ndenumerate(mat):
        x, y = x0 + j * cell_width, y0 + i * cell
        face = INK if i == j else cmap(v / vmax)
        art.add_patch(Rectangle((x, y), cell_width, cell, fc=face, ec=RULE, lw=0.35, gid=f"cell-{i}-{j}"))
        if i != j and v >= threshold:
            art.add_patch(
                Rectangle((x + 0.6, y + 0.6), cell_width - 1.2, cell - 1.2, fc="none", ec=INK, lw=1.15)
            )
            art.add_patch(
                Rectangle((x + 1.1, y + 1.1), cell_width - 2.2, cell - 2.2, fc="none", ec=LIME, lw=0.8)
            )
        if v or i == j:
            art.text(
                x + cell_width / 2,
                y + cell / 2,
                str(v),
                ha="center",
                va="center",
                color="white" if i == j else INK,
                fontsize=8.0,
                weight="bold" if i == j or v >= threshold else "normal",
                gid=f"count-{i}-{j}-{v}",
            )
    art.add_patch(Rectangle((12, 228), 6, 6, fc=INK, ec="none"))
    art.text(21, 231, "Correct", va="center", fontsize=8)
    art.text(64, 231, "Errors", va="center", fontsize=8)
    for x, value in zip((88, 122, 163), (0, round(vmax / 2), vmax)):
        art.add_patch(Rectangle((x, 228), 7, 6, fc=cmap(value / vmax), ec=MUTED, lw=0.45))
        art.text(x + 10, 231, str(value), va="center", fontsize=8)
    a, b = 279, 409
    global_rate = 100.0 * (size - expected_correct) / size
    bound = max(1.0, ceil(max(rates)))

    def rate_x(rate):
        return a + rate / bound * (b - a)

    art.text(245, 29, "Digit", va="center", fontsize=8, color=MUTED)
    art.plot([rate_x(global_rate)] * 2, [y0, y0 + 10 * cell], ls=(0, (3, 2)), lw=0.8, color=MUTED)
    art.text(rate_x(global_rate), 32, f"Overall  {global_rate:.2f}%", ha="center", va="center", fontsize=8)
    for i, rate in enumerate(rates):
        y = y0 + (i + 0.5) * cell
        art.text(251, y, str(i), ha="center", va="center", weight="bold" if i == selected else "normal")
        art.plot([a, b], [y, y], color="#E6EBED", lw=0.45, zorder=0)
        art.plot([a, rate_x(rate)], [y, y], color=VIOLET if i == selected else BLUE, lw=1.65)
        art.plot(
            rate_x(rate),
            y,
            marker="D" if i == selected else "o",
            ms=5.3 if i == selected else 4.2,
            mfc=VIOLET if i == selected else SKY,
            mec=INK,
            mew=0.45,
        )
        art.text(
            rate_x(rate) + 5,
            y,
            f"{rate:.2f}%" if nrows[i] else "n/a",
            va="center",
            fontsize=8.1,
            weight="bold" if i == selected else "normal",
        )
    yaxis = y0 + 10 * cell + 2
    art.plot([a, b], [yaxis, yaxis], color=MUTED, lw=0.55)
    ticks = MaxNLocator(nbins=4, integer=True).tick_values(0, bound)
    for v in (t for t in ticks if t <= bound):
        x = rate_x(v)
        art.plot([x, x], [yaxis, yaxis + 3], color=MUTED, lw=0.55)
        art.text(x, yaxis + 10, f"{v:g}", ha="center", va="center", fontsize=8)
    art.text((a + b) / 2, 237, "Error rate (%)", ha="center", va="center", fontsize=8.5)
    return fig


def batch_figure(initial_dir, repeat_dir):
    """Plot the unprofiled timings. The independent traces have their own summaries."""
    sessions = []
    for folder in (initial_dir, repeat_dir):
        groups = {}
        for row in read(folder / "timed", "runs.csv"):
            if row["placement"] != "resident" or int(row["gpu_count"]) != 2:
                raise ValueError("The batch figure requires the two-GPU resident measurements")
            values = groups.setdefault(int(row["batch"]), {})
            repeat = int(row["repeat"])
            if repeat in values:
                raise ValueError("Duplicate timing for a batch and round")
            values[repeat] = float(row["wall_ms"])
        sessions.append(groups)
    batches = sorted(sessions[0])
    if not batches or set(batches) != set(sessions[1]):
        raise ValueError("The two sessions must use the same batch sizes")

    height = 186
    fig, art = sheet(height)
    for i, (groups, title, colour, x) in enumerate(
        zip(sessions, ("Initial session", "Repeat session"), (BLUE, VIOLET), (35, 273))
    ):
        counts = {len(values) for values in groups.values()}
        if len(counts) != 1 or min(counts) < 2:
            raise ValueError("Each size must have the same number of repeats within a session")
        count = counts.pop()
        heading(art, 1 if i == 0 else 238, "ab"[i], f"{title} ({count} timings per size)")
        ax = fig.add_axes([x / WIDTH, (height - 136) / height, 175 / WIDTH, 111 / height])
        all_times = [value for values in groups.values() for value in values.values()]
        lo, hi = min(all_times), max(all_times)
        pad = max(0.09 * (hi - lo), 0.05)
        ax.set(xlim=(lo - pad, hi + pad), ylim=(len(batches) - 0.47, -0.56))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.set_yticks(np.arange(len(batches)), [f"{b:,}" for b in batches])
        ax.tick_params(axis="y", length=0, pad=4, labelsize=8)
        ax.tick_params(axis="x", length=2.5, pad=2, labelsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set(color=MUTED, linewidth=0.55)
        ax.grid(axis="x", color=RULE, linewidth=0.4)
        ax.set_axisbelow(True)
        for j, batch in enumerate(batches):
            values = np.array([v for _, v in sorted(groups[batch].items())])
            q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            iqr = q3 - q1
            inside = values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]
            wl, wr = min(inside), max(inside)
            if batch == 2048:
                ax.axhspan(j - 0.47, j + 0.47, color=tint(LIME, 0.16), lw=0, zorder=-1)
                ax.get_yticklabels()[j].set_weight("bold")
            y = j - 0.08
            ax.hlines(y, wl, wr, color=MUTED, lw=0.65)
            ax.vlines([wl, wr], y - 0.10, y + 0.10, color=MUTED, lw=0.65)
            ax.add_patch(Rectangle((q1, y - 0.17), q3 - q1, 0.34, fc=tint(colour, 0.32), ec=MUTED, lw=0.7))
            ax.vlines(med, y - 0.17, y + 0.17, color=INK, lw=1.0)
            ax.plot(
                values.mean(), y, marker="D", ms=3.7, mfc=colour, mec=INK, mew=0.5, gid=f"mean-{i}-{batch}"
            )
            # Deterministic vertical offsets keep every measured time on its exact x-coordinate.
            offsets = 0.075 * np.sin(np.arange(count) * 2.4)
            ax.scatter(
                values,
                j + 0.25 + offsets,
                s=5,
                facecolors=colour,
                edgecolors=INK,
                linewidths=0.22,
                alpha=0.84,
                gid=f"raw-{i}-{batch}",
            )
    art.text(1, 23, "Batch / GPU", va="center", fontsize=8, color=MUTED)
    art.text(238, 23, "Batch / GPU", va="center", fontsize=8, color=MUTED)
    art.text(124, 162, "Request time (ms)", ha="center", va="center", fontsize=8.5)
    art.text(361, 162, "Request time (ms)", ha="center", va="center", fontsize=8.5)
    y = 179
    art.plot(42, y, marker="o", ms=3, mfc=SKY, mec=INK, mew=0.4)
    art.text(48, y, "One timing", va="center", fontsize=8)
    art.plot(115, y, marker="D", ms=4, mfc=VIOLET, mec=INK, mew=0.5)
    art.text(122, y, "Mean", va="center", fontsize=8)
    art.add_patch(Rectangle((172, y - 3), 14, 6, fc=tint(BLUE, 0.32), ec=MUTED, lw=0.65))
    art.vlines(180, y - 3, y + 3, color=INK, lw=0.9)
    art.text(191, y, "Middle 50% and median", va="center", fontsize=8)
    art.plot([315, 333], [y, y], color=MUTED, lw=0.65)
    art.vlines([315, 333], y - 2.6, y + 2.6, color=MUTED, lw=0.65)
    art.text(339, y, "Whiskers", va="center", fontsize=8)
    return fig


def plots(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, scale = read(folder, "summary.csv"), read(folder, "scaling.csv")
    sizes = sorted(int(r["n_test"]) for r in scale)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#727982",
            "axes.labelcolor": "#262F35",
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    def save(fig, name):
        fig.savefig(out / (name + ".png"), dpi=220, bbox_inches="tight", pad_inches=0.12)
        fig.savefig(out / (name + ".svg"), bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)

    prof = read(folder, "device_profile.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1.08, 1]})
    vals = [float(next(r["gpu2_speedup"] for r in scale if int(r["n_test"]) == n)) for n in sizes]
    a.plot(range(len(sizes)), vals, "o-", color=VIOLET, lw=2.2, ms=7)
    a.axhline(1, color="#858D94", linestyle="--", lw=1)
    a.axhline(2, color="#858D94", linestyle=":", lw=1)
    for i, v in enumerate(vals):
        a.annotate(
            f"{v:.2f}×\n{v / 2:.0%} eff.",
            (i, v),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
    a.set(
        xticks=range(len(sizes)),
        xticklabels=[f"{s:,}" for s in sizes],
        xlabel="Test images per job",
        ylabel="Speedup over one H100",
        ylim=(0, max(2.24, max(vals) + 0.52)),
        xlim=(-0.4, len(sizes) - 0.6),
    )
    a.set_title("A   Adding the second GPU", loc="left", fontsize=12, pad=9)
    prows = sorted(
        (r for r in prof if int(r["n_test"]) == sizes[-1]), key=lambda r: (r["mode"], int(r["device"]))
    )
    bottom = np.zeros(len(prows))
    for column, label, color in [
        ("h2d_ms", "Host to GPU", BLUE),
        ("compute_ms", "KNN compute", VIOLET),
        ("d2h_ms", "GPU to host", CREAM),
    ]:
        value = np.array([float(r[column]) for r in prows])
        b.bar(range(len(prows)), value, bottom=bottom, color=color, label=label, width=0.58)
        bottom += value
    b.set(
        xticks=range(len(prows)),
        xticklabels=["1 GPU\ncard 0", "2 GPUs\ncard 0", "2 GPUs\ncard 1"],
        ylabel="Milliseconds on each device",
    )
    b.set_title(f"B   Device profile at {sizes[-1]:,} images", loc="left", fontsize=12, pad=9)
    b.grid(axis="y", alpha=0.2)
    b.set_axisbelow(True)
    b.legend(loc="upper center", bbox_to_anchor=(0.50, -0.19), ncol=3, fontsize=8, columnspacing=0.8)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.24, top=0.89, wspace=0.38)
    save(fig, "scaling")

    size = sizes[-1]
    correct = next(int(r["correct"]) for r in rows if int(r["n_test"]) == size and r["mode"] == "cpu")
    with plt.rc_context(PRINT_STYLE):
        figures = {
            "performance": performance_figure(folder, rows, sizes),
            "confusion": confusion_figure(folder, size, correct),
        }
        if size >= 2:
            n_train = int(next(r["n_train"] for r in rows if int(r["n_test"]) == size))
            figures["figure-2-query-partition"] = partition_figure(size, n_train)
        example = folder.parent / "figure-1-example.json"
        if example.is_file():
            figures["figure-1-neighbour-vote"] = vote_figure(example)
        sessions = (folder.parent / "batch_sweep", folder.parent / "batch_repeat5")
        if all((session / "timed" / "runs.csv").is_file() for session in sessions):
            figures["batch-distributions"] = batch_figure(*sessions)
        for name, figure in figures.items():
            figure.savefig(out / f"{name}.pdf", dpi=300, metadata={"CreationDate": None, "ModDate": None})
            figure.savefig(out / f"{name}.svg", dpi=300, metadata={"Date": None})
            figure.savefig(out / f"{name}.png", dpi=300)
            plt.close(figure)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("--out", type=Path, default=Path("figures"))
    args = p.parse_args()
    plots(args.results, args.out)
