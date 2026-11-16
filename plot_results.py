"""Rebuild the report figures from the measurement CSV files."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator


def read(folder, name):
    with (folder/name).open(newline="") as f:
        return list(csv.DictReader(f))


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

    fig, (grid, bars) = plt.subplots(1, 2, figsize=(8.4, 4.5), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=.07, right=.975, top=.77, bottom=.14, wspace=.40)
    fig.text(.035, .97, f"MNIST 5-NN on {size:,} test images", va="top", fontsize=14, weight="bold")
    fig.text(.035, .895, f"{correct:,} correct   ·   {size-correct:,} errors   ·   {100*correct/size:.2f}% accuracy",
             va="top", fontsize=10)

    peak = int(matrix.max())
    im = grid.imshow(matrix, cmap="Blues", vmin=0, vmax=peak, interpolation="nearest")
    grid.set(xticks=digits, yticks=digits, xlabel="Predicted digit", ylabel="True digit")
    grid.set_title("Full confusion matrix", loc="left", fontsize=11, pad=8)
    grid.set_xticks(np.arange(10) - .5, minor=True)
    grid.set_yticks(np.arange(10) - .5, minor=True)
    grid.grid(which="minor", color="#cdd9e3", linewidth=.45)
    grid.tick_params(which="both", length=0)
    for (i, j), count in np.ndenumerate(matrix):
        color = "white" if im.norm(count) > .62 else ("#57636d" if count == 0 else "#172d40")
        grid.text(j, i, str(count), ha="center", va="center", fontsize=8 if count >= 1000 else 8.7, color=color)
    ticks = [v for v in MaxNLocator(nbins=3, integer=True).tick_values(0, peak) if 0 <= v < .86 * peak]
    colorbar = fig.colorbar(im, ax=grid, ticks=[*ticks, peak], fraction=.04, pad=.04, shrink=.80,
                            format=FuncFormatter(lambda value, _: f"{value:,.0f}"))
    colorbar.ax.set_title("Images", loc="left", fontsize=9, fontweight="normal", pad=7)

    errors = [(i, j, count) for (i, j), count in counts.items() if i != j and count > 0]
    errors.sort(key=lambda item: (-item[2], item[0], item[1]))
    errors = errors[:5]
    bars.set_title("Top errors (true → predicted)", loc="left", fontsize=11, pad=8)
    if errors:
        values = [count for _, _, count in errors]
        rects = bars.barh(range(len(errors)), values, height=.52, color="#23746f")
        bars.bar_label(rects, padding=4, fontsize=10)
        bars.set(yticks=range(len(errors)), yticklabels=[f"{i} → {j}" for i, j, _ in errors],
                 xlabel="Misclassified images", xlim=(0, max(values) * 1.19))
        bars.invert_yaxis()
        bars.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
        bars.set_axisbelow(True)
        bars.grid(axis="x", color="#dbe1e5", linewidth=.6)
    else:
        bars.set_axis_off()
        bars.text(.5, .5, "All predictions correct", transform=bars.transAxes, ha="center")
    return fig


def plots(folder, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, scale = read(folder, "summary.csv"), read(folder, "scaling.csv")
    sizes = sorted(int(r["n_test"]) for r in scale)
    modes = [("cpu","CPU, 1 thread","#525B66","o"),("gpu1","1 H100","#077477","s"),
             ("gpu2","2 H100s","#C06426","^")]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.edgecolor":"#727982","axes.labelcolor":"#262F35",
                         "axes.titleweight":"bold","legend.frameon":False,
                         "figure.facecolor":"white","savefig.facecolor":"white"})
    def save(fig,name):
        fig.savefig(out/(name+".png"),dpi=220,bbox_inches="tight",pad_inches=.12)
        fig.savefig(out/(name+".svg"),bbox_inches="tight",pad_inches=.12)
        plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,3.9))
    for mode, label, color, marker in modes:
        data = sorted((r for r in rows if r["mode"]==mode),key=lambda r:int(r["n_test"]))
        times = np.array([float(r["median_wall_s"]) for r in data])
        bottom = np.maximum(0,times-np.array([float(r["min_wall_s"]) for r in data]))
        top = np.maximum(0,np.array([float(r["max_wall_s"]) for r in data])-times)
        axes[0].errorbar(sizes,times,yerr=[bottom,top],label=label,color=color,marker=marker,
                         linewidth=1.9,capsize=4,markersize=6)
        axes[1].plot(sizes,np.array(sizes)/times,label=label,color=color,marker=marker,
                     linewidth=1.9,markersize=6)
    for a,title,unit in zip(axes,["A   Elapsed time","B   Throughput"],["Seconds per job","Images per second"]):
        a.set(xscale="log",yscale="log",xlabel="Test images per job",ylabel=unit)
        a.grid(axis="y",alpha=.23); a.set_axisbelow(True); a.set_xticks(sizes,[f"{n:,}" for n in sizes])
        a.xaxis.set_minor_locator(NullLocator()); a.set_title(title,loc="left",fontsize=12,pad=9)
    fig.legend(*axes[1].get_legend_handles_labels(),loc="lower center",ncol=3,bbox_to_anchor=(.5,-.01))
    fig.subplots_adjust(left=.09,right=.98,top=.90,bottom=.24,wspace=.33)
    save(fig,"performance")

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
    save(confusion_figure(folder, size, correct), "confusion")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("results",type=Path); p.add_argument("--out",type=Path,default=Path("figures"))
    args=p.parse_args(); plots(args.results,args.out)
