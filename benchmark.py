"""Run the nine MNIST configurations sequentially in fresh processes."""
import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns, process_time_ns

import torch
import torchvision
from classify import BATCH, MODES, ROOT, classify, configure, load_inputs
from knn import K, check_voting, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(mode, count, repeats, path):
    configure(mode)
    check_voting()
    x, y, q, truth = load_inputs(count)
    devices = range(MODES[mode])
    config = dict(n_test=count, mode=mode, gpu_count=len(devices), n_train=len(x),
                  k=K, batch_size=BATCH, dtype="float64", intraop_threads=torch.get_num_threads())
    runs = []
    with ThreadPoolExecutor(max_workers=max(1, len(devices))) as pool:
        start = perf_counter_ns()
        expected, _ = classify(mode, x, y, q, pool)
        warmup = (perf_counter_ns() - start) / 1e9
        correct = int((expected == truth).sum())
        for repeat in range(1, repeats + 1):
            for dev in devices:
                torch.cuda.reset_peak_memory_stats(dev)
            cpu_start = process_time_ns()
            start = perf_counter_ns()
            pred, _ = classify(mode, x, y, q, pool)
            wall_s = (perf_counter_ns() - start) / 1e9
            cpu_s = (process_time_ns() - cpu_start) / 1e9
            if not torch.equal(pred, expected):
                raise ValueError(f"Unstable predictions for {mode} / {count}")
            peak = [torch.cuda.max_memory_allocated(d) / 2**20 for d in devices]
            runs.append(dict(config, run=repeat, wall_s=wall_s, host_cpu_s=cpu_s,
                             throughput_images_s=count/wall_s, amortized_ms_per_image=1000*wall_s/count,
                             host_cpu_pct_one_core=100*cpu_s/wall_s, correct=correct,
                             accuracy_pct=100*correct/count, gpu0_peak_alloc_mib=peak[0] if peak else None,
                             gpu1_peak_alloc_mib=peak[1] if len(peak)>1 else None))
            print(f"{mode:4s} n={count:5d} run={repeat} wall={wall_s:.6f}s", flush=True)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        trace = []
        if devices:
            profiled, trace = classify(mode, x, y, q, pool, profile=True)
            if not torch.equal(profiled, expected) or sum(t["shard_n"] for t in trace) != count:
                raise ValueError("GPU profile/shard mismatch")
    sec = [row["wall_s"] for row in runs]
    median = statistics.median(sec)
    summary = dict(config, repeats=repeats, median_wall_s=median,
                   mean_wall_s=statistics.mean(sec), sample_std_wall_s=statistics.stdev(sec),
                   min_wall_s=min(sec), max_wall_s=max(sec), throughput_images_s=count/median,
                   amortized_ms_per_image=1000*median/count, process_peak_rss_mib=rss,
                   correct=correct, errors=count-correct, accuracy_pct=100*correct/count)
    data = dict(summary=summary, runs=runs, warmup_s=warmup, profile=trace,
                truth=truth.tolist(), pred=expected.tolist(),
                affinity=sorted(os.sched_getaffinity(0)), interop=torch.get_num_interop_threads())
    path.write_text(json.dumps(data), encoding="utf-8")


def export(cells, folder, sizes, repeats):
    summaries, runs, scaling, predictions, confusion, traces = [], [], [], [], [], []
    for n in sizes:
        group = {mode: cells[n, mode] for mode in MODES}
        cpu = group["cpu"]
        truth, ref = cpu["truth"], cpu["pred"]
        for mode, cell in group.items():
            if cell["truth"] != truth or len(cell["pred"]) != n:
                raise ValueError(f"Input ordering differs in {mode}/{n}")
            errors = sum(a != b for a, b in zip(ref, cell["pred"]))
            if errors:
                raise ValueError(f"{errors} predictions differ from CPU in {mode}/{n}")
            summaries.append(dict(cell["summary"], mismatches_vs_cpu=errors))
            runs.extend(cell["runs"])
            traces.extend(dict(n_test=n, mode=mode, **row) for row in cell["profile"])
        predictions.extend(dict(n_test=n, test_index=i, true_label=truth[i], cpu=ref[i],
                                gpu1=group["gpu1"]["pred"][i], gpu2=group["gpu2"]["pred"][i])
                           for i in range(n))
        counts = Counter(zip(truth, ref))
        confusion.extend(dict(n_test=n, true_label=i, predicted_label=j, count=counts[i,j])
                         for i in range(10) for j in range(10))
        tc, t1, t2 = [group[m]["summary"]["median_wall_s"] for m in MODES]
        scaling.append(dict(n_test=n, cpu_wall_s=tc, gpu1_wall_s=t1, gpu2_wall_s=t2,
                            cpu_to_gpu1=tc/t1, cpu_to_gpu2=tc/t2, gpu2_speedup=t1/t2,
                            gpu2_efficiency_pct=100*t1/(2*t2)))
    if len(runs) != len(sizes)*len(MODES)*repeats:
        raise ValueError("Incomplete experiment matrix")
    for name, rows in [("summary", summaries), ("runs", runs), ("scaling", scaling),
                       ("predictions", predictions), ("confusion", confusion), ("device_profile", traces)]:
        write_csv(folder/(name+".csv"), rows)
    # The earlier, separately measured CPU sample is a correctness cross-check.
    old = ROOT/"results/cpu_100.csv"
    matched_old = None
    if 100 in sizes and old.is_file():
        with old.open(newline="") as f:
            historical = [(int(r["test_index"]), int(r["true_label"]), int(r["predicted_label"])) for r in csv.DictReader(f)]
        actual = [(r["test_index"],r["true_label"],r["cpu"]) for r in predictions if r["n_test"]==100]
        matched_old = historical == actual
        if not matched_old:
            raise ValueError("The historical 100-query predictions disagree")
    return matched_old


def run_matrix(out, sizes, repeats):
    if out.exists():
        raise FileExistsError(f"Choose a fresh output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="."+out.name+"-", dir=out.parent))
    cell_dir = stage/"cells"
    cell_dir.mkdir()
    inputs = ROOT/"data/MNIST/raw"
    archive_md5 = {name: hashlib.md5((inputs/name).read_bytes()).hexdigest()
                   for name, _ in torchvision.datasets.MNIST.resources}
    if archive_md5 != dict(torchvision.datasets.MNIST.resources):
        raise ValueError("MNIST archives do not match torchvision's published checksums")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    gpu_info = subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.total,driver_version",
                                        "--format=csv"], text=True).strip()
    for n in sizes:
        for mode in MODES:
            target = cell_dir/f"{mode}_{n}.json"
            cmd = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--cell", mode,
                   str(n), str(target), "--repeats", str(repeats)]
            if mode == "cpu":
                cmd = ["taskset","-c",str(min(os.sched_getaffinity(0)))] + cmd
            subprocess.run(cmd, check=True, env=env)
    cells = {(n,m):json.loads((cell_dir/f"{m}_{n}.json").read_text()) for n in sizes for m in MODES}
    matched_old = export(cells, stage, sizes, repeats)
    meta = dict(completed_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                torch=str(torch.__version__), torchvision=str(torchvision.__version__), cuda=torch.version.cuda,
                gpu_inventory=gpu_info, historical_cpu100_match=matched_old, archive_md5=archive_md5,
                sizes=sizes, order=list(MODES), warmups_per_cell=1, measured_per_cell=repeats,
                comparison="same-prefix, end-to-end wall time; host-ready FP64 to ordered host predictions",
                cpu_info=(ROOT/"results/cpu_info.txt").read_text() if (ROOT/"results/cpu_info.txt").is_file() else None,
                sources={p:digest(ROOT/p) for p in ("knn.py","classify.py","benchmark.py")},
                csv_sha256={p.name:digest(p) for p in sorted(stage.glob("*.csv"))})
    (stage/"environment.json").write_text(json.dumps(meta,indent=2)+"\n", encoding="utf-8")
    stage.rename(out)
    print(f"Verified {len(sizes)*3*repeats} timed runs; saved {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/full_run")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100,1000,10000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cell", nargs=3, metavar=("MODE","N","OUTPUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats<2 or len(set(args.sizes))!=len(args.sizes) or not all(1<=n<=10000 for n in args.sizes):
        parser.error("Use distinct sizes in 1..10000 and at least two repeats.")
    if args.cell:
        mode, count, target = args.cell
        measure(mode,int(count),args.repeats,Path(target))
    else:
        run_matrix(args.out.resolve(),args.sizes,args.repeats)
