"""Interleave CPU and GPU requests in separately initialised benchmark sessions."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pynvml as nv
import torch
import torchvision

from benchmark import digest, export
from classify import BATCH, MODES, classify, configure, load_inputs
from knn import K, ROOT, check_voting, write_csv


def schedule(sizes, sessions, repeats, seed):
    rng = random.Random(seed)
    size_order = list(sizes)
    rng.shuffle(size_order)
    mode_orders = {}
    for size in sizes:
        mode_orders[size] = list(MODES)
        rng.shuffle(mode_orders[size])
    jobs = []
    for session in range(1, sessions + 1):
        for repeat in range(1, repeats + 1):
            round_index = (session - 1) * repeats + repeat - 1
            shift = round_index % len(size_order)
            ordered_sizes = size_order[shift:] + size_order[:shift]
            for size_position, size in enumerate(ordered_sizes, 1):
                block, mode_shift = divmod(round_index, len(MODES))
                order = mode_orders[size]
                if block % 2:
                    order = list(reversed(order))
                order = order[mode_shift:] + order[:mode_shift]
                for position, mode in enumerate(order, 1):
                    jobs.append(dict(session=session, run=repeat, n_test=size, mode=mode,
                                     size_position=size_position, mode_position=position))
    return jobs


def phase(kind, **fields):
    print(json.dumps(dict(phase=kind, **fields)), flush=True)


def run_session(args):
    folder = args.out / f"session_{args.worker:02d}"
    folder.mkdir()
    configure("gpu2")
    check_voting()
    affinity = set(os.sched_getaffinity(0))
    cpu = min(affinity)
    x, labels, queries, truth = load_inputs(max(args.sizes))
    expected = {}
    runs = {(size, mode): [] for size in args.sizes for mode in MODES}
    warmups = {key: [] for key in runs}
    profiles = {key: [] for key in runs}
    jobs = [r for r in schedule(args.sizes, args.sessions, args.repeats, args.seed)
            if r["session"] == args.worker]

    with ThreadPoolExecutor(max_workers=2) as pool:
        for size in args.sizes:
            for mode in MODES:
                os.sched_setaffinity(0, {cpu} if mode == "cpu" else affinity)
                for warmup in range(1, args.warmups + 1):
                    phase("warmup", mode=mode, n_test=size, warmup=warmup)
                    start = time.perf_counter_ns()
                    prediction, _ = classify(mode, x, labels, queries[:size], pool)
                    warmups[size, mode].append((time.perf_counter_ns() - start) / 1e9)
                    if size not in expected:
                        expected[size] = prediction
                    if not torch.equal(prediction, expected[size]):
                        raise ValueError(f"Warm-up prediction mismatch: {size}/{mode}")

        collected = []
        for job in jobs:
            size, mode = job["n_test"], job["mode"]
            os.sched_setaffinity(0, {cpu} if mode == "cpu" else affinity)
            for device in range(MODES[mode]):
                torch.cuda.reset_peak_memory_stats(device)
            phase("timed", **job)
            unix_start_ns = time.time_ns()
            cpu_start = time.process_time_ns()
            start = time.perf_counter_ns()
            prediction, _ = classify(mode, x, labels, queries[:size], pool)
            wall_s = (time.perf_counter_ns() - start) / 1e9
            host_cpu_s = (time.process_time_ns() - cpu_start) / 1e9
            unix_end_ns = time.time_ns()
            if not torch.equal(prediction, expected[size]):
                raise ValueError(f"Timed prediction mismatch: {job}")
            peak = [torch.cuda.max_memory_allocated(device) / 2**20
                    for device in range(MODES[mode])]
            correct = int((prediction == truth[:size]).sum())
            row = dict(**job, gpu_count=MODES[mode], n_train=len(x), k=K,
                       batch_size=BATCH, dtype="float64", intraop_threads=torch.get_num_threads(),
                       unix_start_ns=unix_start_ns, unix_end_ns=unix_end_ns,
                       wall_s=wall_s, host_cpu_s=host_cpu_s,
                       throughput_images_s=size / wall_s, amortized_ms_per_image=1000 * wall_s / size,
                       host_cpu_pct_one_core=100 * host_cpu_s / wall_s,
                       correct=correct, accuracy_pct=100 * correct / size,
                       gpu0_peak_alloc_mib=peak[0] if peak else None,
                       gpu1_peak_alloc_mib=peak[1] if len(peak) > 1 else None,
                       mismatch_vs_cpu=0, cpu_affinity=str(cpu) if mode == "cpu" else "available")
            runs[size, mode].append(row)
            collected.append(row)
            write_csv(folder / "runs.partial.csv", collected)
            phase("finished", **job, wall_s=wall_s)

        os.sched_setaffinity(0, affinity)
        for size in args.sizes:
            for mode in ("gpu1", "gpu2"):
                phase("profile", mode=mode, n_test=size)
                prediction, detail = classify(mode, x, labels, queries[:size], pool, profile=True)
                if not torch.equal(prediction, expected[size]):
                    raise ValueError("CUDA event profile changed predictions")
                profiles[size, mode] = detail

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    cells = {}
    for (size, mode), rows in runs.items():
        seconds = [r["wall_s"] for r in rows]
        correct = rows[0]["correct"]
        summary = dict(n_test=size, mode=mode, gpu_count=MODES[mode], n_train=len(x),
                       k=K, batch_size=BATCH, dtype="float64", intraop_threads=1,
                       repeats=args.repeats, median_wall_s=statistics.median(seconds),
                       mean_wall_s=statistics.mean(seconds), sample_std_wall_s=statistics.stdev(seconds),
                       min_wall_s=min(seconds), max_wall_s=max(seconds),
                       throughput_images_s=size / statistics.median(seconds),
                       amortized_ms_per_image=1000 * statistics.median(seconds) / size,
                       process_peak_rss_mib=rss, correct=correct, errors=size - correct,
                       accuracy_pct=100 * correct / size)
        cells[size, mode] = dict(summary=summary, runs=rows, profile=profiles[size, mode],
                                truth=truth[:size].tolist(), pred=expected[size].tolist(),
                                warmup_s=warmups[size, mode], affinity=sorted(affinity),
                                cpu_logical_id=cpu, interop=torch.get_num_interop_threads())
    cell_dir = folder / "cells"
    cell_dir.mkdir()
    for (size, mode), cell in cells.items():
        (cell_dir / f"{mode}_{size}.json").write_text(json.dumps(cell), encoding="utf-8")
    export(cells, folder, args.sizes, args.repeats)
    (folder / "runs.partial.csv").unlink()
    metadata = dict(session=args.worker, pid=os.getpid(), completed_utc=datetime.now(timezone.utc).isoformat(),
                    warmups_per_cell=args.warmups, measured_per_cell=args.repeats,
                    cpu_logical_id=cpu, affinity_available=sorted(affinity),
                    intraop_threads=1, interop_threads=torch.get_num_interop_threads(),
                    process_peak_rss_mib=rss,
                    csv_sha256={p.name: digest(p) for p in sorted(folder.glob("*.csv"))})
    (folder / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    phase("session_complete", session=args.worker, requests=len(jobs))


def monitor_boards(path, state, stop, errors):
    fields = ["unix_ns", "session", "worker_pid", "phase", "mode", "n_test", "run", "gpu",
              "gpu_busy_pct", "memory_busy_pct", "temperature_c", "sm_clock_mhz",
              "memory_clock_mhz", "power_w", "used_mib", "other_compute_processes"]
    try:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(2)]
            while not stop.is_set():
                context = dict(state)
                for gpu, handle in enumerate(handles):
                    usage = nv.nvmlDeviceGetUtilizationRates(handle)
                    processes = nv.nvmlDeviceGetComputeRunningProcesses(handle)
                    row = {key: context.get(key, "") for key in fields[:7]}
                    row.update(unix_ns=time.time_ns(), gpu=gpu, gpu_busy_pct=usage.gpu,
                               memory_busy_pct=usage.memory,
                               temperature_c=nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU),
                               sm_clock_mhz=nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM),
                               memory_clock_mhz=nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM),
                               power_w=nv.nvmlDeviceGetPowerUsage(handle) / 1000,
                               used_mib=nv.nvmlDeviceGetMemoryInfo(handle).used / 2**20,
                               other_compute_processes=sum(p.pid != context.get("worker_pid") for p in processes))
                    writer.writerow(row)
                stream.flush()
                stop.wait(0.2)
    except Exception as error:
        errors.append(repr(error))


def run(args):
    if args.out.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.out}")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("This comparison requires two visible CUDA devices")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="." + args.out.name + "-", dir=args.out.parent))
    jobs = schedule(args.sizes, args.sessions, args.repeats, args.seed)
    write_csv(stage / "schedule.csv", jobs)
    archive_md5 = {name: hashlib.md5((ROOT / "data/MNIST/raw" / name).read_bytes()).hexdigest()
                   for name, _ in torchvision.datasets.MNIST.resources}
    if archive_md5 != dict(torchvision.datasets.MNIST.resources):
        raise ValueError("MNIST archives do not match the published checksums")
    names = ("knn.py", "classify.py", "benchmark.py", "benchmark_sessions.py")
    environment = dict(started_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                       python=platform.python_version(), torch=str(torch.__version__),
                       torchvision=str(torchvision.__version__), cuda=torch.version.cuda,
                       sources={name: digest(ROOT / name) for name in names}, archive_md5=archive_md5,
                       sizes=args.sizes, sessions=args.sessions, measured_per_cell=args.repeats,
                       warmups_per_cell=args.warmups, order_seed=args.seed,
                       session_unit="A fresh process, loading once and warming every mode/size before timed requests",
                       measurement="Host-ready FP64 inputs to ordered CPU labels; per-request reference uploads",
                       schedule="Rotated size and mode orders; mode order reversed in alternating three-round blocks",
                       monitor="Separate controller process samples both boards every 0.2 s",
                       gpu_inventory=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"], text=True),
                       cpu_inventory=subprocess.check_output(["lscpu", "--json"], text=True))
    (stage / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(f"Staging results in {stage}", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    nv.nvmlInit()
    state = {}
    stop = threading.Event()
    errors = []
    monitor = threading.Thread(target=monitor_boards, args=(stage / "boards.csv", state, stop, errors))
    monitor.start()
    try:
        for session in range(1, args.sessions + 1):
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                       "--out", str(stage), "--sessions", str(args.sessions), "--repeats", str(args.repeats),
                       "--warmups", str(args.warmups), "--seed", str(args.seed), "--worker", str(session),
                       "--sizes", *map(str, args.sizes)]
            with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, text=True) as child:
                state.update(session=session, worker_pid=child.pid, phase="starting", mode="", n_test="", run="")
                for line in child.stdout:
                    event = json.loads(line)
                    state.update(event)
                    print(json.dumps(dict(session=session, **{k: v for k, v in event.items() if k != "session"})), flush=True)
                    (stage / "status.json").write_text(json.dumps(dict(state)), encoding="utf-8")
                if child.wait():
                    raise RuntimeError(f"Session {session} failed; partial evidence is in {stage}")
    finally:
        stop.set()
        monitor.join()
        nv.nvmlShutdown()
    if errors:
        raise RuntimeError(f"Board monitoring failed: {errors}")
    environment["completed_utc"] = datetime.now(timezone.utc).isoformat()
    environment["csv_sha256"] = {p.name: digest(p) for p in sorted(stage.glob("*.csv"))}
    (stage / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    stage.rename(args.out)
    print(f"Completed {len(jobs)} timed requests in {args.sessions} sessions: {args.out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/main_repeat")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000])
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=435)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.sessions < 2 or args.repeats < 2 or args.warmups < 1
            or len(set(args.sizes)) != len(args.sizes)
            or not all(1 <= size <= 10000 for size in args.sizes)
            or (args.worker is not None and not 1 <= args.worker <= args.sessions)):
        parser.error("Use distinct sizes in 1..10000, at least two sessions and repeats, and a positive warm-up count")
    args.out = args.out.resolve()
    if args.worker is not None:
        run_session(args)
    else:
        run(args)
