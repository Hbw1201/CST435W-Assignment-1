"""Compare one-shot and resident-reference MNIST inference on the same two H100s."""
import argparse
import csv
import json
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import pynvml as nv
import torch

from classify import BATCH, classify, configure, load_inputs
from knn import ROOT, predict_knn, write_csv


class BoardSampler:
    """NVML board-wide samples. Keep the number of other contexts, not their IDs."""

    def __init__(self):
        nv.nvmlInit()
        self.handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
        self.rows = []
        self.stop = threading.Event()
        self.phase = "before"

    def sample(self):
        for i, handle in enumerate(self.handles):
            tick = time.time_ns()
            label = self.phase
            memory = nv.nvmlDeviceGetMemoryInfo(handle)
            util = nv.nvmlDeviceGetUtilizationRates(handle)
            processes = nv.nvmlDeviceGetComputeRunningProcesses(handle)
            self.rows.append(dict(unix_ns=tick, phase=label, nvml_gpu=i,
                                  board_gpu_busy_pct=util.gpu, board_memory_busy_pct=util.memory,
                                  board_power_w=nv.nvmlDeviceGetPowerUsage(handle) / 1000,
                                  used_mib=memory.used / 2**20, total_mib=memory.total / 2**20,
                                  other_compute_processes=sum(p.pid != os.getpid() for p in processes)))

    def run(self):
        while not self.stop.wait(0.2):
            self.sample()


@torch.inference_mode()
def place_reference(device, x, y):
    with torch.cuda.device(device):
        x_gpu, y_gpu = x.cuda(device), y.cuda(device)
        torch.cuda.synchronize(device)
        return x_gpu, y_gpu


@torch.inference_mode()
def resident_shard(device, reference, query, batch_size=BATCH):
    with torch.cuda.device(device):
        x, y = reference
        guess = predict_knn(x, y, query.cuda(device), batch_size).cpu()
        torch.cuda.synchronize(device)
        return guess


def predict_resident(cache, queries, pool, batch_size=BATCH):
    jobs = [pool.submit(resident_shard, i, cache[i], piece, batch_size)
            for i, piece in enumerate(queries.tensor_split(len(cache)))]
    parts = [job.result() for job in jobs]
    return parts[0] if len(parts) == 1 else torch.cat(parts)


def run_case(placement, gpu_count, x, y, q, pool, cache):
    if placement == "per_request":
        return classify(f"gpu{gpu_count}", x, y, q, pool)[0]
    return predict_resident(cache[:gpu_count], q, pool)


def cpu_predictions(truth):
    with (ROOT / "results/full_run/predictions.csv").open(newline="") as stream:
        previous = [r for r in csv.DictReader(stream) if int(r["n_test"]) == len(truth)]
    if [int(r["test_index"]) for r in previous] != list(range(len(truth))):
        raise ValueError("Incomplete or unordered CPU reference")
    if [int(r["true_label"]) for r in previous] != truth.tolist():
        raise ValueError("The saved test labels differ")
    return torch.tensor([int(r["cpu"]) for r in previous])


def summarize(rows):
    summary = []
    for placement in ("per_request", "resident"):
        times = {}
        for count in (1, 2):
            group = [r for r in rows if r["placement"] == placement and r["gpu_count"] == count]
            durations = [r["wall_ms"] for r in group]
            times[count] = median(durations)
            summary.append(dict(placement=placement, gpu_count=count, measured=len(group),
                                median_ms=times[count], min_ms=min(durations), max_ms=max(durations),
                                images_per_s=10_000_000 / times[count],
                                one_to_two_speedup=times[1] / times[count],
                                scale_efficiency_pct=100 * times[1] / (count * times[count]),
                                correct=group[0]["correct"], mismatch_vs_cpu=0))
    return summary


def run(output, trace, repeats):
    output.mkdir(parents=True, exist_ok=False)
    sampler = BoardSampler()
    sampler.sample()
    monitor = threading.Thread(target=sampler.run, daemon=True)
    monitor.start()
    try:
        configure("gpu2")
        x, y, q, truth = load_inputs(10_000)
        expected = cpu_predictions(truth)
        correct = (expected == truth).sum().item()
        configs = [("per_request", 1), ("resident", 2), ("per_request", 2), ("resident", 1)]
        rows = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            sampler.phase = "warmup_per_request"
            for g in (1, 2):
                if not torch.equal(run_case("per_request", g, x, y, q, pool, None), expected):
                    raise ValueError("Warm-up differs from the recorded CPU labels")
            sampler.phase = "place_reference_on_both"
            start = time.perf_counter_ns()
            uploads = [pool.submit(place_reference, i, x, y) for i in range(2)]
            cache = [job.result() for job in uploads]
            setup_ms = (time.perf_counter_ns() - start) / 1e6
            sampler.phase = "warmup_resident"
            for g in (1, 2):
                if not torch.equal(run_case("resident", g, x, y, q, pool, cache), expected):
                    raise ValueError("Resident warm-up differs from the recorded CPU labels")
            # Rotate the four cases between rounds to spread out changes in shared load.
            for repeat in range(1, repeats + 1):
                shift = (repeat - 1) % len(configs)
                for placement, count in configs[shift:] + configs[:shift]:
                    label = f"knn_diag/{placement}/{count}/{repeat}"
                    sampler.phase = label
                    # Workers synchronize their devices before returning. These two
                    # extra calls on the main thread delimit the job in the CUDA API trace.
                    if trace:
                        torch.cuda.synchronize(0)
                    tick = time.perf_counter_ns()
                    prediction = run_case(placement, count, x, y, q, pool, cache)
                    wall_ms = (time.perf_counter_ns() - tick) / 1e6
                    if trace:
                        torch.cuda.synchronize(0)
                    sampler.phase = "verify"
                    if not torch.equal(prediction, expected):
                        raise ValueError(f"{label} differs from the recorded CPU labels")
                    rows.append(dict(placement=placement, gpu_count=count, repeat=repeat,
                                     n_train=len(x), n_test=len(q), batch=BATCH, wall_ms=wall_ms,
                                     images_per_s=1000 * len(q) / wall_ms, correct=correct,
                                     mismatch_vs_cpu=0))
        sampler.phase = "after"
        write_csv(output / "runs.csv", rows)
        write_csv(output / "summary.csv", summarize(rows))
        metadata = dict(utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                        python=platform.python_version(), torch=str(torch.__version__),
                        cuda=torch.version.cuda, cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                        cuda_devices=[torch.cuda.get_device_name(i) for i in range(2)],
                        trace_markers=trace, timer="host inputs ready to CPU labels ready, perf_counter_ns",
                        repeats=repeats, reference_and_labels_bytes_per_gpu=x.nbytes + y.nbytes,
                        per_request_uploads_training_data=True, resident_recomputes_reference_norms=True,
                        one_time_concurrent_placement_on_both_ms=setup_ms,
                        input_order_matches_cpu=True, predictions_verified=len(rows) * len(q))
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        sampler.stop.set()
        monitor.join()
        sampler.sample()
        write_csv(output / "boards.csv", sampler.rows)
        nv.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh result directory")
    parser.add_argument("--trace", action="store_true", help="Bracket jobs on the main CUDA API timeline")
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run(args.out, args.trace, args.repeats)
