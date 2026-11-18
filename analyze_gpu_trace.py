"""Attribute kernel and copy intervals in a diagnose_gpu.py Nsight Systems CUDA trace."""
import argparse
import csv
import sqlite3
from collections import defaultdict
from math import fsum
from pathlib import Path


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def length(intervals):
    return sum(end - start for start, end in intervals)


def common_duration(first, second):
    i = j = duration = 0
    while i < len(first) and j < len(second):
        a, b = first[i]
        c, d = second[j]
        duration += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return duration


def save(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def analyze(database, runfile, out, by_batch=False, segments=False):
    with runfile.open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        processes = list(db.execute(
            "SELECT DISTINCT k.globalPid, p.pid FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN PROCESSES p ON p.globalPid=k.globalPid"))
        if len(processes) != 1:
            raise ValueError("Expected one profiled CUDA process")
        global_pid, pid = processes[0]
        # Nsight's serialized thread ID combines globalPid and the low 24 bits of TID.
        # On Linux the main thread's TID is its PID. Worker synchronization calls are excluded.
        markers = list(db.execute(
            "SELECT r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId WHERE r.globalTid=? "
            "AND s.value LIKE 'cudaDeviceSynchronize%' ORDER BY r.start",
            (global_pid | pid,)))
        if len(markers) != 2 * len(runs):
            raise ValueError(f"{len(markers)} main-thread markers for {len(runs)} jobs")
        rows = []
        intervals = []
        for run, (_, start), (end, _) in zip(runs, markers[::2], markers[1::2]):
            count = int(run["gpu_count"])
            batch_field = {"batch": int(run["batch"])} if by_batch else {}
            devices = range(count)
            bounds = (global_pid, start, end)
            identity = dict(placement=run["placement"], gpu_count=count,
                            batch=int(run["batch"]), repeat=int(run["repeat"]))
            kernels = defaultdict(list)
            for device, a, b in db.execute(
                    "SELECT deviceId, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL "
                    "WHERE globalPid=? AND end>? AND start<?", bounds):
                if not start <= a < b <= end:
                    raise ValueError("A kernel crosses a job boundary")
                kernels[device].append((a, b))
                if segments:
                    intervals.append(dict(identity, cuda_device=device, activity="Kernel",
                                          start_ms=(a-start)/1e6, end_ms=(b-start)/1e6,
                                          trace_wall_ms=(end-start)/1e6))
            if set(kernels) != set(devices):
                raise ValueError("The active GPUs differ from the requested GPU count")
            copies = defaultdict(list)
            for device, kind, a, b, size in db.execute(
                    "SELECT c.deviceId, e.label, c.start, c.end, c.bytes "
                    "FROM CUPTI_ACTIVITY_KIND_MEMCPY c JOIN ENUM_CUDA_MEMCPY_OPER e "
                    "ON e.id=c.copyKind WHERE c.globalPid=? AND c.end>? AND c.start<?", bounds):
                if not start <= a <= b <= end:
                    raise ValueError("A copy crosses a job boundary")
                copies[device, kind].append((a, b, size))
                if segments:
                    intervals.append(dict(identity, cuda_device=device, activity=kind,
                                          start_ms=(a-start)/1e6, end_ms=(b-start)/1e6,
                                          trace_wall_ms=(end-start)/1e6))
            unions = {i: merged(kernels[i]) for i in devices}
            both = common_duration(unions[0], unions[1]) if count == 2 else None
            n = int(run["n_test"]) // count
            for i in devices:
                h2d = copies[i, "Host-to-Device"]
                d2h = copies[i, "Device-to-Host"]
                training_bytes = int(run["n_train"]) * (784 + 1) * 8
                expected = n * 784 * 8 + (training_bytes if run["placement"] == "per_request" else 0)
                actual = sum(size for _, _, size in h2d)
                if actual != expected or sum(size for _, _, size in d2h) != n * 8:
                    raise ValueError("Transfer bytes differ from the expected images and labels")
                wall = end - start
                kernel_ns = length(unions[i])
                rows.append(dict(placement=run["placement"], gpu_count=count, **batch_field,
                                 repeat=int(run["repeat"]), cuda_device=i,
                                 host_wall_ms=float(run["wall_ms"]), trace_wall_ms=wall / 1e6,
                                 kernels=len(kernels[i]), kernel_union_ms=kernel_ns / 1e6,
                                 kernel_span_pct=100 * kernel_ns / wall,
                                 h2d_bytes=actual, h2d_copy_ms=length(merged((a, b) for a, b, _ in h2d)) / 1e6,
                                 d2h_bytes=n * 8, d2h_copy_ms=length(merged((a, b) for a, b, _ in d2h)) / 1e6,
                                 concurrent_kernel_ms=both / 1e6 if both is not None else None))
    groups = defaultdict(list)
    for row in rows:
        groups[row["placement"], row["gpu_count"], row.get("batch", 0), row["cuda_device"]].append(row)
    summary = []
    for (placement, count, batch, device), group in sorted(groups.items()):
        batch_field = {"batch": batch} if by_batch else {}
        wall = fsum(r["trace_wall_ms"] for r in group)
        kernel = fsum(r["kernel_union_ms"] for r in group)
        overlap = fsum(r["concurrent_kernel_ms"] for r in group) if count == 2 else None
        summary.append(dict(placement=placement, gpu_count=count, **batch_field, cuda_device=device,
                            repeats=len(group), mean_traced_wall_ms=wall / len(group),
                            mean_kernel_union_ms=kernel / len(group), kernel_span_pct=100 * kernel / wall,
                            both_gpus_kernel_span_pct=100 * overlap / wall if overlap is not None else None,
                            h2d_bytes_per_request=group[0]["h2d_bytes"],
                            mean_h2d_copy_ms=fsum(r["h2d_copy_ms"] for r in group) / len(group)))
    save(out / "kernel_timeline.csv", rows)
    save(out / "kernel_summary.csv", summary)
    if segments:
        save(out / "segments.csv", intervals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("traced_runs", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--by-batch", action="store_true", help="Keep a separate group per batch size")
    parser.add_argument("--segments", action="store_true", help="Export request-relative kernel and copy intervals")
    args = parser.parse_args()
    analyze(args.database, args.traced_runs, args.output_directory, args.by_batch, args.segments)
