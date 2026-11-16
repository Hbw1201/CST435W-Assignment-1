"""Single-threaded exact 5-NN on MNIST. Run: python3 knn.py --limit 100"""
import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns, process_time_ns

import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parent
K = 5


def predict_knn(reference, labels, queries, batch_size):
    """Inputs are raw MNIST pixel integers represented exactly in float64."""
    n = len(reference)
    ref_norm2 = reference.square().sum(dim=1)
    ref_ids = torch.arange(n, dtype=torch.int64, device=reference.device)
    predictions = []
    for batch in queries.split(batch_size):
        # Squared Euclidean distances: ||q||^2 + ||x||^2 - 2(q . x).
        distances = torch.addmm(ref_norm2, batch, reference.T, alpha=-2)
        distances.add_(batch.square().sum(dim=1, keepdim=True))

        # The integer key orders by (distance, training row). Its maximum
        # for MNIST is below 3.1e12, well within int64.
        keys = distances.to(torch.int64).mul_(n).add_(ref_ids)
        nearest = keys.topk(K, dim=1, largest=False, sorted=False).indices
        votes = F.one_hot(labels[nearest], num_classes=10).sum(dim=1)
        predictions.append(votes.argmax(dim=1))  # first/smallest digit on ties
    return torch.cat(predictions)


def check_voting():
    # The five closest rows are 0..4. Votes tie between digits 4 and 9.
    ref = torch.tensor([[10], [11], [11], [12], [12], [12]], dtype=torch.float64)
    labels = torch.tensor([9, 4, 9, 4, 2, 9])
    if predict_knn(ref, labels, ref[:1], 1).item() != 4:
        raise RuntimeError("Hand-checked neighbour/voting example failed")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    check_voting()
    train = MNIST(args.data_dir, train=True, download=False)
    test = MNIST(args.data_dir, train=False, download=False)
    if (len(train), len(test)) != (60000, 10000):
        raise ValueError("The standard MNIST train/test split is required")
    reference = train.data.flatten(1).to(torch.float64)
    queries = test.data[:args.limit].flatten(1).to(torch.float64)
    truth = test.targets[:args.limit]
    config = dict(mode="cpu_1t", n_train=len(reference), n_test=len(queries), k=K,
                  batch_size=args.batch_size, dtype="float64",
                  cpu_compute_threads=torch.get_num_threads(), gpu_count=0)

    warm_start = perf_counter_ns()
    baseline = predict_knn(reference, train.targets, queries, args.batch_size)
    warmup_s = (perf_counter_ns() - warm_start) / 1e9
    correct_count = int((baseline == truth).sum().item())
    accuracy_pct = 100.0 * correct_count / args.limit
    runs = []
    for number in range(1, args.repeats + 1):
        cpu_start = process_time_ns()
        wall_start = perf_counter_ns()
        predicted = predict_knn(reference, train.targets, queries, args.batch_size)
        wall_s = (perf_counter_ns() - wall_start) / 1e9
        cpu_s = (process_time_ns() - cpu_start) / 1e9
        if not torch.equal(predicted, baseline):
            raise RuntimeError(f"Prediction difference on repetition {number}")
        runs.append(dict(config, run=number, wall_s=wall_s, process_cpu_s=cpu_s,
                         cpu_pct_of_one_core=100 * cpu_s / wall_s,
                         throughput_images_s=args.limit / wall_s,
                         amortized_ms_per_image=1000 * wall_s / args.limit,
                         correct_count=correct_count, accuracy_pct=accuracy_pct))
        print(f"run {number}: {wall_s:.6f} s", flush=True)

    samples = []
    matrix = [[0] * 10 for _ in range(10)]
    for index, (actual, guess) in enumerate(zip(truth.tolist(), predicted.tolist())):
        samples.append(dict(test_index=index, true_label=actual,
                            predicted_label=guess, correct=int(actual == guess)))
        matrix[actual][guess] += 1
    confusion = [dict(true_label=i, predicted_label=j, count=matrix[i][j])
                 for i in range(10) for j in range(10)]
    seconds = [row["wall_s"] for row in runs]
    median_s = statistics.median(seconds)
    peak_rss_mib = None
    if sys.platform.startswith("linux"):
        import resource
        peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    summary = dict(config, measured_runs=len(runs), warmup_runs=1,
                   median_wall_s=median_s, mean_wall_s=statistics.mean(seconds),
                   sample_std_wall_s=statistics.stdev(seconds) if len(runs) > 1 else None,
                   min_wall_s=min(seconds), max_wall_s=max(seconds),
                   throughput_at_median_images_s=args.limit / median_s,
                   amortized_ms_per_image=1000 * median_s / args.limit,
                   peak_process_rss_mib=peak_rss_mib, correct_count=correct_count,
                   error_count=args.limit-correct_count, accuracy_pct=accuracy_pct)
    meta = dict(recorded_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                python=platform.python_version(), torch=str(torch.__version__),
                torchvision=str(torchvision.__version__),
                intraop_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                affinity_logical_cpus=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                data_root=str(args.data_dir.resolve()), first_test_index=0, last_test_index=args.limit-1,
                warmup_wall_s=warmup_s, prediction_match_all_repeats=True,
                clock="perf_counter_ns wall time; process_time_ns process CPU time",
                time_scope="Host FP64 inputs ready to CPU labels ready. Includes reference norms, distances, top-5, voting and batch concatenation. Disk I/O, input conversion, scoring and recording excluded.",
                amortized_ms_per_image="Batch wall time / number of test images, in ms; not individual query latency.",
                memory_scope="Peak Linux process RSS through all runs, including imports, loading, warm-up and reference tensors (MiB).",
                tie_rules="Equal squared distance: smaller training index. Equal votes: smaller digit.")

    stem = f"cpu_{args.limit}"
    args.output_dir.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temp_name:
        temp = Path(temp_name)
        for suffix, rows in [(".csv", samples), ("_runs.csv", runs),
                             ("_summary.csv", [summary]), ("_confusion.csv", confusion)]:
            write_csv(temp / (stem + suffix), rows)
        meta["csv_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(temp.glob("*.csv"))}
        metadata_file = temp / (stem + "_metadata.json")
        metadata_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        for path in sorted(temp.glob("*.csv")) + [metadata_file]:
            path.replace(args.output_dir / path.name)
    print(f"{correct_count}/{args.limit} correct; median {median_s:.6f} s; "
          f"{args.limit / median_s:.2f} images/s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not (1 <= args.limit <= 10000 and args.batch_size > 0 and args.repeats > 0):
        parser.error("limit must be 1..10000; batch-size and repeats must be positive")
    run(args)
