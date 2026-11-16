"""Classify MNIST prefixes with the same exact 5-NN kernel on CPU or CUDA."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import torch
from torchvision.datasets import MNIST
from knn import ROOT, predict_knn

MODES = {"cpu": 0, "gpu1": 1, "gpu2": 2}
BATCH = 256


def configure(mode):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if mode == "cpu":
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    elif torch.cuda.device_count() < MODES[mode]:
        raise RuntimeError(f"{mode} requires {MODES[mode]} CUDA devices")


def load_inputs(limit):
    if not 1 <= limit <= 10000:
        raise ValueError("--limit must be between 1 and 10000")
    train = MNIST(ROOT / "data", train=True, download=False)
    test = MNIST(ROOT / "data", train=False, download=False)
    if (len(train), len(test)) != (60000, 10000):
        raise ValueError("Expected the standard MNIST split")
    return (train.data.flatten(1).double(), train.targets,
            test.data[:limit].flatten(1).double(), test.targets[:limit])


@torch.inference_mode()
def gpu_part(index, reference, labels, queries, profile):
    device = torch.device("cuda", index)
    with torch.cuda.device(device):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if profile else []
        if profile:
            events[0].record()
        x, y, q = reference.to(device), labels.to(device), queries.to(device)
        if profile:
            events[1].record()
        guesses = predict_knn(x, y, q, BATCH)
        if profile:
            events[2].record()
        host = guesses.cpu()
        if profile:
            events[3].record()
        torch.cuda.synchronize(device)
        detail = {}
        if profile:
            detail = dict(device=index, shard_n=len(q),
                          h2d_ms=events[0].elapsed_time(events[1]),
                          compute_ms=events[1].elapsed_time(events[2]),
                          d2h_ms=events[2].elapsed_time(events[3]))
        return host, detail


@torch.inference_mode()
def classify(mode, reference, labels, queries, workers, profile=False):
    if mode == "cpu":
        return predict_knn(reference, labels, queries, BATCH), []
    pieces = queries.tensor_split(MODES[mode])
    jobs = [workers.submit(gpu_part, i, reference, labels, q, profile)
            for i, q in enumerate(pieces) if len(q)]
    done = [job.result() for job in jobs]  # submission order = input order
    joined = done[0][0] if len(done) == 1 else torch.cat([p for p, _ in done])
    return joined, [row for _, row in done if row]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    configure(args.mode)
    x, y, q, truth = load_inputs(args.limit)
    with ThreadPoolExecutor(max_workers=max(1, MODES[args.mode])) as pool:
        start = perf_counter()
        pred, _ = classify(args.mode, x, y, q, pool)
        elapsed = perf_counter() - start
    print("index  true  predicted")
    for i in range(min(10, len(q))):
        print(f"{i:5d}  {int(truth[i]):4d}  {int(pred[i]):9d}")
    print(f"{args.mode}: {int((pred == truth).sum())}/{len(q)} correct; "
          f"first-call wall {elapsed:.6f} s")
