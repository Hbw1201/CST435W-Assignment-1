"""Measure resident-reference exact MNIST 5-NN on two GPUs over per-card batch sizes."""
import argparse
import hashlib
import json
import os
import platform
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pynvml as nv
import torch

from classify import configure, load_inputs
from diagnose_gpu import BoardSampler, cpu_predictions, place_reference, predict_resident
from knn import ROOT, write_csv

DEFAULT_BATCHES = [256, 512, 1024, 2048, 4096, 5000]


def run(batches, repeats, trace, out):
    out.mkdir(parents=True, exist_ok=False)
    sampler = BoardSampler()
    sampler.sample()
    monitor = threading.Thread(target=sampler.run, daemon=True)
    monitor.start()
    rows = []
    seed = 435
    order = list(batches)
    random.Random(seed).shuffle(order)
    try:
        configure("gpu2")
        x, y, q, truth = load_inputs(10_000)
        expected = cpu_predictions(truth)
        correct = int((expected == truth).sum())
        with ThreadPoolExecutor(max_workers=2) as pool:
            sampler.phase = "place_reference"
            start = time.perf_counter_ns()
            uploads = [pool.submit(place_reference, i, x, y) for i in range(2)]
            cache = [job.result() for job in uploads]
            setup_ms = (time.perf_counter_ns() - start) / 1e6
            # Warm both devices at every shape before timing. All later requests
            # reuse the same device tensors for the reference pixels and labels.
            for warm_order in (batches, list(reversed(batches))):
                for batch in warm_order:
                    sampler.phase = f"warmup/{batch}"
                    if not torch.equal(predict_resident(cache, q, pool, batch), expected):
                        raise ValueError(f"Different warm-up predictions at batch {batch}")
            for repeat in range(1, repeats + 1):
                block, shift = divmod(repeat - 1, len(order))
                sequence = order if block % 2 == 0 else list(reversed(order))
                for position, batch in enumerate(sequence[shift:] + sequence[:shift], 1):
                    for i in range(2):
                        torch.cuda.reset_peak_memory_stats(i)
                    sampler.phase = f"request/{batch}/{repeat}"
                    if trace:
                        torch.cuda.synchronize(0)
                    unix_start = time.time_ns()
                    tick = time.perf_counter_ns()
                    prediction = predict_resident(cache, q, pool, batch)
                    wall_ms = (time.perf_counter_ns() - tick) / 1e6
                    if trace:
                        torch.cuda.synchronize(0)
                    sampler.phase = "verify"
                    if not torch.equal(prediction, expected):
                        raise ValueError(f"Different predictions at batch {batch}, round {repeat}")
                    rows.append(dict(placement="resident", gpu_count=2, repeat=repeat,
                                     position=position, batch=batch, n_train=len(x), n_test=len(q),
                                     chunks_per_gpu=(len(q) // 2 + batch - 1) // batch,
                                     unix_start_ns=unix_start, wall_ms=wall_ms,
                                     images_per_s=1000 * len(q) / wall_ms,
                                     gpu0_peak_alloc_mib=torch.cuda.max_memory_allocated(0) / 2**20,
                                     gpu1_peak_alloc_mib=torch.cuda.max_memory_allocated(1) / 2**20,
                                     correct=correct, mismatch_vs_cpu=0))
                tmp = out / "runs.tmp"
                write_csv(tmp, rows)
                tmp.replace(out / "runs.csv")
        sampler.phase = "after"
        names = ["sweep_batch.py", "diagnose_gpu.py", "classify.py", "knn.py"]
        version = nv.nvmlSystemGetDriverVersion()
        metadata = dict(utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                        python=platform.python_version(), torch=str(torch.__version__),
                        cuda=torch.version.cuda, driver=version.decode() if isinstance(version, bytes) else version,
                        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                        cuda_devices=[torch.cuda.get_device_name(i) for i in range(2)], batches=batches,
                        repeats=repeats, warmups_per_batch=2, order_seed=seed, first_round_order=order,
                        trace_markers=trace, training_bytes_resident_per_gpu=x.nbytes + y.nbytes,
                        reference_setup_ms_including_first_device_use=setup_ms, n_test=len(q),
                        predictions_verified=len(q) * len(rows), correct_per_request=correct,
                        source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names})
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    finally:
        sampler.stop.set()
        monitor.join()
        sampler.sample()
        write_csv(out / "boards.csv", sampler.rows)
        nv.nvmlShutdown()
    print(f"{len(rows)} requests verified against all {len(q)} CPU labels; results in {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--repeats", type=int, default=18)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--out", required=True, type=Path, help="Fresh result directory")
    args = parser.parse_args()
    if len(set(args.batches)) != len(args.batches) or any(not 1 <= b <= 5000 for b in args.batches):
        parser.error("Choose distinct per-GPU batch sizes between 1 and the 5,000-image shard size")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run(args.batches, args.repeats, args.trace, args.out)
