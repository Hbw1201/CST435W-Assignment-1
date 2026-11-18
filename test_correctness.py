"""Check deterministic neighbours, voting, batching and the public classifier entry points."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from classify import MODES, classify, configure
from knn import K, ROOT, predict_knn


def integer_reference(images, labels, queries):
    predictions = []
    for query in queries:
        differences = images.astype(np.int64) - query.astype(np.int64)
        distances = np.einsum("ij,ij->i", differences, differences, dtype=np.int64)
        nearest = np.argsort(distances, kind="stable")[:K]
        predictions.append(np.bincount(labels[nearest], minlength=10).argmax())
    return np.array(predictions, dtype=np.int64)


def verify(cuda, run_cli, output):
    configure("gpu2" if cuda else "cpu")
    modes = list(MODES) if cuda else ["cpu"]
    devices = ["cpu", "cuda:0", "cuda:1"] if cuda else ["cpu"]
    records = []
    rng = np.random.default_rng(435)
    cases = [
        ("equal_distance_at_cutoff", np.array([[10], [11], [11], [12], [12], [12]], dtype=np.uint8),
         np.array([9, 4, 9, 4, 2, 9]), np.array([[10]], dtype=np.uint8), [4]),
        ("equal_vote_counts", np.array([[0], [1], [2], [3], [4], [50]], dtype=np.uint8),
         np.array([4, 9, 4, 9, 2, 2]), np.array([[0]], dtype=np.uint8), [4]),
        ("extreme_raw_pixels", np.array([[v] * 784 for v in [0, 255, 254, 1, 127, 128, 126]], dtype=np.uint8),
         np.array([0, 9, 9, 0, 5, 5, 5]), np.array([[0] * 784, [255] * 784], dtype=np.uint8), None),
    ]
    for name, images, labels, queries, hand_answer in cases:
        expected = integer_reference(images, labels, queries)
        if hand_answer is not None and expected.tolist() != hand_answer:
            raise AssertionError(f"Incorrect hand example: {name}")
        for device in devices:
            x = torch.tensor(images, dtype=torch.float64, device=device)
            y = torch.tensor(labels, dtype=torch.int64, device=device)
            q = torch.tensor(queries, dtype=torch.float64, device=device)
            for batch in (1, 7, 256):
                actual = predict_knn(x, y, q, batch).cpu().numpy()
                np.testing.assert_array_equal(actual, expected, err_msg=f"{name}/{device}/{batch}")
                records.append(dict(case=name, mode=device, batch=batch, queries=len(q), matched=True))

    images = rng.integers(0, 256, size=(19, 784), dtype=np.uint8)
    labels = np.arange(len(images), dtype=np.int64) % 10
    queries = rng.integers(0, 256, size=(513, 784), dtype=np.uint8)
    expected = integer_reference(images, labels, queries)
    x = torch.tensor(images, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.int64)
    q = torch.tensor(queries, dtype=torch.float64)
    with ThreadPoolExecutor(max_workers=2) as workers:
        for count in (1, 2, 255, 256, 257, 513):
            for mode in modes:
                actual, _ = classify(mode, x, y, q[:count], workers)
                np.testing.assert_array_equal(actual.numpy(), expected[:count],
                                              err_msg=f"partition_and_batch/{mode}/{count}")
                records.append(dict(case="partition_and_batch", mode=mode, batch=256,
                                    queries=count, matched=True))

    cli_records = []
    if run_cli:
        from classify import load_inputs
        train, targets, test, truth = load_inputs(10)
        exact = integer_reference(train.numpy().astype(np.uint8), targets.numpy(),
                                  test.numpy().astype(np.uint8))
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for mode in modes:
            command = [sys.executable, "-B", str(ROOT / "classify.py"), "--mode", mode, "--limit", "257"]
            text = subprocess.check_output(command, text=True, env=environment, cwd=ROOT)
            lines = text.strip().splitlines()
            rows = [[int(v) for v in line.split()] for line in lines[1:-1]]
            assert rows == [[i, int(truth[i]), int(exact[i])] for i in range(10)]
            assert lines[-1].startswith(mode + ": ") and "/257 correct; first-call wall " in lines[-1]
            cli_records.append(dict(mode=mode, limit=257, stdout=text, first_ten_integer_match=True))
        for limit in (0, 10001):
            command = [sys.executable, "-B", str(ROOT / "classify.py"), "--mode", "cpu", "--limit", str(limit)]
            result = subprocess.run(command, text=True, env=environment, capture_output=True, cwd=ROOT)
            assert result.returncode != 0 and "--limit must be between 1 and 10000" in result.stderr
            cli_records.append(dict(mode="cpu", limit=limit, rejected=True))

    result = dict(seed=435, cuda_checked=cuda, checks=records, cli_checks=cli_records,
                  checks_passed=len(records) + len(cli_records),
                  oracle="Direct int64 pixel differences, exhaustive distances and stable full sorting",
                  max_mnist_distance=784 * 255**2,
                  max_mnist_key=784 * 255**2 * 60000 + 59999,
                  versions=dict(python=sys.version.split()[0], torch=str(torch.__version__), numpy=np.__version__),
                  sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                           for name in ("knn.py", "classify.py", "test_correctness.py")})
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Choose a fresh validation output: {output}")
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(checks_passed=result["checks_passed"], cuda_checked=cuda, cli_checked=run_cli)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Also require and verify both CUDA devices")
    parser.add_argument("--cli", action="store_true", help="Check real MNIST console output and invalid limits")
    parser.add_argument("--out", type=Path, help="Fresh JSON evidence file")
    args = parser.parse_args()
    verify(args.cuda, args.cli, args.out)
