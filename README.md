# CST435W Assignment 1

Exact five-nearest-neighbour classification of MNIST on one CPU thread, one NVIDIA H100, and two H100s in the same host. All three modes search the same 60,000 labelled reference images. The two-GPU implementation partitions the queries and returns predictions in input order.

## Run the classifier

The measurements used Linux, Python 3.10.12, PyTorch 2.3.0 with CUDA 12.1, and torchvision 0.18.0. The CPU example needs no GPU. The full benchmark needs two CUDA devices and the NVIDIA driver. Install in a fresh environment, then fetch the standard MNIST files once.

```bash
python3 -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install 'numpy<2' matplotlib nvidia-ml-py
python3 -c "from torchvision.datasets import MNIST; MNIST('data', train=True, download=True); MNIST('data', train=False, download=True)"
python3 classify.py --mode cpu --limit 100
python3 classify.py --mode gpu1 --limit 100
python3 classify.py --mode gpu2 --limit 100
```

Each command prints test indices, true and predicted digits, a correct count and first-call elapsed time. For each query, the implementation ranks squared Euclidean distances, breaks distance ties by training row, selects five neighbours and breaks vote ties by the smaller digit.

## Reproduce the comparisons

Run from the repository root. Each measurement command takes a fresh output directory. The resident experiment uses the full-test CPU predictions produced by the first benchmark.

```bash
python3 benchmark.py --out results/full_run
python3 verify_results.py results/full_run --raw data/MNIST/raw --figures figures
python3 plot_results.py results/full_run --out figures

python3 diagnose_gpu.py --out results/placement
out=results/batch_sweep
python3 sweep_batch.py --out "$out/timed" --repeats 18
nsys profile --sample=none --cpuctxsw=none --trace=cuda --output="$out/sweep" \
  python3 sweep_batch.py --out "$out/traced" --trace --repeats 12
nsys export --type=sqlite --output="$out/sweep.sqlite" "$out/sweep.nsys-rep"
python3 analyze_gpu_trace.py --by-batch "$out/sweep.sqlite" "$out/traced/runs.csv" "$out"
python3 summarize_batch.py "$out"
```

The plotting command writes `performance.pdf` and `confusion.pdf`, plus editable SVGs and PNG previews. The performance plot shows all individual GPU times and throughput on all three platforms. The confusion matrix prints all 100 counts, including correct predictions and zeros, alongside error rates for each true digit.

The trace commands require NVIDIA Nsight Systems with CUDA tracing. To repeat the six-size experiment with five timings and five separate traces per size, choose a fresh `out` and set both repeat counts to 5.

## Measured results

For 10,000 queries at batch 256 with per-request training uploads, median elapsed times were 19.946 s on one CPU thread, 114.77 ms on one H100 and 78.66 ms on two. Predictions matched on every image, with 9,688 correct.

With the training set resident on both GPUs, the recommended batch was 2,048 queries per GPU. Across two timing sessions its mean stayed within 1.52% of the fastest mean in each session. Batch 5,000 led in the five-run repeat. The peak PyTorch tensor allocations at these settings were 3.19 and 4.92 GiB per GPU.

The project code is released under the [MIT License](LICENSE). MNIST is downloaded separately through torchvision.
