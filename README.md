# CST435W Assignment 1

Exact five-nearest-neighbour classification of MNIST on one CPU thread, one H100, and two H100s on the same Linux host. Every mode searches all 60,000 training images. GPU workers replicate the reference, split queries, and return labels in input order.

## Installation

Use Python 3.10. The measurement host used torch 2.3.0+cu121, torchvision 0.18.0+cu121, NumPy 1.26.4, SciPy 1.12.0 and Matplotlib 3.8.4. Install the matching PyTorch pair first, then the shared requirements, including the NVML monitoring binding and pypdf report dependency.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.0.1
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
python -c "from torchvision.datasets import MNIST; MNIST('data', train=True, download=True); MNIST('data', train=False, download=True)"
```

For CPU-only checks, replace `cu121` with `cpu` in the wheel URL. GPU experiments need two visible CUDA devices and a compatible NVIDIA driver. Some Linux installations need their Python `venv` package installed first. Use an isolated environment without system site packages.

The full submission records a clean CPU installation in a fresh Linux container, including dependency checks, 18 correctness/CLI checks, statistics verification, trace parsing and figure regeneration. The 50 CPU/GPU checks ran on the measurement server. These are separate validation scopes; the clean CPU environment does not claim to reproduce H100 timings.

## Run and check the classifier

```bash
python classify.py --mode cpu --limit 100
python classify.py --mode gpu1 --limit 100
python classify.py --mode gpu2 --limit 100
python test_correctness.py --cli
python test_correctness.py --cuda --cli --out results/my_boundary_check.json
```

The CLI prints ten zero-based test indices, true and predicted labels, the correct count and first-call wall time. Raw 0–255 pixels become FP64 vectors. Squared distances and training indices determine the five nearest neighbours; tied votes favour the smaller digit. The supported prefix is 1–10,000 queries.

The independent oracle uses direct int64 pixel differences and stable full sorting. Tests cover distance ties at the fifth neighbour, tied votes, extreme pixels, one-query and odd splits, batch boundaries at 255/256/257/513, actual CLI output and invalid limits. There are 50 checks with `--cuda --cli`, or 18 with `--cli` alone.

## Three-session main experiment

Run from the project root and choose fresh output paths. Measurement commands refuse to overwrite an experiment.

```bash
python benchmark_sessions.py --out results/main_repeat
python summarize_sessions.py results/main_repeat
python test_correctness.py --cuda --cli --out results/boundary_validation.json
```

Three fresh worker processes each load the data once, warm every condition twice, then interleave five requests per size/mode: 135 measured requests in total. Sizes are 100, 1,000 and 10,000 queries; modes are CPU, one GPU and two GPUs. Seed 435 sets the starting orders, which rotate and reverse across rounds. Across all sessions, each mode occupies each order position five times per size.

K=5, FP64, all 60,000 references, and batch 256 are fixed. Both PyTorch CPU thread counts are one. The main thread is pinned to one available CPU for serial work; its original affinity is restored for GPU work. A separate controller samples temperatures, clocks, board load and competing processes every 0.2 seconds.

Timing starts with host tensors ready and ends with ordered labels on the CPU. It includes reference/query uploads, classification, return and joining. Loading, conversion, validation and logging are outside the interval. CUDA-event passes are separate from ordinary timings.

| Evidence | Contents |
|---|---|
| `runs.csv` | All 135 requests, session/order, time boundaries, accuracy and peak allocated tensor memory. |
| `session_summary.csv` | Five-request statistics for each of 27 session/size/mode cells. |
| `summary.csv` | Mean and sample SD of three session means; approximate 95% Student t intervals, 2 degrees of freedom. |
| `session_scaling.csv`, `scaling.csv` | Ratios of means within each session, then summaries across sessions. Two-GPU efficiency is speedup divided by two. |
| `boards.csv`, `monitoring_summary.csv` | Timestamped load, temperatures, clocks and competing-process observations. |
| `session_01/`–`session_03/` | Original timings, ordered predictions, confusion counts, separate event profiles, per-condition records and hashes. |
| `schedule.csv`, `environment.json`, `validation.json` | Planned order, hardware/software, input/source hashes, agreement and statistics checks. |

Process RSS is a session-lifetime high-water mark, not an isolated allocation per condition. Seeing no competing process in the samples does not establish exclusive hardware access.

At 10,000 queries, mean ± SD across sessions was **20.066 ± 0.059 s** on the CPU, **120.36 ± 2.18 ms** on one H100 and **83.24 ± 1.38 ms** on two. Mean two-GPU speedup was **1.446** (approximate 95% interval **1.408–1.484**), with 72.3% efficiency. All labels agreed; accuracy was 96.88%. At 100 queries the speedup interval includes parity. The three sessions share one host and date.

## CUDA timeline and figures

The placement and batch tools use the serial reference in `results/full_run/predictions.csv`. In a fresh checkout, create that historical-protocol baseline first. The complete submission already contains it; preserve the archived results.

```bash
python benchmark.py --out results/full_run
python verify_results.py results/full_run --raw data/MNIST/raw --figures figures
out=results/trace_repeat
mkdir -p "$out"
nsys profile --sample=none --cpuctxsw=none --trace=cuda --output="$out/trace" \
  python diagnose_gpu.py --out "$out/traced" --trace --repeats 3
nsys export --type=sqlite --output="$out/trace.sqlite" "$out/trace.nsys-rep"
python analyze_gpu_trace.py "$out/trace.sqlite" "$out/traced/runs.csv" "$out" --segments
python plot_results.py results/full_run --out figures
python plot_sessions.py results/main_repeat --trace "$out" --out figures
```

Tracing needs NVIDIA Nsight Systems. `plot_sessions.py` produces the primary `performance-sessions` figure as PDF, SVG and PNG. It shows session means, within-session speedups with intervals, and the second separately traced request at batch 256. Trace durations are excluded from ordinary timing statistics. The trace parser checks process/device identity, request boundaries and actual transfer bytes. Simultaneous kernels do not measure SM occupancy or bandwidth saturation.

`plot_results.py` retains the historical performance figure, full confusion matrix, query-partition diagram and batch boxplots. The real-neighbour figure also needs the archived `results/figure-1-example.json` supplied in the complete submission. Figures use the requested five-colour palette at 162 mm width.

## Resident reference and six batch sizes

```bash
python diagnose_gpu.py --out results/placement
out=results/batch_sweep
mkdir -p "$out"
python sweep_batch.py --out "$out/timed" --repeats 18
nsys profile --sample=none --cpuctxsw=none --trace=cuda --output="$out/sweep" \
  python sweep_batch.py --out "$out/traced" --trace --repeats 12
nsys export --type=sqlite --output="$out/sweep.sqlite" "$out/sweep.nsys-rep"
python analyze_gpu_trace.py --by-batch "$out/sweep.sqlite" "$out/traced/runs.csv" "$out"
python summarize_batch.py "$out"
```

For the repeat, choose `out=results/batch_repeat5` and use `--repeats 5` for both ordinary and traced requests. Re-run plotting after both batch directories exist. Boxplots retain every ordinary observation. The initial session recorded another compute process on each GPU; the repeat recorded none. Report their absolute times separately.

For repeated 10,000-query requests with a resident reference, **2,048 per GPU** remains the practical batch choice. Its mean times were 55.97 and 34.45 ms in the two sessions; batch 5,000 gave 57.83 and 33.94 ms. The larger loss relative to either session's fastest mean was 1.52% at 2,048 versus 3.31% at 5,000. Peak tensor allocation was 3.19 versus 4.92 GiB per GPU. Both paired confidence intervals include zero; these observations do not establish an intrinsic speed advantage of 2,048 over 5,000. The main experiment uses batch 256 with per-request training uploads.

## Report and archive

The full submission additionally contains the ten-page PDF, editable XeLaTeX source, all raw evidence and a SHA-256 manifest. The PDF embeds its source ZIP. See `report/BUILD.md` in that package. With XeLaTeX installed, `python build_report.py` regenerates tables, compiles and verifies the package. The public repository contains code and reproduction instructions without the student cover page or MNIST data. Code is released under the MIT License.
