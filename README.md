# CST435W Assignment 1

This project compares exact five-nearest-neighbour classification of MNIST on one CPU thread, one H100 and two H100s on the same Linux host. Every mode searches all 60,000 training images. Each GPU worker holds a copy of the reference and processes its share of the queries. The host returns labels in input order.

## Check the recorded results without the server

The [experiment archive](results/README.md) contains raw timings, ordered predictions, confusion counts, monitoring samples, per-session metadata and CUDA event data. You can check the reported results locally without an H100 or an SSH connection:

```bash
python verify_archive.py
# Install requirements.txt if the plotting/statistics packages are not available.
python reproduce_results.py --out reproduced
```

`verify_archive.py` needs only standard Python. The second command rebuilds statistics and all five report figures in a new folder and checks them against the archive. Recorded files stay unchanged. It reads the SQLite traces directly; Nsight Systems is needed only to capture new GPU activity. The [data map and publication notes](results/README.md) explain the files and metadata redactions.

To prepare a local Linux CPU runtime, run `docker build -f Dockerfile.cpu -t cst435w-cpu .`. Once the image is installed, Docker can check results and rebuild figures with networking disabled. The PDF and archived results remain usable when the original server is offline. New H100 measurements require suitable GPUs.

## Installation

Use Python 3.10. The measurement host used torch 2.3.0+cu121, torchvision 0.18.0+cu121, NumPy 1.26.4, SciPy 1.12.0 and Matplotlib 3.8.4. Install the matching PyTorch pair first, followed by the shared requirements. These include the NVML binding for monitoring and pypdf for the report.

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

The full submission records an installation check in a fresh Linux CPU container. Dependency checks, 18 correctness/CLI checks, statistics verification, trace parsing and figure regeneration passed there. The 50 CPU/GPU checks ran on the measurement server. The container check covers installation and correctness; reproducing H100 timings requires GPU hardware.

## Run and check the classifier

```bash
python classify.py --mode cpu --limit 100
python classify.py --mode gpu1 --limit 100
python classify.py --mode gpu2 --limit 100
python test_correctness.py --cli
python test_correctness.py --cuda --cli --out results/my_boundary_check.json
```

The CLI prints ten zero-based test indices with true and predicted labels, followed by the number of correct predictions and first-call wall time. Raw pixel values from 0 to 255 become FP64 vectors. Squared distances and training indices determine the five nearest neighbours; tied votes favour the smaller digit. The classifier accepts the first 1 to 10,000 queries.

An independent oracle computes direct int64 pixel differences and uses stable full sorting. Tests cover distance ties at the fifth neighbour, tied votes, extreme pixels, one-query and odd splits, batch boundaries at 255/256/257/513, CLI output and invalid limits. Running with `--cuda --cli` performs 50 checks; `--cli` alone performs 18.

## Three-session main experiment

Run from the project root with new output paths. The measurement commands refuse to overwrite existing experiments. The `new_*` paths below are examples: choose another unused name if you have already run them. Archived results remain the inputs for rebuilding the submitted report.

```bash
python benchmark_sessions.py --out results/new_main_repeat
python summarize_sessions.py results/new_main_repeat
python test_correctness.py --cuda --cli --out results/new_boundary_validation.json
```

Each of three fresh worker processes loads the data once, warms every condition twice, then interleaves five requests per size/mode, for 135 measured requests in total. The experiment uses 100, 1,000 and 10,000 queries on the CPU, one GPU and two GPUs. Seed 435 sets the starting orders, which rotate and reverse across rounds. Across all sessions, each mode occupies each order position five times per size.

All conditions use K=5, FP64, all 60,000 references and batch 256. Both PyTorch CPU thread counts are one. The main thread is pinned to one available CPU during serial work and regains its original affinity for GPU work. A separate controller samples temperatures, clocks, board load and competing processes every 0.2 seconds.

Timing starts with tensors ready on the host and ends with ordered labels on the CPU. It includes reference/query uploads, classification, return and joining. Loading, conversion, validation and logging occur outside this interval. CUDA-event passes are collected separately from ordinary timings.

| Evidence | Contents |
|---|---|
| `runs.csv` | All 135 requests, session/order, time boundaries, accuracy and peak allocated tensor memory. |
| `session_summary.csv` | Five-request statistics for each of 27 session/size/mode cells. |
| `summary.csv` | Mean and sample SD of three session means; approximate 95% Student t intervals, 2 degrees of freedom. |
| `session_scaling.csv`, `scaling.csv` | Ratios of means within each session, then summaries across sessions. Two-GPU efficiency is speedup divided by two. |
| `boards.csv`, `monitoring_summary.csv` | Timestamped load, temperatures, clocks and competing-process observations. |
| `session_01/` through `session_03/` | Original timings, ordered predictions, confusion counts, separate event profiles, per-condition records and hashes. |
| `schedule.csv`, `environment.json`, `validation.json` | Planned order, hardware/software, input/source hashes, agreement and statistics checks. |

Process RSS records the high-water mark over the session's lifetime, so it cannot isolate allocations for each condition. An absence of competing processes in the samples does not establish exclusive hardware access.

At 10,000 queries, mean ± SD across sessions was 20.066 ± 0.059 s on the CPU, 120.36 ± 2.18 ms on one H100 and 83.24 ± 1.38 ms on two. Mean two-GPU speedup was 1.446 (approximate 95% interval 1.408 to 1.484), with 72.3% efficiency. All labels agreed, and accuracy was 96.88%. At 100 queries, the speedup interval includes parity. All three sessions ran on one host and date.

For 10,000 queries, the separate CUDA-event passes give the following mean stage times. Each row averages one profiled request per session across the three sessions in `results/main_repeat/device_profile.csv`.

| Configuration and device | Upload (ms) | Compute (ms) | Return (ms) |
|---|---:|---:|---:|
| One H100, GPU 0 | 52.69 | 68.82 | 0.073 |
| Two H100s, GPU 0 | 45.11 | 34.69 | 0.060 |
| Two H100s, GPU 1 | 45.13 | 34.63 | 0.087 |

Each GPU still uploads the full reference, so transfer time remains substantial as computation halves. These profiles are separate from ordinary wall timings and the Nsight trace. Do not subtract their sums from the main request times to estimate host overhead, or add overlapping device intervals. The scaling result covers one and two GPUs on this host.

## CUDA timeline and figures

The placement and batch tools use the serial reference in `results/full_run/predictions.csv`, included in the repository. Use `reproduce_results.py` to parse and plot the archived traces. To capture a new trace on a CUDA host, run the commands below; they save it separately from the recorded trace:

```bash
python verify_results.py results/full_run
out=results/new_trace
mkdir -p "$out"
nsys profile --sample=none --cpuctxsw=none --trace=cuda --output="$out/trace" \
  python diagnose_gpu.py --out "$out/traced" --trace --repeats 3
nsys export --type=sqlite --output="$out/trace.sqlite" "$out/trace.nsys-rep"
python analyze_gpu_trace.py "$out/trace.sqlite" "$out/traced/runs.csv" "$out" --segments
python plot_sessions.py results/main_repeat --trace "$out" --out figures/new_trace
```

Collecting new traces requires NVIDIA Nsight Systems. `plot_sessions.py` exports the primary `performance-sessions` figure as PDF, SVG and PNG. It shows session means, within-session speedups with intervals, and the second separately traced request at batch 256. Ordinary timing statistics exclude trace durations. The parser checks process/device identity, request boundaries and transfer bytes. Simultaneous kernels do not measure SM occupancy or bandwidth saturation.

`plot_results.py` produces the historical performance figure, full confusion matrix, query-partition diagram and batch boxplots. The real-neighbour figure uses `results/figure-1-example.json`, which is also included. Report figures are 162 mm wide and use the five-colour palette.

## Resident reference and six batch sizes

```bash
python diagnose_gpu.py --out results/new_placement
out=results/new_batch_sweep
mkdir -p "$out"
python sweep_batch.py --out "$out/timed" --repeats 18
nsys profile --sample=none --cpuctxsw=none --trace=cuda --output="$out/sweep" \
  python sweep_batch.py --out "$out/traced" --trace --repeats 12
nsys export --type=sqlite --output="$out/sweep.sqlite" "$out/sweep.nsys-rep"
python analyze_gpu_trace.py --by-batch "$out/sweep.sqlite" "$out/traced/runs.csv" "$out"
python summarize_batch.py "$out"
```

For a new repeat, choose `out=results/new_batch_repeat5` and use `--repeats 5` for both ordinary and traced requests. Every ordinary observation appears in the archived boxplots. The initial session recorded another compute process on each GPU; the repeat recorded none. Keep the sessions' absolute times separate.

Table 4's 99.93 ms upload-per-request baseline is a median of eight requests in the separate placement experiment. Table 3's 83.24 ms is the mean of three session means in the main experiment. Their sessions and summaries differ; calculate the placement benefit within Table 4, not by combining these baselines.

For repeated 10,000-query requests with a resident reference, we recommend a batch of 2,048 per GPU. Its mean times were 55.97 and 34.45 ms in the two sessions; batch 5,000 took 57.83 and 33.94 ms. The larger percentage loss relative to either session's fastest mean was 1.52% at 2,048 and 3.31% at 5,000. Peak tensor allocation was 3.19 and 4.92 GiB per GPU, respectively. Both paired confidence intervals include zero, so these observations do not establish an intrinsic speed advantage of 2,048 over 5,000. The main experiment uses batch 256 and uploads training data on each request.

## Report and archive

The complete local submission contains the ten-page PDF, editable XeLaTeX source, original native traces and a SHA-256 manifest. The source ZIP is also embedded in the PDF. See `report/BUILD.md` in the package for build instructions. With the complete package and XeLaTeX installed, `python build_report.py` regenerates tables, compiles the report and verifies the submission. The public repository provides code and matching result data; it excludes the student cover page and full MNIST dataset. Code is released under the MIT License.
