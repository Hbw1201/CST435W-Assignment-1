# Recorded experiment data

These files support the submitted CST435W report. They are included in Git, so checking results and rebuilding figures does not require access to the measurement server. Run commands from the repository root.

```bash
python verify_archive.py
python reproduce_results.py --out reproduced
```

The first command uses only the Python standard library. The second needs NumPy, SciPy and Matplotlib from `requirements.txt`; it recomputes the statistics, checks all four CUDA trace databases, and exports the five report figures as PDF, SVG and PNG. It writes to a fresh output folder and leaves the recorded evidence unchanged. Neither command downloads MNIST, connects to SSH, needs an H100, or invokes Nsight Systems.

The current batch summarizer adds five descriptive-statistic columns that were absent from the initial sweep's archived summary: mean time, time SD, time CV, mean per-request overlap and overlap SD. Reproduction checks every original column and allows exactly these additions for that one historical table.

## Where each result comes from

| Evidence | Report use and sample size |
|---|---|
| [main_repeat/runs.csv](main_repeat/runs.csv) | Primary comparison: 135 ordinary requests, three fresh process sessions, 100/1,000/10,000 queries, three modes, five repeats per cell. |
| [main_repeat/summary.csv](main_repeat/summary.csv), [scaling.csv](main_repeat/scaling.csv) | Table 3 and Figure 3a–b. Means and SDs across three session means; speedups are calculated within sessions first. Approximate t intervals have two degrees of freedom. |
| [main_repeat/session_summary.csv](main_repeat/session_summary.csv), [session_scaling.csv](main_repeat/session_scaling.csv) | Each session's statistics before aggregation. Per-image predictions, confusion counts and per-condition JSON are in `session_01/`–`session_03/`. |
| [main_repeat/boards.csv](main_repeat/boards.csv), [schedule.csv](main_repeat/schedule.csv) | The 0.2-second board samples and the planned interleaved measurement order. |
| [trace_repeat/segments.csv](trace_repeat/segments.csv), [trace.sqlite](trace_repeat/trace.sqlite) | Figure 3c uses the second separately traced per-request-upload request, batch 256. Twelve requests cover four placement/device conditions with three traces each. |
| [full_run/](full_run/) | Historical fixed-order 45-request experiment, its serial reference predictions and the full confusion matrix used in Figure 4. Preserved separately from the newer main experiment. |
| [gpu_diagnosis/](gpu_diagnosis/) | Table 4: eight ordinary requests per placement/device condition, plus three separate traces per condition; 32 ordinary requests and 12 traces. |
| [batch_sweep/timed/runs.csv](batch_sweep/timed/runs.csv) | Initial resident sweep: six sizes × 18 ordinary requests. Twelve separate traces per size produce the overlap estimates. Table 5, Table 7 and Figure 5a. |
| [batch_repeat5/timed/runs.csv](batch_repeat5/timed/runs.csv) | Repeat resident sweep: six sizes × five ordinary requests, plus five separate traces per size. Every ordinary time appears in Table 6; Table 7 and Figure 5b compare sessions. |
| [figure-1-example.json](figure-1-example.json) | The real MNIST query and five neighbour images in Figure 1, including pixel hashes, labels, distances and row IDs. |
| [boundary_validation.json](boundary_validation.json), [clean_environment.json](clean_environment.json) | The recorded 50 CPU/GPU boundary and CLI checks, and the separate clean Linux CPU installation with 18 checks and reproduction commands. |
| `cpu_100*`, [cpu_info.txt](cpu_info.txt), `sample_output*.txt` | The early single-thread pilot, CPU topology and the archived CLI excerpts. |

Ordinary timing and profiled timing are separate observations. Kernel overlap is the intersection of the two GPUs' merged kernel intervals divided by the traced request duration; it is not an SM occupancy or memory-bandwidth measurement. The two resident batch sessions are kept separate because their observed competing-process counts differ.

## Integrity and publication changes

[SHA256SUMS](SHA256SUMS) pins the public evidence and the reproduction sources. All published CSV files are byte-for-byte copies of the measured or derived files in the submission. Experiment metadata also records the source and input hashes used during measurement.

The public copies replace the private project path and host name with portable labels. In the four SQLite exports, machine/environment, network-interface, captured-file and export metadata rows were removed. Private path labels in the string table were replaced, and the databases were compacted. **Every row and value in all 20 CUPTI activity tables is unchanged.** Re-parsing these databases must reproduce the recorded kernel, copy and overlap statistics. [publication.json](publication.json) records the changed-file hashes, retained event-table row counts and hashes, and omitted native-capture hashes.

The four original `.nsys-rep` captures and unmodified SQLite files remain in the complete local submission and offline backup. They are not needed to recompute the public results. Full MNIST input files are downloaded separately through torchvision; no measurement-server URL is used. Six archived MNIST example images are included only as inputs to Figure 1.

The PDF pins the measured implementation at code commit `cb9da8a`. This data release adds the matching evidence and offline verification tools without changing those measured implementations.
