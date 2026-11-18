"""Build the measured tables, XeLaTeX report, editable source and evidence archive."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from io import StringIO
from pathlib import Path
from statistics import fmean, median, stdev

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission"
BASE = ROOT / "results/full_run"
MAIN = ROOT / "results/main_repeat"
TRACE = ROOT / "results/trace_repeat"
INITIAL = ROOT / "results/batch_sweep"
REPEAT = ROOT / "results/batch_repeat5"
PLACEMENT = ROOT / "results/gpu_diagnosis"
MAIN_TEX = Path("report/CST435W-Assignment-1.tex")
LABELS = {"cpu": "CPU, 1 thread", "gpu1": "1 H100", "gpu2": "2 H100s"}
BATCHES = (256, 512, 1024, 2048, 4096, 5000)
PROGRAMS = ("knn.py", "classify.py", "benchmark.py", "verify_results.py", "plot_results.py",
            "diagnose_gpu.py", "sweep_batch.py", "analyze_gpu_trace.py", "summarize_batch.py",
            "benchmark_sessions.py", "summarize_sessions.py", "test_correctness.py", "plot_sessions.py")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex_rows(command, rows):
    content = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return "\\newcommand{\\" + command + "}{%\n" + content + "\n}\n"


def write_generated(path, content):
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", suffix=".tmp", encoding="utf-8",
                                     newline="\n", delete=False) as handle:
        handle.write(content)
    Path(handle.name).replace(path)


def batch_samples(directory, repeats, traces):
    samples = defaultdict(dict)
    for row in read_csv(directory / "timed/runs.csv"):
        key, index = int(row["batch"]), int(row["repeat"])
        if index in samples[key] or (row["correct"], row["mismatch_vs_cpu"]) != ("9688", "0"):
            raise ValueError(f"Unexpected timing record in {directory}")
        samples[key][index] = float(row["wall_ms"])
    overlaps = defaultdict(list)
    for row in read_csv(directory / "overlap_runs.csv"):
        overlaps[int(row["batch"])].append(row)
    summaries = {int(row["batch"]): row for row in read_csv(directory / "summary.csv")}
    if set(samples) != set(BATCHES) or set(summaries) != set(BATCHES):
        raise ValueError("Unexpected batch grid")
    for batch in BATCHES:
        group = overlaps[batch]
        if set(samples[batch]) != set(range(1, repeats + 1)) or len(group) != traces:
            raise ValueError(f"Incomplete samples at batch {batch}")
        if sorted(int(r["repeat"]) for r in group) != list(range(1, traces + 1)):
            raise ValueError("Duplicated trace round")
        overlap = 100 * math.fsum(float(r["both_kernel_ms"]) for r in group)
        overlap /= math.fsum(float(r["trace_wall_ms"]) for r in group)
        if not math.isclose(overlap, float(summaries[batch]["both_kernel_span_pct"]), rel_tol=1e-10):
            raise ValueError("Pooled kernel overlap disagrees with the individual traces")
    return samples, summaries


def write_measurements():
    env = json.loads((BASE / "environment.json").read_bytes())
    for directory, hashes in ((BASE, env["csv_sha256"]), (ROOT, env["sources"])):
        for name, expected in hashes.items():
            if sha(directory / name) != expected:
                raise ValueError(f"Archived checksum mismatch for {name}")
    repeated = json.loads((MAIN / "environment.json").read_bytes())
    for name, expected in repeated["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Repeated experiment source changed: {name}")
    for directory in [MAIN, *sorted(MAIN.glob("session_*"))]:
        if not directory.is_dir():
            continue
        metadata = json.loads((directory / "environment.json").read_bytes())
        for name, expected in metadata["csv_sha256"].items():
            if sha(directory / name) != expected:
                raise ValueError(f"New evidence checksum mismatch: {directory / name}")
    validation = json.loads((MAIN / "validation.json").read_bytes())
    if validation["timed_requests"] != 135 or validation["prediction_mismatches"]:
        raise ValueError("Unexpected main experiment validation")
    for name, expected in validation["derived_sha256"].items():
        if sha(MAIN / name) != expected:
            raise ValueError(f"Derived table changed: {name}")
    boundary = json.loads((ROOT / "results/boundary_validation.json").read_bytes())
    if boundary["checks_passed"] != 50 or not boundary["cuda_checked"]:
        raise ValueError("Boundary validation is incomplete")
    for name, expected in boundary["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Boundary-checked source changed: " + name)
    clean = json.loads((ROOT / "results/clean_environment.json").read_bytes())
    if clean.get("status") != "passed" or clean["boundary"]["checks_passed"] != 18:
        raise ValueError("The isolated CPU reproduction has not passed")
    if sha(ROOT / "requirements.txt") != clean["requirements_sha256"]:
        raise ValueError("Dependencies changed after the clean installation check")
    for name, expected in clean["sources"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Clean-environment source changed: " + name)
    timings = defaultdict(dict)
    for row in read_csv(MAIN / "runs.csv"):
        cell = timings[int(row["n_test"]), row["mode"], int(row["session"])]
        run = int(row["run"])
        if run in cell:
            raise ValueError("Duplicate timed request")
        cell[run] = float(row["wall_s"]) * 1000
    details = StringIO()
    writer = csv.writer(details, lineterminator="\n")
    writer.writerow(["test_images", "mode", "hardware", "session", *[f"run_{i}_ms" for i in range(1, 6)],
                     "session_mean_ms", "within_session_sd_ms", "median_ms", "cv_pct"])
    recorded = {(int(r["n_test"]), r["mode"]): r for r in read_csv(MAIN / "summary.csv")}
    table = []
    for n in (100, 1000, 10000):
        cells = [f"{n:,}"]
        for mode, name in LABELS.items():
            session_means = []
            for session in (1, 2, 3):
                cell = timings[n, mode, session]
                if set(cell) != {1, 2, 3, 4, 5}:
                    raise ValueError("Each session requires timed requests 1 through 5")
                values = [cell[i] for i in range(1, 6)]
                mean, sd, med = fmean(values), stdev(values), median(values)
                session_means.append(mean)
                stats = [*values, mean, sd, med, 100 * sd / mean]
                writer.writerow([n, mode, name, session, *[f"{number:.9f}" for number in stats]])
            mean, sd = fmean(session_means), stdev(session_means)
            if not (math.isclose(mean, 1000 * float(recorded[n, mode]["mean_wall_s"]), rel_tol=1e-10)
                    and math.isclose(sd, 1000 * float(recorded[n, mode]["sd_session_mean_wall_s"]), rel_tol=1e-10)):
                raise ValueError("Session summary disagrees with raw times")
            cells.append(f"{mean:,.2f} " + r"\(\pm\) " + f"{sd:.2f}")
        table.append(cells)
    blocks = [tex_rows("MainRows", table)]
    pred = read_csv(BASE / "predictions.csv")
    for mode, name in (("cpu", "sample_output_cpu.txt"), ("gpu2", "sample_output.txt")):
        lines = (ROOT / "results" / name).read_text(encoding="utf-8").splitlines()
        expect = [[r[k] for k in ("test_index", "true_label", mode)] for r in pred
                  if r["n_test"] == "100" and int(r["test_index"]) < 10]
        if ([l.split() for l in lines[1:-1]] != expect
                or not re.fullmatch(mode + r": 99/100 correct; first-call wall [0-9.]+ s", lines[-1])):
            raise ValueError(f"Unexpected console output in {name}")

    placements = {(r["placement"], r["gpu_count"]): r for r in read_csv(PLACEMENT / "timed/summary.csv")}
    traces = {r["placement"]: float(r["both_gpus_kernel_span_pct"])
              for r in read_csv(PLACEMENT / "kernel_summary.csv") if r["gpu_count"] == "2" and r["cuda_device"] == "0"}
    placement_rows = []
    for kind, name in (("per_request", "Upload each request"), ("resident", "Resident")):
        one, two = (float(placements[kind, g]["median_ms"]) for g in ("1", "2"))
        placement_rows.append([name, f"{one:.2f}", f"{two:.2f}", f"{one/two:.3f}", f"{traces[kind]:.2f}" + r"\%"])
    blocks.append(tex_rows("PlacementRows", placement_rows))
    first, old = batch_samples(INITIAL, 18, 12)
    new, recent = batch_samples(REPEAT, 5, 5)
    old_mean = {b: fmean(first[b].values()) for b in BATCHES}
    new_mean = {b: fmean(new[b].values()) for b in BATCHES}
    fastest_old, fastest_new = min(old_mean.values()), min(new_mean.values())
    sweep, rerun, choice = [], [], []
    for b in BATCHES:
        row = old[b]
        sweep.append([f"{b:,}", row["chunks_per_gpu"], f"{median(first[b].values()):.2f}",
                      f"{float(row['wall_p25_ms']):.2f}--{float(row['wall_p75_ms']):.2f}",
                      f"{float(row['both_kernel_span_pct']):.2f}" + r"\%"])
        sd = stdev(new[b].values())
        rerun.append([f"{b:,}", *[f"{new[b][i]:.2f}" for i in range(1, 6)],
                      f"{new_mean[b]:.2f} " + r"\(\pm\) " + f"{sd:.2f}", f"{100*sd/new_mean[b]:.2f}"])
        loss = 100 * (max(old_mean[b] / fastest_old, new_mean[b] / fastest_new) - 1)
        choice.append([f"{b:,}", f"{old_mean[b]:.3f}", f"{new_mean[b]:.3f}",
                       f"{float(row['both_kernel_span_pct']):.2f}" + r"\%",
                       f"{float(recent[b]['both_kernel_span_pct']):.2f}" + r"\%", f"{loss:.2f}" + r"\%"])
    blocks += [tex_rows("SweepRows", sweep), tex_rows("RepeatRows", rerun), tex_rows("ChoiceRows", choice)]
    content = "% Rebuilt from the recorded CSVs by build_report.py.\n" + "\n".join(blocks)
    write_generated(ROOT / "report/measurements.tex", content)
    write_generated(ROOT / "report/table-3-details.csv", details.getvalue())


def compile_latex(engine, temp):
    command = [engine, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
               "-file-line-error", f"-output-directory={temp.as_posix()}", MAIN_TEX.as_posix()]
    env = dict(os.environ, PATH=str(Path(engine).parent) + os.pathsep + os.environ.get("PATH", ""))
    log = temp / MAIN_TEX.with_suffix(".log").name
    for run in range(3):
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        messages = proc.stdout.decode(errors="replace")
        if proc.returncode:
            raise RuntimeError("XeLaTeX failed:\n" + "\n".join(messages.splitlines()[-35:]))
        logs = log.read_text(encoding="utf-8", errors="replace")
        if run >= 1 and not re.search(r"Rerun to get|undefined references|Citation .+ undefined", logs):
            break
    else:
        raise RuntimeError("XeLaTeX cross-references did not settle")
    problems = [s for s in logs.splitlines() if "Overfull " in s or "Missing character:" in s]
    if problems:
        raise ValueError("Layout warnings:\n" + "\n".join(problems))
    pdf = temp / MAIN_TEX.with_suffix(".pdf").name
    if len(PdfReader(pdf).pages) > 10:
        raise ValueError("The cover, body and appendix exceed the ten-page limit")
    return pdf


def build(engine):
    write_measurements()
    OUT.mkdir(exist_ok=True)
    source_name = "CST435W-KNN-source.zip"
    pdf_name = "CST435W-KNN-report.pdf"
    package_name = "CST435W-KNN.zip"
    source_files = {ROOT / name for name in (*PROGRAMS, "build_report.py", "README.md", "WORKFLOW.md", "LICENSE", "requirements.txt")}
    source_files.update((ROOT / "report").glob("*.tex"))
    source_files.update(ROOT / f for f in ("report/usm-mark.pdf", "report/BUILD.md", "report/table-3-details.csv",
                                           "results/figure-1-example.json",
                                           "results/sample_output_cpu.txt", "results/sample_output.txt"))
    for name in ("figure-1-neighbour-vote", "figure-2-query-partition", "performance", "performance-sessions", "confusion",
                 "batch-distributions"):
        source_files.update(ROOT / "figures" / (name + suffix) for suffix in (".pdf", ".svg"))
    source_files.update(directory / "timed/runs.csv" for directory in (INITIAL, REPEAT))
    source_files.update(BASE / f for f in ("runs.csv", "summary.csv", "scaling.csv", "confusion.csv",
                                           "device_profile.csv"))
    source_files.update(MAIN / f for f in ("runs.csv", "summary.csv", "session_summary.csv", "scaling.csv",
                                          "session_scaling.csv", "device_profile.csv"))
    source_files.add(TRACE / "segments.csv")
    evidence = {ROOT / "results/cpu_info.txt", ROOT / "results/sample_output.txt",
                ROOT / "results/sample_output_cpu.txt", ROOT / "figures/figure-4-digit-errors.pdf"}
    evidence.update((ROOT / "results").glob("cpu_100*"))
    evidence.update(p for p in (ROOT / "figures").iterdir() if p.suffix in (".png", ".svg"))
    evidence.update(ROOT / "results" / name for name in ("boundary_validation.json", "verified_environment.json",
                                                         "clean_environment.json"))
    for directory in (BASE, MAIN, TRACE, INITIAL, REPEAT, PLACEMENT):
        evidence.update(p for p in directory.rglob("*") if p.is_file() and
                        p.suffix in (".csv", ".json", ".md", ".nsys-rep", ".sqlite", ".png", ".svg"))
    with tempfile.TemporaryDirectory(dir=OUT) as folder:
        temp = Path(folder)
        compiled = compile_latex(engine, temp)
        source = temp / source_name
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(source_files):
                z.write(file, file.relative_to(ROOT).as_posix())
        pdf = temp / pdf_name
        writer = PdfWriter(clone_from=str(compiled))
        writer.add_attachment(source_name, source.read_bytes())
        with pdf.open("wb") as f:
            writer.write(f)
        reader = PdfReader(pdf)
        if reader.attachments[source_name][0] != source.read_bytes():
            raise ValueError("Embedded source differs from the standalone source archive")
        inputs = {p.relative_to(ROOT).as_posix(): p for p in source_files | evidence}
        inputs["submission/" + source_name] = source
        inputs["submission/" + pdf_name] = pdf
        hashes = {n: sha(f) for n, f in sorted(inputs.items())}
        package = temp / package_name
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in hashes:
                z.write(inputs[name], name)
            z.writestr("MANIFEST.sha256", "".join(f"{h}  {n}\n" for n, h in hashes.items()))
        with zipfile.ZipFile(package) as z:
            if z.testzip() or any(hashlib.sha256(z.read(n)).hexdigest() != h for n, h in hashes.items()):
                raise ValueError("Submission archive failed verification")
        for file in (source, pdf, package):
            file.replace(OUT / file.name)
    print(f"{len(reader.pages)} pages, editable XeLaTeX source, {len(hashes)} verified package members")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xelatex", help="Path to the XeLaTeX executable")
    args = parser.parse_args()
    engine = args.xelatex or shutil.which("xelatex")
    if not engine:
        parser.error("Install XeLaTeX or pass its executable path with --xelatex")
    build(engine)
