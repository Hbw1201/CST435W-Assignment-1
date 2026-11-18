"""Verify interleaved runs and summarise variation between independent processes."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from scipy.stats import t


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = float(t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return dict(mean=mean, sd=sd, min=min(values), max=max(values), ci95_low=mean - half, ci95_high=mean + half)


def summarise(folder):
    metadata = json.loads((folder / "environment.json").read_text())
    root = Path(__file__).resolve().parent
    for name, expected in metadata["sources"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    for name, expected in metadata["csv_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected, name
    sessions = metadata["sessions"]
    repeats = metadata["measured_per_cell"]
    sizes = metadata["sizes"]
    modes = ("cpu", "gpu1", "gpu2")
    schedule = read_csv(folder / "schedule.csv")
    all_runs, session_rows, profiles = [], [], []
    expected_predictions = None
    timings = {}
    for session in range(1, sessions + 1):
        directory = folder / f"session_{session:02d}"
        env = json.loads((directory / "environment.json").read_text())
        for name, expected in env["csv_sha256"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected, name
        rows = read_csv(directory / "runs.csv")
        assert len(rows) == len(sizes) * len(modes) * repeats
        chronological = sorted(rows, key=lambda r: int(r["unix_start_ns"]))
        planned = [r for r in schedule if int(r["session"]) == session]
        keys = ("session", "run", "n_test", "mode", "size_position", "mode_position")
        assert [[r[k] for k in keys] for r in chronological] == [[r[k] for k in keys] for r in planned]
        for previous, current in zip(chronological, chronological[1:]):
            assert int(previous["unix_end_ns"]) < int(current["unix_start_ns"])
        predictions = read_csv(directory / "predictions.csv")
        if expected_predictions is None:
            expected_predictions = predictions
        assert predictions == expected_predictions, "Predictions differ across sessions"
        for row in predictions:
            assert row["cpu"] == row["gpu1"] == row["gpu2"]
        summaries = {(int(r["n_test"]), r["mode"]): r for r in read_csv(directory / "summary.csv")}
        for size in sizes:
            part = [r for r in predictions if int(r["n_test"]) == size]
            assert [int(r["test_index"]) for r in part] == list(range(size))
            correct = sum(r["cpu"] == r["true_label"] for r in part)
            for mode in modes:
                group = [r for r in rows if int(r["n_test"]) == size and r["mode"] == mode]
                assert sorted(int(r["run"]) for r in group) == list(range(1, repeats + 1))
                values = [float(r["wall_s"]) for r in group]
                assert min(values) > 0
                assert all(int(r["correct"]) == correct and int(r["mismatch_vs_cpu"]) == 0 for r in group)
                recorded = summaries[size, mode]
                for key, value in (("mean_wall_s", statistics.mean(values)),
                                   ("sample_std_wall_s", statistics.stdev(values)),
                                   ("median_wall_s", statistics.median(values))):
                    assert math.isclose(float(recorded[key]), value, rel_tol=1e-12), key
                timings[session, size, mode] = statistics.mean(values)
                session_rows.append(dict(recorded, session=session))
        all_runs.extend(rows)
        profiles.extend(dict(r, session=session) for r in read_csv(directory / "device_profile.csv"))

    summary = []
    for size in sizes:
        for mode in modes:
            values = [timings[s, size, mode] for s in range(1, sessions + 1)]
            moments = stats(values)
            example = next(r for r in session_rows if int(r["n_test"]) == size and r["mode"] == mode)
            summary.append(dict(n_test=size, mode=mode, sessions=sessions, repeats_per_session=repeats,
                                timed_requests=sessions * repeats, mean_wall_s=moments["mean"],
                                sd_session_mean_wall_s=moments["sd"],
                                min_session_mean_wall_s=moments["min"], max_session_mean_wall_s=moments["max"],
                                ci95_low_wall_s=moments["ci95_low"], ci95_high_wall_s=moments["ci95_high"],
                                correct=int(example["correct"]), accuracy_pct=float(example["accuracy_pct"])))

    session_scaling = []
    for session in range(1, sessions + 1):
        for size in sizes:
            cpu, one, two = (timings[session, size, mode] for mode in modes)
            session_scaling.append(dict(session=session, n_test=size, cpu_mean_s=cpu,
                                        gpu1_mean_s=one, gpu2_mean_s=two, cpu_to_gpu1=cpu / one,
                                        cpu_to_gpu2=cpu / two, gpu2_speedup=one / two,
                                        gpu2_efficiency_pct=50 * one / two))
    scaling = []
    for size in sizes:
        row = dict(n_test=size, sessions=sessions)
        for metric in ("cpu_to_gpu1", "cpu_to_gpu2", "gpu2_speedup", "gpu2_efficiency_pct"):
            values = [r[metric] for r in session_scaling if r["n_test"] == size]
            row.update({metric + "_" + key: value for key, value in stats(values).items()})
        scaling.append(row)

    boards = read_csv(folder / "boards.csv")
    monitoring = []
    for session in range(1, sessions + 1):
        for device in (0, 1):
            group = [r for r in boards if r["session"] == str(session) and r["gpu"] == str(device)]
            assert group
            row = dict(session=session, gpu=device, samples=len(group),
                       samples_with_other_compute=sum(int(r["other_compute_processes"]) > 0 for r in group),
                       max_other_compute=max(int(r["other_compute_processes"]) for r in group))
            for metric in ("temperature_c", "sm_clock_mhz", "memory_clock_mhz", "power_w", "gpu_busy_pct"):
                values = [float(r[metric]) for r in group]
                row[metric + "_min"] = min(values)
                row[metric + "_max"] = max(values)
            monitoring.append(row)
    positions = Counter((r["n_test"], r["mode"], r["mode_position"]) for r in all_runs)
    audit = dict(timed_requests=len(all_runs), sessions=sessions, repeats_per_cell=repeats,
                 schedule_verified=True, prediction_mismatches=0,
                 per_mode_position_counts=sorted(set(positions.values())),
                 aggregation="Mean and sample SD of session means; speedup is calculated within each session first",
                 ci_method="Two-sided Student t interval across sessions; df = sessions - 1",
                 process_ids=[json.loads((folder / f"session_{s:02d}/environment.json").read_text())["pid"]
                              for s in range(1, sessions + 1)],
                 board_samples=len(boards),
                 other_process_observations=sum(r["samples_with_other_compute"] for r in monitoring))
    assert len(set(audit["process_ids"])) == sessions
    for name, rows in (("runs.csv", all_runs), ("session_summary.csv", session_rows),
                       ("summary.csv", summary), ("session_scaling.csv", session_scaling),
                       ("scaling.csv", scaling), ("device_profile.csv", profiles),
                       ("monitoring_summary.csv", monitoring)):
        write_csv(folder / name, rows)
    audit["derived_sha256"] = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                               for name in ("runs.csv", "session_summary.csv", "summary.csv", "session_scaling.csv",
                                            "scaling.csv", "device_profile.csv", "monitoring_summary.csv")}
    (folder / "validation.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    summarise(args.directory)
