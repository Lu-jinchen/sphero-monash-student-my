"""Validate and plot ``35301856_lab3.csv`` before Moodle submission.

The published brief gives four thresholds but does not state how trajectory
distance is reduced to one number. This checker uses the stricter maximum
point-to-optimal-polyline distance: if it passes, mean/RMS/percentile versions
also pass.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

import numpy as np

try:
    from Planner import Planner
    import lab3
except ImportError:
    from .Planner import Planner
    from . import lab3


HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "exports" / f"{lab3.STUDENT_ID}_lab3.csv"
REQUIRED_COLUMNS = ["sim_x", "sim_y", "real_x", "real_y"]


def read_submission(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    values = np.array(
        [[float(row[column]) for column in REQUIRED_COLUMNS] for row in rows],
        dtype=float,
    ) if rows else np.empty((0, 4), dtype=float)
    return columns, values


def nearest_path_distances(points, path):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    distances = []
    for point in points:
        best = math.inf
        for start, end in zip(path, path[1:]):
            edge = end - start
            denominator = float(edge @ edge)
            fraction = 0.0 if denominator < 1e-12 else float(
                np.clip((point - start) @ edge / denominator, 0.0, 1.0)
            )
            best = min(best, float(np.linalg.norm(point - (start + fraction * edge))))
        distances.append(best)
    return np.asarray(distances)


def evaluate(values):
    planner = Planner(lab3.MAP)
    state = np.array([lab3.START[0], lab3.START[1], 0.0, 0.0])
    optimal = planner.plan(state, lab3.GOAL)
    sim = values[:, :2]
    real = values[:, 2:]
    sim_distance = nearest_path_distances(sim, optimal)
    real_distance = nearest_path_distances(real, optimal)

    checks = {
        "final_distance_real": {
            "value": float(np.linalg.norm(real[-1] - lab3.GOAL)),
            "threshold": lab3.GOAL_TOLERANCE,
        },
        "distance_to_optimal_path_real": {
            "value": float(np.max(real_distance)),
            "threshold": lab3.PATH_TOLERANCE,
        },
        "final_distance_sim": {
            "value": float(np.linalg.norm(sim[-1] - lab3.GOAL)),
            "threshold": lab3.GOAL_TOLERANCE,
        },
        "distance_to_optimal_path_sim": {
            "value": float(np.max(sim_distance)),
            "threshold": lab3.PATH_TOLERANCE,
        },
    }
    for item in checks.values():
        item["pass"] = bool(item["value"] <= item["threshold"])
    return {
        "pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
        "mean_deviation_real": float(np.mean(real_distance)),
        "mean_deviation_sim": float(np.mean(sim_distance)),
        "rms_deviation_real": float(np.sqrt(np.mean(real_distance ** 2))),
        "rms_deviation_sim": float(np.sqrt(np.mean(sim_distance ** 2))),
        "optimal_path": [[float(v) for v in point] for point in optimal],
    }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]).resolve() if argv else DEFAULT
    if not path.exists():
        print(f"not found: {path}")
        return 2

    print(f"checking {path}\n")
    try:
        columns, values = read_submission(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] could not read numeric CSV: {type(exc).__name__}: {exc}")
        return 1

    structural = [
        (columns == REQUIRED_COLUMNS,
         f"exact columns {REQUIRED_COLUMNS}; found {columns}"),
        (len(values) >= 2, f"contains the full run ({len(values)} rows)"),
        (bool(np.all(np.isfinite(values))), "all four columns are finite"),
    ]
    failed = False
    for passed, detail in structural:
        print(f"  [{'PASS' if passed else 'FAIL'}] {detail}")
        failed = failed or not passed
    if failed:
        return 1

    report = evaluate(values)
    report.update({"steps": len(values), "collisions": "unknown", "replans": "unknown"})
    print()
    print(lab3.format_report(report))
    print(f"  mean path deviation (real/sim): "
          f"{report['mean_deviation_real']:.4f} / {report['mean_deviation_sim']:.4f} m")
    print(f"  RMS path deviation  (real/sim): "
          f"{report['rms_deviation_real']:.4f} / {report['rms_deviation_sim']:.4f} m")

    records = [
        dict(zip(REQUIRED_COLUMNS, (float(value) for value in row)))
        for row in values
    ]
    result = {
        "records": records,
        "optimal_path": report["optimal_path"],
        "report": report,
    }
    plot_path = lab3.plot_run(result, path.with_suffix(".png"))
    print(f"\n  plot: {plot_path}")
    print("  path-distance pass uses the maximum sample deviation (conservative).")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
