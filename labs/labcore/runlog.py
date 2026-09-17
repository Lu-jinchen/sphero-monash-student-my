"""Run directory structure and log bookkeeping.

Every execution - simulated or real - gets its own directory:

    labs/lab1/runs/<UTC timestamp>__<source>__<experiment>__<model>/
        meta.json     what was run: model, parameters, gains, git hash, result
        data.csv      the rich per-step log written by the runner
        env_log.csv   the CSV written by the repo's own Visualiser logger
        plots/        every figure generated from this run

`index()` scans that tree and is what the dashboard's run browser lists.
`RunWriter` is the streaming writer used while a run is in progress.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from labcore.paths import EXPORTS_DIR, LAB_DIR, RUNS_DIR  # noqa: F401

LAB1_DIR = LAB_DIR

#: columns of data.csv, in order
DATA_COLUMNS: List[str] = [
    "step",
    "t",            # elapsed time on the control clock (integrated `dt`)
    "dt",           # timestep the dynamics actually advanced by:
                    #   sim  -> the environment's fixed dt
                    #   real -> the measured wall-clock duration of the step
    "t_wall",       # wall-clock seconds since the first control step
    "dt_wall",      # measured wall-clock duration of this step
    "speed_cmd",
    "heading_cmd",
    "x",            # logged state (ground truth in sim, odometry on the robot)
    "y",
    "heading",
    "speed",
    "odom_x",
    "odom_y",
    "odom_heading",
    "odom_speed",
    "obs_x",
    "obs_y",
    "obs_heading",
    "obs_speed",
    "setpoint_x",
    "setpoint_y",
    "dist_to_goal",
    "collision",
    # Shadow model: the active dynamics model driven with the same commands and
    # the same measured timestep, starting from the same pose. On a real run
    # this is "where the simulator thought the robot would be", so the gap is
    # visible live instead of only after a post-hoc comparison.
    "pred_x",
    "pred_y",
    "pred_heading",
    "pred_speed",
    "pred_error",
    # Raw robot sensor channels (empty for simulated runs). Saved in full
    # because you cannot re-derive a channel you did not record, and which one
    # turns out to be trustworthy is what calibration is meant to discover.
    "meas_yaw_rad",          # attitude yaw, absolute but noisy
    "meas_yaw_rate_rad_s",   # gyro yaw rate, bias removed - smooth but drifts
    "meas_vx_m_s",           # encoder velocity, right(+)/left(-)
    "meas_vy_m_s",           # encoder velocity, forward(+)/back(-)
    "meas_wheel_left_m_s",   # derived from vy and yaw rate via track_width
    "meas_wheel_right_m_s",
    "meas_distance_m",       # cumulative odometer
    "core_time_s",           # the robot's own clock
    "cmd_heading_rad",       # NB: api.get_heading() is a COMMAND echo
    "cmd_speed_raw",         # NB: api.get_speed() is a COMMAND echo
    "gyro_pitch",
    "gyro_roll",
    "gyro_yaw",
    "att_pitch",
    "att_roll",
    "att_yaw",
    "accel_x",
    "accel_y",
    "accel_z",
]

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str) -> str:
    return _SAFE.sub("-", str(text)).strip("-") or "run"


def git_hash() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(LAB1_DIR), capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def new_run_dir(source: str, experiment: str, model: str,
                tag: str = "") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [stamp, _slug(source), _slug(experiment), _slug(model)]
    if tag:
        parts.append(_slug(tag))
    path = RUNS_DIR / "__".join(parts)
    suffix = 1
    while path.exists():
        suffix += 1
        path = RUNS_DIR / ("__".join(parts) + f"__{suffix}")
    (path / "plots").mkdir(parents=True, exist_ok=True)
    return path


class RunWriter:
    """Streaming writer for one run directory."""

    def __init__(self, run_dir: Path, meta: Optional[Dict[str, Any]] = None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        self.meta: Dict[str, Any] = dict(meta or {})
        self.meta.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
        self.meta.setdefault("git", git_hash())
        self.meta.setdefault("status", "running")
        self._rows = 0
        self._file = open(self.run_dir / "data.csv", "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=DATA_COLUMNS,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()
        self.write_meta()

    # -- paths -------------------------------------------------------------- #

    @property
    def data_path(self) -> Path:
        return self.run_dir / "data.csv"

    @property
    def env_log_path(self) -> Path:
        return self.run_dir / "env_log.csv"

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    # -- writing ------------------------------------------------------------ #

    def append(self, row: Dict[str, Any]) -> None:
        clean = {}
        for key in DATA_COLUMNS:
            value = row.get(key)
            if value is None:
                clean[key] = ""
            elif isinstance(value, bool):
                clean[key] = int(value)
            else:
                clean[key] = value
        self._writer.writerow(clean)
        self._file.flush()
        self._rows += 1

    def write_meta(self, **updates: Any) -> None:
        self.meta.update(updates)
        self.meta["n_steps"] = self._rows
        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as handle:
            json.dump(self.meta, handle, indent=2, default=str)
            handle.write("\n")

    def close(self, status: str = "complete", **updates: Any) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        self.write_meta(status=status,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        **updates)

    def __enter__(self) -> "RunWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(status="error" if exc_type else "complete",
                   error=repr(exc) if exc else None)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def read_meta(run_dir: Path) -> Dict[str, Any]:
    path = Path(run_dir) / "meta.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _parse_iso(text: str) -> Optional[datetime]:
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def local_time(text: str) -> str:
    """Render a stored UTC timestamp in the machine's local time.

    Run directories are named in UTC so they sort correctly and mean the same
    thing on every machine; humans want local time, so both are kept.
    """
    stamp = _parse_iso(text)
    if stamp is None:
        return ""
    return stamp.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def primary_score(meta: Dict[str, Any]) -> Optional[float]:
    """One number per run, lower is better, so runs can be ranked.

    Random-goal benchmarks are scored on mean final error with a penalty for
    every goal not reached; everything else on the final distance to target.
    Open-loop calibration runs have no meaningful score.
    """
    metrics = meta.get("metrics") or {}
    if meta.get("experiment") == "random_goals" and "mean_final_error_m" in metrics:
        miss = 1.0 - float(metrics.get("reach_rate", 0.0))
        return float(metrics["mean_final_error_m"]) + miss
    if "final_dist_to_goal" in metrics and meta.get("experiment") in ("pid", "goto"):
        return float(metrics["final_dist_to_goal"])
    return None


def _entry(path: Path) -> Dict[str, Any]:
    meta = read_meta(path)
    plots = sorted(p.name for p in (path / "plots").glob("*.png")) \
        if (path / "plots").exists() else []
    started = _parse_iso(meta.get("created_utc", ""))
    finished = _parse_iso(meta.get("finished_utc", ""))
    duration = (finished - started).total_seconds() if started and finished else None
    return {
        "id": path.name,
        "path": str(path),
        "source": meta.get("source", "?"),
        "experiment": meta.get("experiment", "?"),
        "model": meta.get("model", "?"),
        "status": meta.get("status", "?"),
        "n_steps": meta.get("n_steps", 0),
        "created_utc": meta.get("created_utc", ""),
        "created_local": local_time(meta.get("created_utc", "")),
        "duration_s": round(duration, 1) if duration is not None else None,
        "metrics": meta.get("metrics", {}),
        "notes": meta.get("notes", ""),
        "plots": plots,
        "has_data": (path / "data.csv").exists(),
        "starred": bool(meta.get("starred", False)),
        # Imported legacy CSVs are real-robot data, but they were NOT recorded
        # by this system: no timestamps, assumed dt. Flagged so they are never
        # mistaken for a run you actually did here.
        "imported": bool(meta.get("imported_from")),
        "imported_from": meta.get("imported_from"),
        "preset": (meta.get("experiment_overrides") or {}).get("preset"),
        "score": primary_score(meta),
    }


SORT_KEYS = ("time", "score", "steps", "duration")


def sort_runs(entries: List[Dict[str, Any]], key: str = "time",
              reverse: bool = False) -> List[Dict[str, Any]]:
    """Sort an index. `time` is newest-first; `score` is best-first."""
    if key == "score":
        scored = [e for e in entries if e.get("score") is not None]
        unscored = [e for e in entries if e.get("score") is None]
        scored.sort(key=lambda e: e["score"], reverse=reverse)
        return scored + unscored
    if key == "steps":
        return sorted(entries, key=lambda e: e.get("n_steps") or 0, reverse=not reverse)
    if key == "duration":
        return sorted(entries, key=lambda e: e.get("duration_s") or 0, reverse=not reverse)
    # Older writers stored basic ISO timestamps (``20260910T114131Z``), while
    # newer ones store extended ISO (``2026-09-11T01:03:23+00:00``). Comparing
    # those strings puts every basic-format run ahead of newer extended-format
    # runs. Parse both accepted forms before sorting so the Runs tab is truly
    # chronological across Lab 1, 2, and 3 archives.
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    return sorted(entries,
                  key=lambda e: _parse_iso(e.get("created_utc") or "") or oldest,
                  reverse=not reverse)


def index(limit: Optional[int] = None, sort: str = "time",
          source: Optional[str] = None, experiment: Optional[str] = None,
          starred_only: bool = False) -> List[Dict[str, Any]]:
    """List runs, newest first by default."""
    if not RUNS_DIR.exists():
        return []
    entries = [_entry(p) for p in RUNS_DIR.iterdir() if p.is_dir()]
    if source:
        entries = [e for e in entries if e["source"] == source]
    if experiment:
        entries = [e for e in entries if e["experiment"] == experiment]
    if starred_only:
        entries = [e for e in entries if e["starred"]]
    entries = sort_runs(entries, sort)
    return entries[:limit] if limit else entries


def set_star(run_id: str, starred: bool = True) -> Dict[str, Any]:
    """Mark a run as one worth keeping."""
    path = resolve(run_id)
    meta = read_meta(path)
    meta["starred"] = bool(starred)
    with open(path / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)
        handle.write("\n")
    return meta


def set_notes(run_id: str, notes: str) -> Dict[str, Any]:
    """Attach a physical observation to an archived run."""
    path = resolve(run_id)
    meta = read_meta(path)
    meta["notes"] = str(notes).strip()
    meta["notes_updated_utc"] = datetime.now(timezone.utc).isoformat()
    with open(path / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, default=str)
        handle.write("\n")
    return meta


def resolve(run_id: str) -> Path:
    """Map a run id back to its directory, rejecting path traversal."""
    safe = _slug(run_id)
    path = (RUNS_DIR / safe).resolve()
    if RUNS_DIR.resolve() not in path.parents:
        raise ValueError(f"Refusing to resolve run id outside runs/: {run_id!r}")
    if not path.exists():
        raise FileNotFoundError(f"No such run: {run_id}")
    return path


def load_run(run_id_or_path) -> "Any":
    """Load a run's data.csv as a DataFrame (pandas imported lazily)."""
    import pandas as pd

    path = Path(run_id_or_path)
    if not path.exists():
        path = resolve(str(run_id_or_path))
    if path.is_dir():
        path = path / "data.csv"
    return pd.read_csv(path)


def import_legacy(csv_path, source: str = "real", experiment: str = "imported",
                  model: str = "unknown", dt: float = 0.1,
                  notes: str = "") -> Path:
    """Ingest an old-format `Visualiser` CSV as a run directory.

    Legacy logs (gt_x/gt_y/heading/speed/...) carry no timestamps, so the
    control period has to be assumed. That matters: speed scale and timestep
    are perfectly confounded in a positions-only log, so a run imported with
    the wrong `dt` will fit a speed gain that is wrong by exactly the same
    factor. Treat anything identified from an imported log as provisional.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    run_dir = new_run_dir(source, experiment, model, tag="imported")
    meta = {
        "source": source,
        "experiment": experiment,
        "model": model,
        "imported_from": str(csv_path),
        "assumed_dt": dt,
        "notes": notes or "Imported legacy log: dt assumed, not measured.",
        "env": {"dt": dt},
    }
    writer = RunWriter(run_dir, meta)

    def col(*names):
        for name in names:
            if name in df.columns:
                return df[name].to_numpy()
        return None

    x = col("x", "gt_x", "odom_x")
    y = col("y", "gt_y", "odom_y")
    heading = col("heading", "odom_heading")
    speed = col("speed", "odom_speed")
    speed_cmd = col("speed_cmd")
    heading_cmd = col("heading_cmd")
    setpoint_x = col("setpoint_x")
    setpoint_y = col("setpoint_y")

    for k in range(len(df)):
        writer.append(
            {
                "step": k,
                "t": round(k * dt, 6),
                "dt": dt,
                "t_wall": round(k * dt, 6),
                "dt_wall": dt,
                "speed_cmd": None if speed_cmd is None else float(speed_cmd[k]),
                "heading_cmd": None if heading_cmd is None else float(heading_cmd[k]),
                "x": None if x is None else float(x[k]),
                "y": None if y is None else float(y[k]),
                "heading": None if heading is None else float(heading[k]),
                "speed": None if speed is None else float(speed[k]),
                "odom_x": None if x is None else float(x[k]),
                "odom_y": None if y is None else float(y[k]),
                "odom_heading": None if heading is None else float(heading[k]),
                "odom_speed": None if speed is None else float(speed[k]),
                "setpoint_x": None if setpoint_x is None else float(setpoint_x[k]),
                "setpoint_y": None if setpoint_y is None else float(setpoint_y[k]),
            }
        )
    writer.close(status="complete")
    return run_dir


def latest(source: Optional[str] = None,
           experiment: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for entry in index():
        if source and entry["source"] != source:
            continue
        if experiment and entry["experiment"] != experiment:
            continue
        if entry["has_data"]:
            return entry
    return None


def find_runs(sources: Iterable[str] = (), experiments: Iterable[str] = (),
              status: str = "complete") -> List[Dict[str, Any]]:
    sources = set(sources)
    experiments = set(experiments)
    out = []
    for entry in index():
        if sources and entry["source"] not in sources:
            continue
        if experiments and entry["experiment"] not in experiments:
            continue
        if status and entry["status"] != status:
            continue
        if not entry["has_data"]:
            continue
        out.append(entry)
    return out
