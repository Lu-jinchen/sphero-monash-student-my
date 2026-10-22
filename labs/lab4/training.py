"""Leakage-safe behaviour-cloning dataset and trainer for Lab 4."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from Policy import FEATURE_NAMES, TARGET_NAMES, Policy, require_torch
except ImportError:
    from .Policy import FEATURE_NAMES, TARGET_NAMES, Policy, require_torch


HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE.parent / "lab1" / "runs"
WEIGHTS_PATH = HERE / "weight.pth"
METADATA_PATH = HERE / "training_metadata.json"
HISTORY_PATH = HERE / "training_history.json"

# A usable demonstration is a whole mission: it starts on the start plate and
# finishes on the goal plate. Fragments that began mid-maze reach the goal and
# are logged as passing, but they cross shared plates on a different heading to
# the real route. A memoryless, position-conditioned policy cannot tell the two
# apart, so it averages the conflicting heading labels and rounds the corner.
START_WORLD = (-0.5, -0.5)
GOAL_WORLD = (0.5, 0.5)
PLATE_TOLERANCE = 0.125


@dataclass
class Rollout:
    run_id: str
    source: str
    observations: np.ndarray
    actions: np.ndarray

    @property
    def samples(self) -> int:
        return int(len(self.observations))


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def discover_rollouts(
    *,
    sources: Iterable[str] = ("real",),
    require_pass: bool = True,
    collision_free_rows: bool = True,
    exclude_non_goal_stops: bool = True,
    require_full_route: bool = True,
    runs_dir: Path = RUNS_DIR,
) -> List[Rollout]:
    """Load usable Lab 3 expert rollouts from the shared run archive."""

    allowed_sources = {str(source) for source in sources}
    rollouts: List[Rollout] = []
    if not Path(runs_dir).exists():
        return rollouts

    for run_dir in sorted(Path(runs_dir).iterdir()):
        if not run_dir.is_dir() or "__lab3__" not in run_dir.name:
            continue
        metadata = _load_json(run_dir / "meta.json")
        source = str(metadata.get("source", ""))
        if allowed_sources and source not in allowed_sources:
            continue
        if metadata.get("status") != "complete":
            continue
        if require_pass and not bool((metadata.get("metrics") or {}).get("pass")):
            continue

        observations: List[List[float]] = []
        actions: List[List[float]] = []
        first_pos: Optional[tuple] = None
        last_pos: Optional[tuple] = None
        try:
            with open(run_dir / "data.csv", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    if collision_free_rows and str(row.get("collision", "0")) not in ("", "0", "0.0"):
                        continue
                    values = [
                        row.get("odom_x"), row.get("odom_y"),
                        row.get("heading"), row.get("speed"),
                        row.get("speed_cmd"), row.get("heading_cmd"),
                    ]
                    if any(value in (None, "") for value in values):
                        continue
                    numeric = [float(value) for value in values]
                    if not np.all(np.isfinite(numeric)):
                        continue
                    # Endpoints come from every parseable row, before the
                    # stop-row filter below drops the stationary start sample.
                    if first_pos is None:
                        first_pos = (numeric[0], numeric[1])
                    last_pos = (numeric[0], numeric[1])
                    # A memoryless policy cannot observe the follower's hidden
                    # "pivot now" latch. Zero-speed rows at intermediate
                    # corners therefore give the same visible state two
                    # conflicting labels: stop and drive. The deployed safety
                    # layer owns corner pivots, so retain stops only in the
                    # goal neighbourhood where stopping is the intended task.
                    if (exclude_non_goal_stops and abs(numeric[4]) < 0.02
                            and math.hypot(numeric[0] - 0.5, numeric[1] - 0.5) > 0.12):
                        continue
                    observations.append(numeric[:4])
                    actions.append(numeric[4:])
        except (OSError, ValueError):
            continue

        if require_full_route:
            if first_pos is None or last_pos is None:
                continue
            if math.dist(first_pos, START_WORLD) > PLATE_TOLERANCE:
                continue
            if math.dist(last_pos, GOAL_WORLD) > PLATE_TOLERANCE:
                continue

        if len(observations) >= 8:
            rollouts.append(Rollout(
                run_id=run_dir.name,
                source=source,
                observations=np.asarray(observations, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32),
            ))
    return rollouts


def dataset_summary(runs_dir: Path = RUNS_DIR) -> dict:
    """Return archive counts for the dashboard without importing PyTorch."""

    summary: Dict[str, dict] = {}
    for source in ("real", "sim"):
        all_runs = discover_rollouts(
            sources=(source,), require_pass=False, collision_free_rows=False,
            runs_dir=runs_dir,
        )
        passing = discover_rollouts(
            sources=(source,), require_pass=True, collision_free_rows=True,
            runs_dir=runs_dir,
        )
        summary[source] = {
            "complete_runs": len(all_runs),
            "passing_runs": len(passing),
            "usable_samples": sum(run.samples for run in passing),
        }
    summary["default"] = "real passing runs"
    return summary


def split_by_rollout(
    rollouts: Sequence[Rollout], val_fraction: float = 0.2, seed: int = 7
) -> tuple[List[Rollout], List[Rollout]]:
    if len(rollouts) < 2:
        raise ValueError("Training needs at least two complete expert runs")
    shuffled = list(rollouts)
    random.Random(int(seed)).shuffle(shuffled)
    n_val = min(max(1, int(round(len(shuffled) * float(val_fraction)))), len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def _stack(rollouts: Sequence[Rollout]) -> tuple[np.ndarray, np.ndarray]:
    observations = np.concatenate([run.observations for run in rollouts], axis=0)
    actions = np.concatenate([run.actions for run in rollouts], axis=0)
    return observations.astype(np.float32), actions.astype(np.float32)


def _loss(model, observations, actions, heading_weight: float = 1.0):
    predicted = model(observations)
    speed_scale = torch.clamp(model.max_speed_cmd, min=1e-6)
    speed_loss = torch.mean(((predicted[:, 0] - actions[:, 0]) / speed_scale) ** 2)
    heading_loss = torch.mean(
        (torch.sin(predicted[:, 1]) - torch.sin(actions[:, 1])) ** 2
        + (torch.cos(predicted[:, 1]) - torch.cos(actions[:, 1])) ** 2
    )
    return speed_loss + float(heading_weight) * heading_loss, speed_loss, heading_loss


def _diagnostics(model, observations, actions) -> dict:
    with torch.no_grad():
        predicted = model(observations)
        angle_error = torch.atan2(
            torch.sin(predicted[:, 1] - actions[:, 1]),
            torch.cos(predicted[:, 1] - actions[:, 1]),
        )
        return {
            "speed_mae": float(torch.mean(torch.abs(predicted[:, 0] - actions[:, 0]))),
            "heading_mae_deg": float(torch.mean(torch.abs(angle_error)) * 180.0 / math.pi),
        }


def train(
    *,
    sources: Iterable[str] = ("real",),
    epochs: int = 300,
    learning_rate: float = 2e-3,
    batch_size: int = 256,
    hidden: int = 64,
    val_fraction: float = 0.2,
    seed: int = 7,
    patience: int = 60,
    heading_weight: float = 1.0,
    weights_path: Path = WEIGHTS_PATH,
    progress: Optional[Callable[[dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> dict:
    """Train, validate, checkpoint, and write reproducibility metadata."""

    require_torch()
    if int(hidden) != 64:
        raise ValueError(
            "hidden must remain 64 so weight.pth loads into the submitted Policy()"
        )
    selected_sources = tuple(dict.fromkeys(str(value) for value in sources))
    rollouts = discover_rollouts(sources=selected_sources)
    train_runs, val_runs = split_by_rollout(rollouts, val_fraction, seed)
    train_obs_np, train_actions_np = _stack(train_runs)
    val_obs_np, val_actions_np = _stack(val_runs)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    model = Policy(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    train_obs = torch.tensor(train_obs_np)
    train_actions = torch.tensor(train_actions_np)
    val_obs = torch.tensor(val_obs_np)
    val_actions = torch.tensor(val_actions_np)
    generator = torch.Generator().manual_seed(seed)

    history: List[dict] = []
    best_val = math.inf
    best_epoch = -1
    best_state = None
    stale_epochs = 0
    status = "complete"

    for epoch in range(1, int(epochs) + 1):
        if should_stop is not None and should_stop():
            status = "stopped"
            break
        model.train()
        order = torch.randperm(len(train_obs), generator=generator)
        total = 0.0
        count = 0
        for start in range(0, len(order), int(batch_size)):
            indices = order[start:start + int(batch_size)]
            optimizer.zero_grad()
            loss, _speed, _heading = _loss(
                model, train_obs[indices], train_actions[indices], heading_weight
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)

        model.eval()
        with torch.no_grad():
            validation, val_speed, val_heading = _loss(
                model, val_obs, val_actions, heading_weight
            )
        row = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "val_loss": float(validation),
            "val_speed_loss": float(val_speed),
            "val_heading_loss": float(val_heading),
        }
        history.append(row)
        if row["val_loss"] < best_val - 1e-6:
            best_val = row["val_loss"]
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress is not None:
            progress({**row, "epochs": int(epochs), "best_epoch": best_epoch})
        if stale_epochs >= int(patience):
            status = "early_stopped"
            break

    if best_state is None:
        raise RuntimeError("Training stopped before a checkpoint was produced")
    model.load_state_dict(best_state)
    model.eval()

    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    diagnostics = _diagnostics(model, val_obs, val_actions)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "approach": "behaviour cloning from Lab 3 A* + PID expert",
        "weights": str(weights_path),
        "sources": list(selected_sources),
        "require_pass": True,
        "collision_free_rows": True,
        "excluded_non_goal_zero_speed_rows": True,
        "split": "whole rollout (no adjacent-timestep leakage)",
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "epochs_completed": len(history),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "train_runs": [run.run_id for run in train_runs],
        "validation_runs": [run.run_id for run in val_runs],
        "train_samples": int(len(train_obs)),
        "validation_samples": int(len(val_obs)),
        "independent_variables": list(FEATURE_NAMES),
        "dependent_variables": list(TARGET_NAMES),
        "loss": "MSE(speed_cmd/max_speed_cmd) + heading_weight * MSE(direction vector)",
        "hyperparameters": {
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "hidden": int(hidden),
            "val_fraction": float(val_fraction),
            "patience": int(patience),
            "heading_weight": float(heading_weight),
        },
        **diagnostics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    HISTORY_PATH.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {"model": model, "metadata": metadata, "history": history}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the Lab 4 behaviour-cloning policy")
    parser.add_argument("--sources", choices=("real", "sim", "both"), default="real")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--output", type=Path, default=WEIGHTS_PATH)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sources = ("real", "sim") if args.sources == "both" else (args.sources,)

    def show(row):
        if row["epoch"] == 1 or row["epoch"] % 10 == 0:
            print(
                f"epoch {row['epoch']:4d}/{row['epochs']}  "
                f"train={row['train_loss']:.5f}  val={row['val_loss']:.5f}"
            )

    result = train(
        sources=sources,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden=args.hidden,
        seed=args.seed,
        patience=args.patience,
        weights_path=args.output,
        progress=show,
    )
    print(json.dumps(result["metadata"], indent=2))
    return result


if __name__ == "__main__":
    main()
