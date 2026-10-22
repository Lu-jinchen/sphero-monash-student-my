"""Learn the robot's dynamics from every logged real run.

Why this exists
---------------
Every hardware failure in the first Lab 4 runs was a dynamics mismatch, not a
planning one: the ball runs on about 12 cm after the drive is cut, a command of
0.042 produces 0.130 m/s, and constants tuned in the simulator were wrong on
the floor. Those were patched by hand from two data points. This module learns
the same relationships from all the real transitions on disk instead.

Frame
-----
Everything is expressed in the frame of the COMMANDED heading: "forward" is
along the direction the ball was told to go and "lateral" is perpendicular to
it. That makes the model independent of where the ball is in the maze and of
whether a run logged in the maze frame or the robot's own odometry frame,
because the command and the odometry always share a frame within one run. It
also means a turn shows up for free: when the commanded heading swings, the
ball's existing velocity appears as a large lateral component.

Model
-----
inputs   v_fwd, v_lat     current velocity in the commanded-heading frame (m/s)
         speed_cmd        the command the policy actually emits
         prev_fwd,        the PREVIOUS command, resolved into the current
           prev_lat       command's frame: its speed times cos/sin of the turn
         dt               control period (s)

The previous command is there because the robot has actuation latency. On the
step right after a command from rest the ball moves almost nothing, 0.06 cm in
the passing runs, and the motion arrives on the following steps. A model given
only (velocity, current command) sees "at rest, commanded" both times and so
predicts no motion on every step of a rollout, and a controller using it never
leaves the start. Resolving the previous command into the current frame also
carries the turn, so a pivot looks different from a straight start.
outputs  d_fwd, d_lat   displacement over the step (m)
         v_fwd', v_lat' velocity at the end of the step (m/s)

Rolled forward with zero command it predicts the coast; rolled forward with a
candidate command it predicts where that command puts the ball, which is what
a controller needs to choose commands against.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE.parent / "lab1" / "runs"
#: One model per plant. The simulator does not coast and has no gap between
#: the commanded and the achieved speed, so a model of the robot would
#: mispredict it, and the reverse. The controller loads the one for the plant
#: it is driving.
def model_path(source: str) -> Path:
    return HERE / f"dynamics_model_{source}.pth"


def report_path(source: str) -> Path:
    return HERE / f"dynamics_report_{source}.json"


#: Where the plant's actual position is logged. On the robot that is the
#: locator, archived as odom_*; the x/y columns there are the simulator shadow.
#: In the simulator the plant is the true state, x/y, which is what collides.
POSITION_COLUMNS = {"real": ("odom_x", "odom_y"), "sim": ("x", "y")}

INPUT_NAMES = ("v_fwd", "v_lat", "speed_cmd", "prev_fwd", "prev_lat", "dt")
OUTPUT_NAMES = ("d_fwd", "d_lat", "v_fwd_next", "v_lat_next")

#: Locator glitches: a step implying more than this is a resync, not motion.
#: The fastest honest step in the logs is about 0.10 m.
MAX_STEP_M = 0.15
#: Control periods outside this range are stalls or batched BLE samples.
DT_RANGE = (0.05, 1.2)


def wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def to_cmd_frame(vec: np.ndarray, heading: float) -> Tuple[float, float]:
    """World vector -> (forward, lateral) in a frame whose forward is ``heading``.

    Heading 0 is world +y and +pi/2 is world +x.
    """

    fwd = np.array([math.sin(heading), math.cos(heading)])
    lat = np.array([math.cos(heading), -math.sin(heading)])
    return float(vec @ fwd), float(vec @ lat)


def from_cmd_frame(f: float, l: float, heading: float) -> np.ndarray:
    fwd = np.array([math.sin(heading), math.cos(heading)])
    lat = np.array([math.cos(heading), -math.sin(heading)])
    return f * fwd + l * lat


@dataclass
class Run:
    run_id: str
    lab: str
    X: np.ndarray          # (n, 4) inputs
    Y: np.ndarray          # (n, 4) targets
    pos: np.ndarray        # (n+1, 2) odometry, for rollouts
    cmd: np.ndarray        # (n, 2) speed_cmd, heading_cmd
    dt: np.ndarray         # (n,)


def _f(row: dict, key: str) -> Optional[float]:
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def load_run(run_dir: Path, pos_cols: Tuple[str, str] = ("odom_x", "odom_y")) -> Optional[Run]:
    """Turn one archived real run into model transitions."""

    path = run_dir / "data.csv"
    if not path.exists():
        return None
    try:
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    except (OSError, ValueError):
        return None
    clean = []
    for row in rows:
        vals = [_f(row, k) for k in (pos_cols[0], pos_cols[1], "speed_cmd", "heading_cmd", "dt")]
        if any(v is None for v in vals):
            continue
        clean.append(vals)
    if len(clean) < 6:
        return None
    a = np.asarray(clean, dtype=float)
    pos, cmd, dts = a[:, 0:2], a[:, 2:4], a[:, 4]

    X, Y, keep_pos, keep_cmd, keep_dt = [], [], [], [], []
    # Needs t-1 for the incoming velocity and t+1 for the outcome.
    for t in range(1, len(a) - 1):
        dt_in, dt_out = dts[t], dts[t + 1]
        if not (DT_RANGE[0] <= dt_in <= DT_RANGE[1] and DT_RANGE[0] <= dt_out <= DT_RANGE[1]):
            continue
        step_in = pos[t] - pos[t - 1]
        step_out = pos[t + 1] - pos[t]
        if np.linalg.norm(step_in) > MAX_STEP_M or np.linalg.norm(step_out) > MAX_STEP_M:
            continue
        heading = cmd[t, 1]
        v_in = step_in / dt_in
        v_out = step_out / dt_out
        vf, vl = to_cmd_frame(v_in, heading)
        df, dl = to_cmd_frame(step_out, heading)
        vf2, vl2 = to_cmd_frame(v_out, heading)
        turn = wrap(cmd[t - 1, 1] - heading)
        pf, pl = cmd[t - 1, 0] * math.cos(turn), cmd[t - 1, 0] * math.sin(turn)
        X.append([vf, vl, cmd[t, 0], pf, pl, dt_out])
        Y.append([df, dl, vf2, vl2])
    if len(X) < 4:
        return None
    lab = run_dir.name.split("__")[2] if run_dir.name.count("__") >= 2 else "?"
    return Run(run_dir.name, lab, np.asarray(X), np.asarray(Y), pos, cmd, dts)


def discover(runs_dir: Path = RUNS_DIR, sources: Sequence[str] = ("real",),
             labs: Optional[Sequence[str]] = None) -> List[Run]:
    runs = []
    for d in sorted(Path(runs_dir).iterdir()):
        if not d.is_dir():
            continue
        parts = d.name.split("__")
        if len(parts) < 3 or parts[1] not in sources:
            continue
        if labs and parts[2] not in labs:
            continue
        run = load_run(d, POSITION_COLUMNS[parts[1]])
        if run is not None:
            runs.append(run)
    return runs


# --------------------------------------------------------------------- model

if torch is not None:

    class DynamicsModel(torch.nn.Module):
        """Small MLP over normalised inputs, predicting normalised outputs."""

        def __init__(self, hidden: int = 64):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(len(INPUT_NAMES), hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, len(OUTPUT_NAMES)),
            )
            for name in ("x_mean", "x_std", "y_mean", "y_std"):
                n = len(INPUT_NAMES) if name.startswith("x") else len(OUTPUT_NAMES)
                self.register_buffer(name, torch.zeros(n) if "mean" in name else torch.ones(n))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            z = (x - self.x_mean) / self.x_std
            return self.net(z) * self.y_std + self.y_mean

        def step(self, vf, vl, speed_cmd, prev_fwd, prev_lat, dt) -> np.ndarray:
            with torch.no_grad():
                out = self(torch.tensor([[vf, vl, speed_cmd, prev_fwd, prev_lat, dt]],
                                        dtype=torch.float32))
            return out[0].numpy().astype(float)


def train(train_runs: List[Run], val_runs: List[Run], epochs: int = 400,
          lr: float = 3e-3, seed: int = 0) -> Tuple["DynamicsModel", dict]:
    torch.manual_seed(seed)
    Xtr = np.concatenate([r.X for r in train_runs]); Ytr = np.concatenate([r.Y for r in train_runs])
    Xva = np.concatenate([r.X for r in val_runs]);   Yva = np.concatenate([r.Y for r in val_runs])
    model = DynamicsModel()
    model.x_mean.copy_(torch.tensor(Xtr.mean(0), dtype=torch.float32))
    model.x_std.copy_(torch.tensor(Xtr.std(0) + 1e-6, dtype=torch.float32))
    model.y_mean.copy_(torch.tensor(Ytr.mean(0), dtype=torch.float32))
    model.y_std.copy_(torch.tensor(Ytr.std(0) + 1e-6, dtype=torch.float32))
    xt = torch.tensor(Xtr, dtype=torch.float32); yt = torch.tensor(Ytr, dtype=torch.float32)
    xv = torch.tensor(Xva, dtype=torch.float32); yv = torch.tensor(Yva, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ys = model.y_std
    best, best_state, hist = math.inf, None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = (((model(xt[idx]) - yt[idx]) / ys) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float((((model(xv) - yv) / ys) ** 2).mean())
        hist.append(vl)
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best, "train_n": len(Xtr), "val_n": len(Xva)}


# ---------------------------------------------------------------- evaluation

def coast_events(run: Run, min_cmd: float = 0.02, max_steps: int = 14):
    """Moments the drive was cut from moving, and how far the ball then ran."""

    events = []
    c, p, dts = run.cmd[:, 0], run.pos, run.dt
    for t in range(2, len(c) - 2):
        if not (c[t - 1] >= min_cmd and abs(c[t]) < 1e-6):
            continue
        end = t
        while end + 1 < len(c) and abs(c[end + 1]) < 1e-6 and end - t < max_steps:
            end += 1
        if end - t < 3:
            continue
        steps = [p[k + 1] - p[k] for k in range(t, end)]
        if any(np.linalg.norm(s) > MAX_STEP_M for s in steps):
            continue
        dt_in = dts[t]
        if not (DT_RANGE[0] <= dt_in <= DT_RANGE[1]):
            continue
        step_in = p[t] - p[t - 1]
        if np.linalg.norm(step_in) > MAX_STEP_M:
            continue
        heading = run.cmd[t, 1]
        v0 = step_in / dt_in
        actual = float(np.linalg.norm(p[end] - p[t]))
        turn = wrap(run.cmd[t - 1, 1] - heading)
        events.append({"v0": v0, "heading": heading, "dts": dts[t + 1:end + 1],
                       "prev": (c[t - 1] * math.cos(turn), c[t - 1] * math.sin(turn)),
                       "actual": actual, "speed": float(np.linalg.norm(v0))})
    return events


def predict_coast(model, ev) -> float:
    vf, vl = to_cmd_frame(ev["v0"], ev["heading"])
    pf, pl = ev["prev"]
    disp = np.zeros(2)
    for dt in ev["dts"]:
        df, dl, vf, vl = model.step(vf, vl, 0.0, pf, pl, float(dt))
        pf, pl = 0.0, 0.0
        disp += np.array([df, dl])
    return float(np.linalg.norm(disp))


def rollout_errors(model, runs, horizon=6, stride=3):
    """Open-loop: replay logged commands through the model from a true state
    and compare where it ends up with where the ball actually went.

    This is the test that matters for a controller that plans several steps
    ahead, because one-step error compounds. It is also the test the first
    version of this model would have failed: rolled out from rest, it never
    moved.
    """

    out = {"model": [], "cmd_x_dt": [], "const_vel": []}
    for r in runs:
        p, c, dts = r.pos, r.cmd, r.dt
        n = len(c)
        for t0 in range(1, n - horizon - 1, stride):
            seg = range(t0, t0 + horizon)
            if not (DT_RANGE[0] <= dts[t0] <= DT_RANGE[1]):
                continue
            if any(not (DT_RANGE[0] <= dts[k + 1] <= DT_RANGE[1]) for k in seg):
                continue
            if any(np.linalg.norm(p[k + 1] - p[k]) > MAX_STEP_M
                   for k in range(t0 - 1, t0 + horizon)):
                continue
            v0 = (p[t0] - p[t0 - 1]) / dts[t0]
            v = v0.copy()
            pos, pos_cv, pos_cmd = p[t0].copy(), p[t0].copy(), p[t0].copy()
            prev_s, prev_h = c[t0 - 1, 0], c[t0 - 1, 1]
            for k in seg:
                h, sp, dt = c[k, 1], c[k, 0], dts[k + 1]
                vf, vl = to_cmd_frame(v, h)
                turn = wrap(prev_h - h)
                df, dl, vf2, vl2 = model.step(vf, vl, sp, prev_s * math.cos(turn),
                                              prev_s * math.sin(turn), dt)
                pos = pos + from_cmd_frame(df, dl, h)
                v = from_cmd_frame(vf2, vl2, h)
                pos_cv = pos_cv + v0 * dt
                pos_cmd = pos_cmd + from_cmd_frame(sp * dt, 0.0, h)
                prev_s, prev_h = sp, h
            truth = p[t0 + horizon]
            out["model"].append(np.linalg.norm(pos - truth))
            out["const_vel"].append(np.linalg.norm(pos_cv - truth))
            out["cmd_x_dt"].append(np.linalg.norm(pos_cmd - truth))
    return {k: np.asarray(v) for k, v in out.items()}


def hand_fit(speed: float) -> float:
    """The constant fitted by hand from two points: d = 0.03 + 0.70 v."""

    return 0.03 + 0.70 * speed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--source", choices=("real", "sim"), default="real")
    ap.add_argument("--labs", default="",
                    help="comma list of experiments to train on, e.g. lab3,lab4. "
                         "The simulator's settings changed between labs, so the "
                         "sim model should learn the environment it will drive.")
    args = ap.parse_args(argv)

    labs = [x.strip() for x in args.labs.split(",") if x.strip()] or None
    runs = discover(sources=(args.source,), labs=labs)
    rng = random.Random(args.seed)
    order = runs[:]; rng.shuffle(order)
    n_val = max(1, int(round(len(order) * args.val_fraction)))
    val, tr = order[:n_val], order[n_val:]
    print(f"{args.source} runs {len(runs)} | train {len(tr)} runs, {sum(len(r.X) for r in tr)} transitions"
          f" | held out {len(val)} runs, {sum(len(r.X) for r in val)} transitions")

    model, info = train(tr, val, epochs=args.epochs, seed=args.seed)
    print(f"best held-out loss {info['best_val_loss']:.4f}")

    # one-step displacement error on held-out runs, against two baselines
    Xv = np.concatenate([r.X for r in val]); Yv = np.concatenate([r.Y for r in val])
    with torch.no_grad():
        P = model(torch.tensor(Xv, dtype=torch.float32)).numpy()
    err_model = np.linalg.norm(P[:, 0:2] - Yv[:, 0:2], axis=1)
    # simulator assumption: the ball moves at the command, along the command
    err_cmd = np.linalg.norm(np.stack([Xv[:, 2] * Xv[:, 5], np.zeros(len(Xv))], 1) - Yv[:, 0:2], axis=1)
    # constant velocity: it keeps doing what it was doing
    err_cv = np.linalg.norm(Xv[:, 0:2] * Xv[:, 5:6] - Yv[:, 0:2], axis=1)
    print("\none-step displacement error on HELD-OUT runs (mm):")
    for name, e in (("sim assumption (cmd x dt)", err_cmd),
                    ("constant velocity", err_cv),
                    ("learned model", err_model)):
        print(f"  {name:28} median {1000*np.median(e):6.1f}   p90 {1000*np.percentile(e,90):6.1f}")

    # the thing that actually broke the corners: the coast
    evs = [e for r in val for e in coast_events(r)]
    report = {"source": args.source, "labs": labs or "all", "runs": len(runs), "train_runs": len(tr), "val_runs": len(val), **info,
              "one_step_mm": {"sim_assumption": float(1000*np.median(err_cmd)),
                              "constant_velocity": float(1000*np.median(err_cv)),
                              "learned": float(1000*np.median(err_model))}}
    if evs:
        a = np.array([e["actual"] for e in evs])
        pm = np.array([predict_coast(model, e) for e in evs])
        ph = np.array([hand_fit(e["speed"]) for e in evs])
        print(f"\nCOAST after the drive is cut - {len(evs)} events on held-out runs:")
        print(f"  actual distance run on      mean {100*a.mean():5.1f} cm")
        print(f"  hand fit 0.03 + 0.70 v      mean abs error {100*np.abs(ph-a).mean():5.1f} cm")
        print(f"  learned model               mean abs error {100*np.abs(pm-a).mean():5.1f} cm")
        report["coast"] = {"events": len(evs), "actual_mean_cm": float(100*a.mean()),
                           "hand_fit_mae_cm": float(100*np.abs(ph-a).mean()),
                           "learned_mae_cm": float(100*np.abs(pm-a).mean())}
    else:
        print("\nno coast events in the held-out runs")

    ro = rollout_errors(model, val)
    if len(ro["model"]):
        print(f"\nOPEN-LOOP ROLLOUT, logged commands replayed 6 steps ahead - "
              f"{len(ro['model'])} windows on held-out runs (cm at the end):")
        for name, key in (("sim assumption (cmd x dt)", "cmd_x_dt"),
                          ("constant velocity", "const_vel"), ("learned model", "model")):
            e = ro[key]
            print(f"  {name:28} median {100*np.median(e):5.2f}   p90 {100*np.percentile(e,90):5.2f}")
        report["rollout_6_cm"] = {k: float(100 * np.median(v)) for k, v in ro.items()}

    torch.save(model.state_dict(), model_path(args.source))
    report_path(args.source).write_text(json.dumps(report, indent=2))
    print(f"\nsaved {model_path(args.source).name} and {report_path(args.source).name}")
    return model, report


if __name__ == "__main__":
    main()
