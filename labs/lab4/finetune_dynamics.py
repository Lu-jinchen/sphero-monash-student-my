"""Fine-tune the learned robot dynamics on the good stretches of every run.

There are few fully clean robot runs, but most failed runs still contain plenty
of good driving before they went wrong. This keeps those stretches and drops
the parts that would teach the model something false, then trains on top of an
existing model with a low learning rate rather than starting again.

A transition is kept only if every row it touches is good:

* no IMU collision flag within +-2 steps. A contact or a jam on a floor seam
  makes the wheels slip, and the position logged for Lab 3/4 runs is the EKF
  estimate built from wheel odometry, so it keeps moving while the ball does
  not;
* not a safety takeover (Lab 4): recoveries are deliberately abnormal driving;
* within 7 cm of that run's own route: off the track is off the maze surface;
* not stuck: commanded to move while the estimate moves less than 4 mm for
  three steps or more, which is slip rather than motion.

Lab 3 runs that failed are left out entirely. They come from before the motor
mapping was fixed, when a command of 0.04 genuinely did not move the ball, and
including them is what taught the previous retrain to under-power every start.

Acceptance, before anything is installed (see ``main``):

1. start-up physics: from rest, a larger command must move the ball further;
2. held-out runs: six-step rollout error no worse than the model it replaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

import dynamics_learning as DL

ROUTE_LIMIT = 0.07
IMU_GUARD = 2
STUCK_EPS = 0.004
STUCK_STEPS = 3


def _f(row, key):
    try:
        v = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _route_dev(p, path):
    best = math.inf
    for a, b in zip(path[:-1], path[1:]):
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def good_mask(run_dir: Path, rows: list) -> np.ndarray:
    """Per-row flag: True where the row is safe to learn from."""

    n = len(rows)
    good = np.ones(n, dtype=bool)
    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}

    hit = np.array([str(r.get("collision", "")).strip() in ("True", "1", "1.0") for r in rows])
    for i in np.where(hit)[0]:
        good[max(0, i - IMU_GUARD):i + IMU_GUARD + 1] = False

    dec_file = run_dir / "policy_decisions.json"
    if dec_file.exists():
        try:
            dec = {int(d["step"]): bool(d.get("safety_takeover")) for d in json.loads(dec_file.read_text())}
            for i, r in enumerate(rows):
                if dec.get(int(float(r.get("step", -1))), False):
                    good[i] = False
        except (ValueError, KeyError):
            pass

    route = meta.get("optimal_path")
    xy = np.array([[_f(r, "odom_x") or 0.0, _f(r, "odom_y") or 0.0] for r in rows])
    if route and len(route) > 1:
        path = [np.asarray(p, dtype=float)[:2] for p in route]
        for i in range(n):
            if _route_dev(xy[i], path) > ROUTE_LIMIT:
                good[i] = False

    cmd = np.array([_f(r, "speed_cmd") or 0.0 for r in rows])
    still = np.r_[False, np.linalg.norm(np.diff(xy, axis=0), axis=1) < STUCK_EPS]
    run_len = 0
    for i in range(n):
        run_len = run_len + 1 if (cmd[i] > 0.015 and still[i]) else 0
        if run_len >= STUCK_STEPS:
            good[max(0, i - run_len + 1):i + 1] = False
    return good


def load_good(run_dir: Path) -> Optional[DL.Run]:
    """Like dynamics_learning.load_run, but only transitions inside good rows."""

    try:
        rows = list(csv.DictReader(open(run_dir / "data.csv", newline="", encoding="utf-8")))
    except OSError:
        return None
    if len(rows) < 6:
        return None
    mask = good_mask(run_dir, rows)
    # Rebuild contiguous segments so no transition straddles a dropped row.
    segments, cur = [], []
    for r, g in zip(rows, mask):
        if g:
            cur.append(r)
        elif cur:
            segments.append(cur); cur = []
    if cur:
        segments.append(cur)
    X, Y = [], []
    for seg in segments:
        if len(seg) < 4:
            continue
        run = _build(seg, run_dir.name)
        if run is not None:
            X.append(run.X); Y.append(run.Y)
    if not X:
        return None
    lab = run_dir.name.split("__")[2]
    return DL.Run(run_dir.name, lab, np.concatenate(X), np.concatenate(Y),
                  np.zeros((1, 2)), np.zeros((1, 2)), np.zeros(1))


def _build(rows, run_id) -> Optional[DL.Run]:
    clean = []
    for r in rows:
        vals = [_f(r, k) for k in ("odom_x", "odom_y", "speed_cmd", "heading_cmd", "dt")]
        if any(v is None for v in vals):
            return None if not clean else _finish(clean, run_id)
        clean.append(vals)
    return _finish(clean, run_id)


def _finish(clean, run_id) -> Optional[DL.Run]:
    a = np.asarray(clean, dtype=float)
    pos, cmd, dts = a[:, 0:2], a[:, 2:4], a[:, 4]
    X, Y = [], []
    for t in range(1, len(a) - 1):
        if not (DL.DT_RANGE[0] <= dts[t] <= DL.DT_RANGE[1] and DL.DT_RANGE[0] <= dts[t + 1] <= DL.DT_RANGE[1]):
            continue
        step_in, step_out = pos[t] - pos[t - 1], pos[t + 1] - pos[t]
        if np.linalg.norm(step_in) > DL.MAX_STEP_M or np.linalg.norm(step_out) > DL.MAX_STEP_M:
            continue
        h = cmd[t, 1]
        vf, vl = DL.to_cmd_frame(step_in / dts[t], h)
        df, dl = DL.to_cmd_frame(step_out, h)
        vf2, vl2 = DL.to_cmd_frame(step_out / dts[t + 1], h)
        turn = DL.wrap(cmd[t - 1, 1] - h)
        X.append([vf, vl, cmd[t, 0], cmd[t - 1, 0] * math.cos(turn), cmd[t - 1, 0] * math.sin(turn), dts[t + 1]])
        Y.append([df, dl, vf2, vl2])
    if not X:
        return None
    return DL.Run(run_id, run_id.split("__")[2] if run_id.count("__") >= 2 else "?",
                  np.asarray(X), np.asarray(Y), pos, cmd, dts)


def eligible(run_dir: Path) -> bool:
    parts = run_dir.name.split("__")
    if len(parts) < 3 or parts[1] != "real":
        return False
    if parts[2] == "lab3":
        meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
        if not bool((meta.get("metrics") or {}).get("pass")):
            return False          # broken-motor era: taught "0.04 does nothing"
    return True


def finetune(base_path: Path, runs: List[DL.Run], val: List[DL.Run], epochs: int,
             lr: float, seed: int):
    torch.manual_seed(seed)
    model = DL.DynamicsModel()
    model.load_state_dict(torch.load(base_path))
    Xtr = torch.tensor(np.concatenate([r.X for r in runs]), dtype=torch.float32)
    Ytr = torch.tensor(np.concatenate([r.Y for r in runs]), dtype=torch.float32)
    Xva = torch.tensor(np.concatenate([r.X for r in val]), dtype=torch.float32)
    Yva = torch.tensor(np.concatenate([r.Y for r in val]), dtype=torch.float32)
    ys = model.y_std        # keep the base model's normalisation: train on top
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state = math.inf, {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = (((model(Xtr[idx]) - Ytr[idx]) / ys) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float((((model(Xva) - Yva) / ys) ** 2).mean())
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


def startup_probe(model) -> List[float]:
    """cm moved after 5 steps from rest, for commands 0.021 / 0.042 / 0.063."""

    out = []
    for cmd in (0.021, 0.042, 0.063):
        vf = vl = 0.0; pf = pl = 0.0; total = 0.0
        for _ in range(5):
            df, dl, vf, vl = model.step(vf, vl, cmd, pf, pl, 0.18)
            pf, pl = cmd, 0.0
            total += df
        out.append(100.0 * total)
    return out


def evaluate(model, runs):
    ro = DL.rollout_errors(model, runs, horizon=6, stride=2)
    return {"rollout6_median_cm": 100 * float(np.median(ro["model"])) if len(ro["model"]) else float("nan"),
            "rollout6_p90_cm": 100 * float(np.percentile(ro["model"], 90)) if len(ro["model"]) else float("nan"),
            "windows": int(len(ro["model"]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", action="append", required=True, help="model(s) to train on top of")
    ap.add_argument("--test", default="", help="comma list of run-id fragments to hold out")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    test_ids = [t for t in args.test.split(",") if t]
    dirs = [d for d in sorted(DL.RUNS_DIR.iterdir()) if d.is_dir() and eligible(d)]
    good, test_good, test_full = [], [], []
    kept = total = 0
    for d in dirs:
        r = load_good(d)
        full = DL.load_run(d, ("odom_x", "odom_y"))
        if full is not None:
            total += len(full.X)
        if r is None:
            continue
        kept += len(r.X)
        if any(t in d.name for t in test_ids):
            test_good.append(r)
            if full is not None:
                test_full.append(full)
        else:
            good.append(r)
    print(f"eligible real runs {len(dirs)} | transitions kept {kept} of {total}"
          f" ({100*kept/max(total,1):.0f}%) | held-out test runs {len(test_full)}", flush=True)

    rng = random.Random(0); order = good[:]; rng.shuffle(order)
    nv = max(1, int(round(0.15 * len(order))))
    report = {}
    for base in args.base:
        bm = DL.DynamicsModel(); bm.load_state_dict(torch.load(base)); bm.eval()
        report[f"base {Path(base).name}"] = {**evaluate(bm, test_full), "startup_cm": startup_probe(bm)}
        ft = finetune(Path(base), order[nv:], order[:nv], args.epochs, args.lr, seed=0)
        name = f"finetuned on top of {Path(base).name}"
        report[name] = {**evaluate(ft, test_full), "startup_cm": startup_probe(ft)}
        torch.save(ft.state_dict(), Path(args.out) / f"ft_{Path(base).stem}.pth")
    for k, v in report.items():
        mono = v["startup_cm"][0] < v["startup_cm"][1] < v["startup_cm"][2]
        print(f"{k:55} 6-step {v['rollout6_median_cm']:.2f} / p90 {v['rollout6_p90_cm']:.2f} cm"
              f" | start-up 0.021/0.042/0.063 -> {'/'.join(f'{x:.2f}' for x in v['startup_cm'])} cm"
              f" {'OK' if mono else 'NOT monotonic'}", flush=True)
    (Path(args.out) / "finetune_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
