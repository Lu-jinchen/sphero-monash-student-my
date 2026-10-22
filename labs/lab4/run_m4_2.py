"""Drive the ball with the exact M4-2 submission, the way the marker runs it.

The submission (``submit_m4_2/37482025_m4_2.py`` plus ``weight.pth``) is loaded
standalone from its own folder and driven by the template's control loop::

    policy = Policy()
    obs, _ = env.reset()
    for step in range(500):
        action = policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
        obs, _, terminated, truncated, info = env.step(action)

Nothing in the submission is edited or imported by name.

Two settings make the robot look like the template's ``Robot`` rather than our
calibrated one:

* ``--power template`` (default): motor power is ``cmd / 0.15 * 255``, as in
  the template ``Robot.step``, with ``--max-raw`` as a safety cap.
  ``--power calibrated`` uses the Lab 1 calibration our own runs use.
* ``--noise 0.05`` (default on the robot): the template ``Robot`` adds this
  much Gaussian noise to every observation; ours does not, so it is added
  here. The simulator adds its own, so the default there is 0.

Usage::

    python labs/lab4/run_m4_2.py --toy NAME             # real robot
    python labs/lab4/run_m4_2.py --sim                  # official maze simulator
    python labs/lab4/run_m4_2.py --toy NAME --power calibrated --noise 0
    python labs/lab4/run_m4_2.py --toy NAME --version mpc   # the MPC submission

Place the ball on the start plate facing up the first straight (maze +y); it
is aimed along that straight before the run, as in our Lab 3/4 runs. Ctrl+C
stops the ball. Each run writes a CSV under ``labs/lab4/exports/m4_2_tests/``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SUBMISSIONS = {"bc": HERE / "submit_m4_2" / "37482025_m4_2.py",
               "mpc": HERE / "submit_m4_2_mpc" / "37482025_m4_2.py",
               "exact": HERE / "submit_m4_2_exact" / "37482025_m4_2.py"}
SUBMISSION = SUBMISSIONS["bc"]
OUT = HERE / "exports" / "m4_2_tests"
START = np.array([-0.5, -0.5])
GOAL = np.array([0.5, 0.5])
PLATE_HALF = 0.125 - 0.0365     # ball centre limit for the whole ball on a plate

for extra in (REPO / "src", REPO / "labs" / "lab2", REPO / "labs" / "lab3"):
    sys.path.insert(0, str(extra))


def load_policy_class():
    if not SUBMISSION.exists():
        raise SystemExit(f"submission not found: {SUBMISSION}")
    spec = importlib.util.spec_from_file_location("m4_2_submission", SUBMISSION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Policy"):
        raise SystemExit("submission defines no Policy class")
    return module.Policy


@contextlib.contextmanager
def robot_env(args):
    """Connect, aim along the first straight, configure power; own log file."""

    with contextlib.redirect_stdout(io.StringIO()):
        import lab3
    lab2 = lab3.lab2_stack
    from sphero_unsw.sphero_edu import SpheroEduAPI

    toy = lab2.connect_named(args.toy)
    print("Connected.")
    with SpheroEduAPI(toy) as api:
        # Command 0 drives up the first straight: start (-0.5,-0.5) -> (-0.5,0).
        lab2.aim_drive_unit(api, np.array([0.0, 0.5]))
        # Same robot set-up as the dashboard's Lab 4 runs (lab3.make_real_env
        # with the dashboard's dt 0.68); the template uses 0.1.
        env = lab3.make_real_env(api, dt=0.1 if args.power == "template" else 0.68)
        lab2.apply_robot_config(env, auto_aim=False, power_scale=lab2.LAB2_PHYSICAL_MAX_SPEED)
        if args.power == "template":
            # Template Robot.step: raw = cmd / vel_limit * 255 with vel_limit 0.15.
            env.raw_min_move = 0
            env.physical_max_speed = 0.15
        env.max_raw_command = int(args.max_raw)
        print(f"  power: {args.power} | raw cap {env.max_raw_command} | "
              f"cmd 0.026 -> raw {env.speed_cmd_to_raw(0.026)}")
        OUT.mkdir(parents=True, exist_ok=True)
        try:
            yield env
        finally:
            env.emergency_stop()
            env.close()


@contextlib.contextmanager
def sim_env(args):
    with contextlib.redirect_stdout(io.StringIO()):
        import lab3
    env = lab3.make_sim_env(render=not args.no_render)
    try:
        yield env
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=("bc", "mpc", "exact"), default="bc",
                        help="which submission folder: submit_m4_2 (bc) or submit_m4_2_mpc (mpc)")
    parser.add_argument("--sim", action="store_true", help="official maze simulator instead of the robot")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--toy", default=None, help="robot name (as typed on the dashboard)")
    parser.add_argument("--power", choices=("template", "calibrated"), default="template")
    parser.add_argument("--max-raw", type=int, default=80, help="safety cap on motor power, 0-255")
    parser.add_argument("--noise", type=float, default=None,
                        help="observation noise added (m, rad, m/s); default 0.05 on the robot, "
                             "0 in the simulator, which adds its own")
    parser.add_argument("--steps", type=int, default=500, help="the template's MAX_STEPS")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.noise is None:
        args.noise = 0.0 if args.sim else 0.05

    import torch
    global SUBMISSION
    SUBMISSION = SUBMISSIONS[args.version]
    Policy = load_policy_class()
    rng = np.random.default_rng(args.seed)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    mode = ("sim" if args.sim else f"robot-{args.power}") + f"-{args.version}"
    out_path = OUT / f"{stamp}_{mode}.csv"
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  policy : {SUBMISSION.relative_to(REPO)}")
    print(f"  mode   : {mode} | noise {args.noise} | steps {args.steps}")
    print("=" * 70)

    rows = []
    reached_at = None
    t0 = None
    with (sim_env(args) if args.sim else robot_env(args)) as env:
        policy = Policy()                          # exactly as the marker
        print(f"  weights: {Path(policy.weights_loaded).name if policy.weights_loaded else 'NOT FOUND'}")
        if args.sim:
            with contextlib.redirect_stdout(io.StringIO()):
                import lab3
                obs = lab3.place_simulator_at_start(env, seed=args.seed)
            obs = np.asarray(obs, dtype=np.float32)
        else:
            obs, _ = env.reset()
        try:
            for step in range(1, args.steps + 1):
                seen = np.asarray(obs, dtype=np.float32).copy()
                if args.sim:
                    seen[:2] -= START             # the template sim starts at (0, 0)
                if args.noise > 0:
                    seen[:4] += rng.normal(0.0, args.noise, 4).astype(np.float32)
                action = policy(torch.tensor(seen, dtype=torch.float32)).detach().numpy()
                obs, _, terminated, truncated, info = env.step(action)
                t0 = t0 or time.perf_counter()

                if hasattr(policy, "estimate"):
                    est = policy.estimate
                else:
                    est = policy.ekf.x if policy.ekf.x is not None else np.full(4, np.nan)
                if args.sim:
                    truth = np.asarray(info["state_true"][:2], dtype=float)
                else:
                    truth = np.asarray(obs[:2], dtype=float) + START   # locator, start-relative
                on_goal = bool(np.all(np.abs(truth - GOAL) <= PLATE_HALF))
                if on_goal and reached_at is None:
                    reached_at = (step, time.perf_counter() - t0)
                rows.append({
                    "step": step, "t": round(time.perf_counter() - t0, 3),
                    "obs_x": float(obs[0]), "obs_y": float(obs[1]),
                    "pos_x": float(truth[0]), "pos_y": float(truth[1]),
                    "est_x": float(est[0]), "est_y": float(est[1]),
                    "speed_cmd": float(action[0]), "heading_cmd": float(action[1]),
                    "raw": info.get("speed_raw", ""), "collision": int(bool(info.get("collision"))),
                    "gain": round(getattr(getattr(policy, "speed", None), "gain", 0.0), 3),
                    "floor": round(getattr(getattr(policy, "speed", None), "floor", 0.0), 4),
                    "speed_echo": int(getattr(getattr(policy, "speed", None), "speed_is_echo", False)),
                    "waypoint": getattr(policy, "waypoint_index", getattr(getattr(policy, "mpc", None), "index", 0)),
                    "at_goal_hold": int(getattr(policy, "_at_goal", False)), "on_goal_plate": int(on_goal),
                })
                if step % 10 == 0 or step == 1:
                    raw = info.get("speed_raw", "-")
                    print(f"  {step:4d}  pos=({truth[0]:+.3f},{truth[1]:+.3f}) "
                          f"est=({est[0]:+.3f},{est[1]:+.3f}) cmd={action[0]:.3f} raw={raw} "
                          f"wp={rows[-1]['waypoint']} {'HOLD' if rows[-1]['at_goal_hold'] else ''}")
                if truncated or terminated:
                    break
        except KeyboardInterrupt:
            print("  stopped by user")
        finally:
            if rows:
                with open(out_path, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    final = rows[-1] if rows else None
    print("=" * 70)
    if final:
        end = np.array([final["pos_x"], final["pos_y"]])
        print(f"  steps run          : {final['step']}")
        print(f"  first on goal plate: "
              + (f"step {reached_at[0]} ({reached_at[1]:.1f} s)" if reached_at else "never"))
        print(f"  final position     : ({end[0]:+.3f}, {end[1]:+.3f}) | "
              f"{np.hypot(*(end - GOAL)):.3f} m from goal | "
              f"{'ON the goal plate' if final['on_goal_plate'] else 'NOT on the goal plate'}")
        print(f"  takeovers          : {policy.takeovers} {dict(policy.reasons)}")
        if hasattr(policy, "speed"):
            print(f"  learned gain       : {policy.speed.gain:.2f} | floor {policy.speed.floor:.3f} | "
                  f"speed channel is echo: {policy.speed.speed_is_echo}")
        print(f"  log                : {out_path.relative_to(REPO)}")
    if not args.sim:
        print("  Note: on the robot 'pos' is the raw locator (wheel odometry), not ground truth.")
    print("=" * 70)


if __name__ == "__main__":
    main()
