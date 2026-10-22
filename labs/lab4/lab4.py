"""Lab 4 learned navigation, integrated with the proven Labs 1-3 stack.

The learned policy replaces Lab 3's normal action selection.  It does *not*
replace the measured plant model, EKF, maze, timing, logger, emergency stop, or
the A* + PID safety fallback.  That separation is the central design decision:

    estimated state x_k -> learned policy pi(x_k) -> action u_k -> robot

Run ``python labs/lab4/training.py`` first, then evaluate with
``python labs/lab4/lab4.py --sim --no-render``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
LABS_DIR = HERE.parent
LAB3_DIR = LABS_DIR / "lab3"
for dependency in (LABS_DIR, LAB3_DIR):
    if str(dependency) not in sys.path:
        sys.path.append(str(dependency))

try:
    import torch
except ImportError:
    torch = None

try:
    import lab3 as lab3_stack
    from Policy import FEATURE_NAMES, TARGET_NAMES, Policy, require_torch, wrap_angle
    import training
except ImportError:  # package-style import
    from ..lab3 import lab3 as lab3_stack
    from .Policy import FEATURE_NAMES, TARGET_NAMES, Policy, require_torch, wrap_angle
    from . import training


STUDENT_ID = lab3_stack.STUDENT_ID
WEIGHTS_PATH = HERE / "weight.pth"
LOGS = HERE / "logs"
EXPORTS = HERE / "exports"
PLOTS = HERE / "plots"
CSV_COLUMNS = list(lab3_stack.CSV_COLUMNS)


def model_contract() -> dict:
    """Machine-readable explanation shown by the dashboard."""

    return {
        "approach": "Behaviour cloning of the Lab 3 A* + PID expert",
        "raw_state": ["x", "y", "heading", "speed"],
        "features": list(FEATURE_NAMES),
        "action": ["speed_cmd", "absolute heading_cmd"],
        "independent_variables": list(FEATURE_NAMES),
        "dependent_variables": list(TARGET_NAMES),
        "raw_input": "x_k = [x, y, heading, speed] from the Lab 2 EKF",
        "raw_output": "u_k = [speed_cmd, absolute heading_cmd]",
        "state_equation": "x_(k+1) = f_d(x_k, u_k, epsilon_k)",
        "kinematics": [
            "x_(k+1) = x_k + v_(k+1) sin(theta_(k+1)) dt",
            "y_(k+1) = y_k + v_(k+1) cos(theta_(k+1)) dt",
        ],
        "loss": (
            "L = MSE(speed_hat/max_speed, speed/max_speed) + "
            "lambda * MSE([sin heading_hat, cos heading_hat], "
            "[sin heading, cos heading])"
        ),
        "validation": "Whole held-out runs; adjacent timesteps never cross the split",
        "safety": [
            "bounded non-negative speed command",
            "forward collision rollout with ball-radius clearance",
            "maximum departure from the A* path",
            "stalled-motion detection",
            "A* + PID takeover",
        ],
    }


def load_policy(weights_path: Path = WEIGHTS_PATH) -> Policy:
    require_torch()
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained weights at {path}. Train in the Lab 4 dashboard or run "
            "python labs/lab4/training.py first."
        )
    model = Policy()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was added.
        state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


#: Coast measured on the robot, run 20261001T072725Z. The drive was cut at an
#: indicated 0.090 m/s and the ball ran on 0.098 m; approaching the first corner
#: at 0.138 m/s it ran on 0.123 m. Both fit d_stop = 0.03 + 0.70 v.
#:
#: Commanded speed is not metres per second: a command of 0.042 produced an
#: indicated 0.130 m/s on that run, a factor of about 3.1, so in command units
#: d_stop = COAST_BIAS_M + COAST_GAIN_M_PER_CMD * speed_cmd.
#: The gain is per unit of COMMAND, and a command does not buy the same ground
#: speed in both places: in the simulator the action is metres per second, so
#: the gain is the measured 0.70, while on the robot a command of 0.042 gave an
#: indicated 0.130 m/s, about 3.1 times larger, hence 0.70 * 3.1 = 2.17.
COAST_BIAS_M = 0.03
COAST_GAIN_SIM = 0.70
COAST_GAIN_REAL = 2.17
#: Braking slows the ball, it never stops it: capping to zero on the approach
#: leaves it creeping too slowly to finish. The floor has to be in the units of
#: the mode it is used in, though. The simulator cruises near 0.12 and the robot
#: near 0.040, so a single floor of 0.035 left the robot's brake able to cut
#: only 0.040 to 0.035, which is why the first braked run still overshot every
#: corner by 5 to 6.5 cm.
MIN_BRAKE_SPEED_SIM = 0.035
MIN_BRAKE_SPEED_REAL = 0.008

#: Measured: commanded zero from cruise, the ball runs on 12 to 13 cm. A corner
#: zone of 0.09 therefore commands the pivot about 4 cm too late however hard
#: the approach is braked, because the stop itself cannot happen any sooner.
#: On the robot the zone is widened to the distance the ball actually needs.
CORNER_RADIUS_REAL = 0.14

#: The cap is suppressed only in the last few centimetres, where the expert's
#: pivot owns the speed and a remaining distance near zero would otherwise pin
#: the ball on the corner it is standing on. It must NOT be tied to the corner
#: zone: braking only begins where the cap first falls below cruise, which on
#: the robot is 11.7 cm out, so a 0.14 m guard suppressed the brake completely
#: and the ball arrived at the corner still at 0.136 m/s.
BRAKE_MIN_REMAINING = 0.04


def stopping_speed_cap(remaining_m: float, gain: float = COAST_GAIN_REAL,
                       floor: float = MIN_BRAKE_SPEED_REAL) -> float:
    """Largest command the ball can still stop from within ``remaining_m``.

    Without this the controller held cruise until roughly 4 cm short of a stop
    point that needed 12 cm, so every corner and the goal were overshot by 3 to
    7 cm. Capping the command by the distance that is actually left makes the
    deceleration start where the physics requires rather than where a fixed
    radius happens to fall.
    """

    usable = float(remaining_m) - COAST_BIAS_M
    cap = usable / float(gain) if usable > 0.0 else 0.0
    return max(cap, float(floor))


def distance_to_next_stop(position, corners, goal) -> float:
    """Distance to the next point the ball has to be slow at."""

    point = np.asarray(position, dtype=float)[:2]
    best = float(np.linalg.norm(point - np.asarray(goal, dtype=float)[:2]))
    for corner in corners:
        best = min(best, float(np.linalg.norm(point - np.asarray(corner, dtype=float)[:2])))
    return best


def route_corners(path: Sequence[Sequence[float]]) -> tuple:
    """Vertices of the planned polyline where the direction actually turns."""

    points = [np.asarray(point, dtype=float)[:2] for point in path]
    corners = []
    for before, here, after in zip(points, points[1:], points[2:]):
        incoming = here - before
        outgoing = after - here
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if abs(float(cross)) > 1e-9:
            corners.append((float(here[0]), float(here[1])))
    return tuple(corners)


@dataclass
class SafetyLayer:
    """Validate a policy action and substitute the expert action if unsafe."""

    planner: Any
    optimal_path: Sequence[Sequence[float]]
    dt: float
    # The simulator collides once the ball centre leaves its 0.125 m cell, so
    # the true failure boundary is 6.25 cm off the centreline. A 10 cm route
    # limit sits outside that: by the time it fires the ball has already hit a
    # wall, and in simulation a jammed true state is invisible to the
    # controller (SpheroEnv resolves the true and odometry positions
    # separately, so odometry keeps sliding while the ball is stuck). The
    # guard therefore has to trip before the collision, not after it.
    path_limit: float = 0.05
    # Probed against the 0.125 m skeleton grid this leaves a lane of only
    # +-1.8 cm. That is far stricter than the simulator (which collides on the
    # ball centre alone, +-6.25 cm) and stricter than the real 0.25 m plate.
    # Measured over 20 seeds it is nonetheless the better setting: the tight
    # lane makes the expert correct drift early, and relaxing it to 2.2 cm
    # dropped the pass rate from 18/20 to 16/20. Keep it tight.
    robot_radius: float = lab3_stack.BALL_DIAMETER_M / 2.0
    clearance_margin: float = 0.008
    lookahead_steps: int = 6
    stall_steps: int = 7
    stall_distance: float = 0.004
    # Handing the expert control does not by itself free a ball that is already
    # pressed into a wall: the expert aims at the next waypoint, which is the
    # direction that jammed it, so it grinds there until the step cap. Once the
    # ball has stopped responding for this many consecutive takeovers, drive it
    # back onto the planned centreline instead, which is roughly perpendicular
    # to the wall it is stuck on.
    # Ceiling applied to the learned speed command. The policy's own sigmoid
    # tops out at its trained max_speed_cmd; a deploy-time scale can lift it,
    # but never past the environment's vel_limit.
    max_speed: float = 0.06
    # Every corner on the route is a turn of 90 degrees over a 0.25 m leg, and
    # the tightest pair leaves no straight at all between two opposite turns.
    # The trainer drops the expert's zero-speed pivot rows, because a memoryless
    # policy sees the same state for "stop and turn" and for "drive on", so the
    # policy only ever learned the smooth part of the demonstration and rounds
    # every corner. Measured per corner, the error is 0.072 m at the tight one
    # against 0.016 m mean on the straights. Let the geometric controller keep
    # the pivots it can already execute and let the policy own the straights.
    corner_radius: float = 0.09
    corners: tuple = ()
    wedge_trigger: int = 5
    wedge_recovery_steps: int = 5
    recovery_speed: float = 0.05
    takeovers: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    _previous_position: Optional[np.ndarray] = None
    _stationary_steps: int = 0
    _wedge_steps: int = 0
    _recovery_left: int = 0

    def _free_with_clearance(self, position: Sequence[float]) -> bool:
        centre = np.asarray(position, dtype=float)[:2]
        radius = float(self.robot_radius + self.clearance_margin)
        offsets = [(0.0, 0.0)] + [
            (radius * math.cos(angle), radius * math.sin(angle))
            for angle in np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        ]
        for dx, dy in offsets:
            cell = self.planner.world_to_cell((centre[0] + dx, centre[1] + dy))
            if not self.planner._is_free(cell):
                return False
        return True

    def _near_corner(self, position: Sequence[float]) -> bool:
        if not self.corners:
            return False
        point = np.asarray(position, dtype=float)[:2]
        return any(
            float(np.linalg.norm(point - np.asarray(corner, dtype=float)[:2]))
            <= self.corner_radius
            for corner in self.corners
        )

    def _nearest_path_point(self, position: np.ndarray) -> tuple[np.ndarray, float]:
        """Closest point on the planned polyline, and the distance to it."""

        point = np.asarray(position, dtype=float)[:2]
        path = [np.asarray(p, dtype=float)[:2] for p in self.optimal_path]
        best = path[0]
        best_distance = float(np.linalg.norm(point - best))
        for start, end in zip(path[:-1], path[1:]):
            edge = end - start
            length_sq = float(np.dot(edge, edge))
            if length_sq < 1e-12:
                continue
            t = float(np.clip(np.dot(point - start, edge) / length_sq, 0.0, 1.0))
            candidate = start + t * edge
            distance = float(np.linalg.norm(point - candidate))
            if distance < best_distance:
                best_distance = distance
                best = candidate
        return best, best_distance

    def _recovery_action(self, state: np.ndarray) -> np.ndarray:
        """Unstick the ball by driving it back onto the planned centreline."""

        position = np.asarray(state, dtype=float)[:2]
        target, distance = self._nearest_path_point(position)
        if distance < 0.01:
            # Already on the line, so the jam is along it: reverse instead.
            # A negative speed is sent as the opposite heading by the env.
            return np.array([-self.recovery_speed, wrap_angle(float(state[2]))],
                            dtype=float)
        delta = target - position
        # Heading 0 is world +y and +pi/2 is world +x, so x and y swap here.
        return np.array([self.recovery_speed,
                         wrap_angle(math.atan2(delta[0], delta[1]))], dtype=float)

    def _predicted_collision(self, state: np.ndarray, action: np.ndarray) -> bool:
        predicted = np.asarray(state[:4], dtype=float).copy()
        small_dt = float(self.dt) / max(int(self.lookahead_steps), 1)
        for _ in range(max(int(self.lookahead_steps), 1)):
            predicted = lab3_stack.dynamics(predicted, action, dt=small_dt)
            if not self._free_with_clearance(predicted[:2]):
                return True
        return False

    def apply(
        self,
        state: Sequence[float],
        policy_action: Sequence[float],
        expert_action: Sequence[float],
    ) -> tuple[np.ndarray, Optional[str]]:
        state = np.asarray(state, dtype=float)
        candidate = np.asarray(policy_action, dtype=float).reshape(-1)[:2]
        expert = np.asarray(expert_action, dtype=float).reshape(-1)[:2]
        candidate[0] = float(np.clip(candidate[0], 0.0, self.max_speed))
        candidate[1] = wrap_angle(candidate[1])

        position = state[:2]
        previous = (self._previous_position.copy()
                    if self._previous_position is not None else position.copy())
        if self._previous_position is not None:
            moved = float(np.linalg.norm(position - self._previous_position))
            if candidate[0] > 0.02 and moved < self.stall_distance:
                self._stationary_steps += 1
            else:
                self._stationary_steps = 0
        self._previous_position = position.copy()

        reason = None
        if not np.all(np.isfinite(candidate)):
            reason = "non-finite policy action"
        elif self._near_corner(position):
            reason = "corner pivot"
        elif not self._free_with_clearance(position):
            reason = "insufficient wall clearance"
        elif lab3_stack.point_to_path_distance(position, self.optimal_path) > self.path_limit:
            reason = "too far from A* route"
        elif self._predicted_collision(state, candidate):
            reason = "predicted wall collision"
        elif self._stationary_steps >= self.stall_steps:
            reason = "no measured movement"

        if reason is None:
            self._wedge_steps = 0
            return candidate, None

        self.takeovers += 1
        self._stationary_steps = 0

        # A takeover that does not move the ball means the expert is pushing
        # into the same wall the policy did.
        stuck = (self._previous_position is not None
                 and float(np.linalg.norm(position - previous)) < self.stall_distance)
        if self._recovery_left > 0:
            self._recovery_left -= 1
            reason = "wedge recovery"
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            return self._recovery_action(state), reason

        self._wedge_steps = self._wedge_steps + 1 if stuck else 0
        if self._wedge_steps >= self.wedge_trigger:
            self._wedge_steps = 0
            self._recovery_left = self.wedge_recovery_steps
            reason = "wedge recovery"
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            return self._recovery_action(state), reason

        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return expert, reason


def run_mission(
    sim_env,
    *,
    robot_env=None,
    weights_path: Path = WEIGHTS_PATH,
    start=None,
    max_steps: int = lab3_stack.MAX_STEPS,
    seed: int = 0,
    hold_steps: int = 12,
    render: bool = True,
    on_step: Optional[Callable[[dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    on_env: Optional[Callable[[Any], None]] = None,
    verbose: bool = True,
    archive: bool = True,
    deadline_s: float = lab3_stack.DEADLINE_S,
    speed_scale: float = 1.0,
    corner3_slow: float = 0.6,
    corner2_slow: float = 0.6,
    goal_slow: float = 0.5,
    wedge_guard: bool = True,
    relocate: bool = True,
    corner_radius: Optional[float] = None,
    brake: bool = True,
    controller: str = "bc",
    **lab3_options,
) -> dict:
    """Run the learned policy through Lab 3's localisation and safety stack."""

    model = load_policy(weights_path)
    real = robot_env is not None
    dt = float(robot_env.dt if real else sim_env.dt)
    selected_start = lab3_stack.normalise_start(
        lab3_stack.START if start is None else start
    )
    planner, optimal_path, _waypoints = lab3_stack.plan_route(
        dt=dt, state=[selected_start[0], selected_start[1], 0.0, 0.0]
    )
    # The expert demonstrations all cruised at 0.040 m/s, so the policy cannot
    # command more than its trained ceiling. Scaling the learned speed is the
    # only dial available without faster demonstrations; vel_limit is the hard
    # stop the environment would clip to anyway.
    vel_limit = float(lab3_stack.lab2_stack.LAG_PARAMS["vel_limit"])
    scale = max(float(speed_scale), 0.0)
    coast_gain = COAST_GAIN_REAL if real else COAST_GAIN_SIM
    brake_floor = MIN_BRAKE_SPEED_REAL if real else MIN_BRAKE_SPEED_SIM
    if corner_radius is None and real:
        # The sim-tuned 0.09 cannot work on the robot: the pivot would be
        # commanded later than the ball can stop.
        corner_radius = CORNER_RADIUS_REAL
    safety = SafetyLayer(
        planner=planner, optimal_path=optimal_path, dt=dt,
        max_speed=min(0.06 * scale, vel_limit),
        corners=route_corners(optimal_path),
        **({} if corner_radius is None
           else {"corner_radius": float(corner_radius)}),
    )
    decisions: Dict[int, dict] = {}

    mpc = None
    mpc_safety = None
    # The robot's IMU collision flag from the most recent step, fed to the
    # fault detector on the next one.
    last_flags = {"collision": False}
    # Lab 4 -> lab3 channel for moving the estimate when the detector has
    # confirmed that the odometry's progress was fake.
    estimate_override: Dict[str, Any] = {}
    # MPCSafety -> runner: re-plan the route from this position.
    replan_request: Dict[str, Any] = {}
    if controller == "mpc":
        import time as _time
        # The dashboard imports this module by path, so labs/lab4 is not
        # necessarily importable the way it is when lab4.py runs as a script.
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import learned_mpc
        source = "real" if real else "sim"
        model_file = HERE / f"dynamics_model_{source}.pth"
        if not model_file.exists():
            raise FileNotFoundError(
                f"{model_file.name} missing - run dynamics_learning.py --source {source}"
            )
        # Cruise in each plant's own command units: the demonstrations on the
        # robot ran at 0.042, and the simulator's action is metres per second.
        cruise = (0.042 if real else 0.06) * scale
        # On the robot the cruise is limited only by the robot's own action
        # range (Robot vel_limit 0.6 = raw 255 with the 0.60 power scale), and
        # the Lab 1 raw-45 safety cap is lifted for Lab 4 runs (below), so the
        # speed scale is honest: 1.93 is raw 45 - what scale 20 used to give
        # once the cap clipped it - and higher scales really drive faster.
        # In simulation the environment's vel_limit still applies.
        cruise = min(cruise, REAL_CRUISE_MAX if real else vel_limit)
        if real:
            robot_env.max_raw_command = LAB4_MAX_RAW

        if real:
            # The robot meets the physical walls with its physical radius.
            walls = plate_walls(planner.map)
            ball_clearance = lab3_stack.BALL_DIAMETER_M / 2.0 + 0.005

            def wall_free(point):
                return clear_of_walls(point, walls, ball_clearance)
        else:
            def wall_free(point, probe=0.02):
                # The simulator collides on the ball centre against the
                # skeleton grid, so that is the geometry to respect there.
                for dx, dy in ((0, 0), (probe, 0), (-probe, 0), (0, probe), (0, -probe)):
                    if not planner._is_free(planner.world_to_cell(
                            (float(point[0]) + dx, float(point[1]) + dy))):
                        return False
                return True

        mpc = learned_mpc.LearnedMPC(
            learned_mpc.load_model(model_file),
            waypoints=[np.asarray(w, dtype=float) for w in _waypoints],
            start=selected_start[:2], cruise=cruise, dt=dt, is_free=wall_free,
            goal_aim_short=0.04 if real else 0.0,
            speed_factor=None,
        )
        if real:
            # Slow zones are a fraction of the speed the robot really reaches.
            # Above about scale 2 the firmware's max-power cap clips cruise, so
            # a fraction of the nominal cruise could still be at the cap and
            # slow nothing (scale 20 with 0.8 was raw 45 either way).
            top = motor_ceiling_command(robot_env, cruise)

            def zone_factor(pos, tgt):
                f = approach_speed_factor(pos, tgt, corner2_slow, corner3_slow, goal_slow)
                if wedge_guard and mpc.at_rest:
                    # A start from rest inside a slow zone has to clear
                    # stiction (raw ~21) and any floor seam or wedge: the
                    # strongest goal-zone start would otherwise be raw 28.
                    f = max(f, SLOW_ZONE_START)
                return f * top / cruise

            mpc.speed_factor = zone_factor
        # Recovery has to beat the coast whatever the speed scale is set to.
        # Tied to cruise alone it shrank with the scale: on a 0.25 run it was
        # 0.0063, too weak to stop a ball coasting off corner 3, which drifted
        # on to 9.5 cm off the route before it turned round. It is floored at
        # its value for the plant's normal cruise.
        base_cruise = 0.042 if real else 0.06
        if real:
            def near_wall(point):
                # within 1.5 cm of touching a physical wall
                return not clear_of_walls(point, walls, ball_clearance + 0.015)
        else:
            def near_wall(point):
                return not wall_free(point)
        mpc_safety = MPCSafety(
            route=optimal_path,
            recovery_speed=0.6 * max(cruise, min(base_cruise, vel_limit)),
            # A push has to beat the friction that stopped the ball.
            push_speed=min(max(cruise, base_cruise), vel_limit),
            near_wall=near_wall,
            walls=walls if real else None,
            wall_clearance=(ball_clearance if real else 0.0415),
            # relocate=False: the safety layer never moves the estimate, and
            # the wrong-turn pocket rule (which relocates) is off with it.
            estimate_override=estimate_override if relocate else None,
            replan_request=replan_request,
        )
        if not relocate:
            mpc_safety.traps = {}
        mpc_clock = {"last": None, "dt": dt}

        def mpc_dt():
            # The robot's real loop period is not the nominal 0.68 s: logged
            # runs show 0.18 to 0.22 s. The model was trained on the logged dt,
            # so on hardware it is measured; in simulation it is exact.
            if not real:
                return dt
            now = _time.monotonic()
            last = mpc_clock["last"]
            mpc_clock["last"] = now
            if last is not None and 0.05 < now - last < 1.5:
                mpc_clock["dt"] = 0.7 * mpc_clock["dt"] + 0.3 * (now - last)
            return mpc_clock["dt"]

    def choose(state, expert_action, _target, step):
        if mpc is not None:
            step_dt = mpc_dt()
            proposed = mpc.act(state[:2], step_dt)
            applied, reason = mpc_safety.apply(
                state, proposed, expert_action, mpc.last_best_hits_wall,
                dt=step_dt, collision_flag=last_flags["collision"])
            if (real and wedge_guard and reason is not None and float(applied[0]) > 0.0
                    and math.hypot(state[0] - GOAL_XY[0], state[1] - GOAL_XY[1]) < GOAL_PUSH_RADIUS):
                # A safety push next to the goal is capped: at full power a
                # ball freed from a wedge a few cm short runs straight past.
                applied = np.array([min(float(applied[0]), GOAL_PUSH * top), applied[1]], dtype=float)
            if real and past_goal_heading_on(state[:2], applied):
                # Past the goal centre: never drive further east. The coast
                # after a cut already carries the ball on, and the plate's
                # east edge is the end of the map.
                applied = np.array([0.0, applied[1]], dtype=float)
                reason = reason or "past goal: no eastward drive"
            if reason is not None:
                # The model's next prediction must start from what was sent.
                mpc.note_applied(applied)
            if replan_request.get("from") is not None:
                # Re-plan from where the ball really is: out of a wrong-turn
                # pocket, or a fresh route to the goal after the push limit.
                new_start = np.asarray(replan_request.pop("from"), dtype=float)
                _pl, new_path, new_wps = lab3_stack.plan_route(
                    dt=dt, state=[new_start[0], new_start[1], 0.0, 0.0])
                if new_wps:
                    mpc.points = [new_start.copy()] + [np.asarray(w, dtype=float)[:2] for w in new_wps]
                    mpc.index, mpc.done = 1, False
                    mpc_safety.set_route(new_path)
                    mpc.prev_pos = None
                    mpc.vel_est = np.zeros(2)
            if mpc_safety.last_relocated is not None:
                # The estimate is about to jump to where the ball really is;
                # that jump is not motion.
                mpc.prev_pos = None
                mpc.vel_est = np.zeros(2)
            # Tell the detector where the model expects the ball next step.
            mpc_safety.expect(mpc.predict_next(state[:2], applied, step_dt))
            decisions[int(step)] = {
                "policy_speed_cmd": float(proposed[0]),
                "policy_heading_cmd": float(proposed[1]),
                "safety_takeover": bool(reason),
                "safety_reason": reason or "",
            }
            return applied

        with torch.no_grad():
            # network_action, not __call__: Policy.forward now runs the whole
            # self-contained controller for the M4-2 submission, and this
            # runner already owns an equivalent planner and safety layer.
            learned = model.network_action(
                torch.tensor(state[:4], dtype=torch.float32)
            ).cpu().numpy()
        learned = np.asarray(learned, dtype=float).copy()
        learned[0] = min(learned[0] * scale, vel_limit)
        # Cap by what the ball can actually stop from, but only on the
        # approach. Inside the corner zone the pivot already owns the speed,
        # and a cap there would read a remaining distance of nearly zero and
        # pin the ball in place, unable to leave the corner it is standing on.
        in_goal_plate = lab3_stack.fully_in_goal_plate(state[:2])
        braking = False
        if brake and not in_goal_plate:
            remaining = distance_to_next_stop(
                state[:2], safety.corners, lab3_stack.GOAL
            )
            if remaining > BRAKE_MIN_REMAINING:
                braking = True
                cap = stopping_speed_cap(remaining, coast_gain, brake_floor)
                if cap < learned[0]:
                    learned[0] = max(cap, 0.0)

        if in_goal_plate:
            # Being inside the plate earns the distance mark, but stopping at
            # its entry edge leaves no margin for noise or the raised lip. Let
            # the proven controller centre the ball; only hold once it is
            # within 3 cm of the actual centre.
            if float(np.linalg.norm(state[:2] - lab3_stack.GOAL)) <= 0.03:
                applied = np.array([0.0, float(learned[1])], dtype=float)
            else:
                applied = np.asarray(expert_action, dtype=float).copy()
            reason = None
        else:
            applied, reason = safety.apply(state, learned, expert_action)
        # The cap is NOT applied to the expert. When the safety layer hands
        # over, the action comes from the Lab 3 follower, which drove these
        # same corners at 0.022 m on hardware for M3 and already shapes its own
        # approach. Capping it as well crawled the ball to 0.008 for whole
        # seconds before each corner, which tripped the stall detector and
        # turned into wedge recoveries reversing out: path error went from
        # 0.067 to 0.078 m and the run from 110 to 133 steps.

        decisions[int(step)] = {
            "policy_speed_cmd": float(learned[0]),
            "policy_heading_cmd": float(learned[1]),
            "safety_takeover": bool(reason),
            "safety_reason": reason or "",
        }
        return applied

    def publish(record):
        if "collision" in record:
            last_flags["collision"] = bool(record.get("collision"))
        enriched = dict(record)
        enriched.update(decisions.get(int(record.get("step", 0)), {
            "policy_speed_cmd": 0.0,
            "policy_heading_cmd": float(record.get("heading_cmd", 0.0)),
            "safety_takeover": False,
            "safety_reason": "",
        }))
        if on_step is not None:
            on_step(enriched)

    result = lab3_stack.run_mission(
        sim_env,
        robot_env=robot_env,
        start=selected_start,
        max_steps=max_steps,
        seed=seed,
        hold_steps=hold_steps,
        render=render,
        on_step=publish,
        should_stop=should_stop,
        on_env=on_env,
        verbose=False,
        archive=False,
        deadline_s=deadline_s,
        action_hook=choose,
        estimate_override=estimate_override,
        **lab3_options,
    )
    for record in result["records"]:
        record.update(decisions.get(int(record.get("step", 0)), {
            "policy_speed_cmd": 0.0,
            "policy_heading_cmd": float(record.get("heading_cmd", 0.0)),
            "safety_takeover": False,
            "safety_reason": "",
        }))

    total_policy_steps = max(len(result["records"]) - 1, 0)
    result["lab"] = 4
    result["approach"] = "behaviour-cloning"
    result["weights"] = str(Path(weights_path))
    # Report whichever safety layer was actually in charge of this run.
    active_safety = mpc_safety if mpc_safety is not None else safety
    result["report"].update({
        "policy_steps": total_policy_steps,
        "safety_takeovers": int(active_safety.takeovers),
        "safety_takeover_fraction": (
            float(active_safety.takeovers) / total_policy_steps if total_policy_steps else 0.0
        ),
        "safety_reasons": dict(active_safety.reasons),
        "controller": controller,
        # Recorded so runs can be compared without inferring the setting from
        # the commands after the fact.
        "speed_scale": float(scale),
        "corner3_slow": float(corner3_slow),
        "corner2_slow": float(corner2_slow),
        "goal_slow": float(goal_slow),
        "wedge_guard": bool(wedge_guard),
        "relocate": bool(relocate),
    })
    if archive:
        result["run_dir"] = str(archive_run(result))
    if verbose:
        print(format_report(result["report"]))
    return result


#: Cruise multiplier on the straight up column 1 into corner 3, (1,1) -> (1,0).
#: On the robot the ball arrived too fast, hit the north wall of (1,0), and
#: bounced into the opening south of (2,0). Approaching slower lets it turn.
CORNER3_SLOW = 0.6


#: Same for the short (0,2) -> (1,2) run into corner 2.
CORNER2_SLOW = 0.6


#: Goal approach: the top speed ramps from full at GOAL_SLOW_FAR from the goal
#: down to GOAL_SLOW of it at GOAL_SLOW_NEAR and inside. On the robot the
#: final leg ran at raw 45 (~0.16 m/s) to within 6-10 cm of the goal and the
#: ball coasted 6-14 cm after the cut, out to x = 0.58, past the plate edge.
GOAL_SLOW = 0.5
GOAL_SLOW_FAR = 0.30
GOAL_SLOW_NEAR = 0.12
GOAL_XY = (0.5, 0.5)
#: Lab 4 robot runs: motor power ceiling (Lab 1's safety cap is 45) and the
#: largest cruise command (the Robot's action range, raw 255).
LAB4_MAX_RAW = 255
REAL_CRUISE_MAX = 0.6
#: Inside a slow zone a start from rest gets at least this fraction of top.
SLOW_ZONE_START = 0.8
#: Safety pushes within GOAL_PUSH_RADIUS of the goal are capped at this
#: fraction of top (raw ~33 at the 45 cap).
GOAL_PUSH = 0.65
GOAL_PUSH_RADIUS = 0.15


def past_goal_heading_on(position, action, margin: float = 0.02) -> bool:
    """True when the ball is east of the goal centre and told to go further east."""

    x, y = float(position[0]), float(position[1])
    if abs(y - GOAL_XY[1]) > 0.125 or x <= GOAL_XY[0] + margin:
        return False
    return float(action[0]) > 0.0 and math.sin(float(action[1])) > 0.3


def approach_speed_factor(position, target, slow2: float = CORNER2_SLOW,
                          slow3: float = CORNER3_SLOW,
                          goal_slow: float = 1.0) -> float:
    """Slow-zone multiplier: into corner 2 along row 2, into corner 3 up column 1,
    and a ramp down on the final approach to the goal."""

    tx, ty = float(target[0]), float(target[1])
    x, y = float(position[0]), float(position[1])
    # goal: target is the goal, ball on the top row
    if (abs(tx - GOAL_XY[0]) < 0.05 and abs(ty - GOAL_XY[1]) < 0.05
            and abs(y - GOAL_XY[1]) < 0.125):
        d = math.hypot(x - GOAL_XY[0], y - GOAL_XY[1])
        ramp = (d - GOAL_SLOW_NEAR) / (GOAL_SLOW_FAR - GOAL_SLOW_NEAR)
        return float(goal_slow) + (1.0 - float(goal_slow)) * min(max(ramp, 0.0), 1.0)
    # corner 3 at (1,0): target there, ball in (1,1) or (1,0)
    if (abs(tx + 0.25) < 0.125 and abs(ty - 0.5) < 0.125
            and abs(x + 0.25) < 0.125 and 0.125 <= y <= 0.625):
        return float(slow3)
    # corner 2 at (1,2): target there, ball in (0,2) or (1,2)
    if (abs(tx + 0.25) < 0.125 and abs(ty) < 0.125
            and -0.625 <= x <= -0.125 and abs(y) < 0.125):
        return float(slow2)
    return 1.0


def motor_ceiling_command(robot_env, cruise: float) -> float:
    """Smallest command that already gives the same raw motor value as cruise.

    Anything above it is clipped by max_raw_command, so it is the real top.
    """

    to_raw = None
    for obj in (robot_env, getattr(robot_env, "unwrapped", None),
                getattr(robot_env, "robot", None), getattr(robot_env, "env", None)):
        if obj is not None and hasattr(obj, "speed_cmd_to_raw"):
            to_raw = obj.speed_cmd_to_raw
            break
    if to_raw is None:
        # config defaults: raw 45 cap, raw_min_move 12, power scale 0.60
        return min(float(cruise), (45 - 12) / 243.0 * 0.60)
    top, cap = float(cruise), to_raw(cruise)
    while top > 0.01 and to_raw(top * 0.99) >= cap:
        top *= 0.99
    return top


def plate_walls(occupancy, plate: float = 0.25):
    """Wall segments of the physical maze, recovered from the occupancy grid.

    The grid is a 0.125 m skeleton: plate centres and the connectors between
    them are free, so a free corridor in it is only half the width of the 0.25 m
    plate the ball actually rolls on. Checking the ball against it said a ball
    6 cm off the centreline of a column was touching the wall, when physically
    the plate edge was still 2.5 cm beyond the ball's edge. On the robot the
    real geometry is used instead: the lines between plates where the grid's
    connector cell is occupied, plus the outer boundary.
    """

    occ = np.asarray(occupancy)
    n = (occ.shape[0] - 1) // 2
    half = n * plate / 2.0
    walls = [((-half, -half), (half, -half)), ((-half, half), (half, half)),
             ((-half, -half), (-half, half)), ((half, -half), (half, half))]
    for row in range(n):
        for col in range(n):
            cx, cy = (col - (n - 1) / 2.0) * plate, ((n - 1) / 2.0 - row) * plate
            if col < n - 1 and occ[2 * row + 1][2 * col + 2]:
                walls.append(((cx + plate / 2, cy - plate / 2), (cx + plate / 2, cy + plate / 2)))
            if row < n - 1 and occ[2 * row + 2][2 * col + 1]:
                walls.append(((cx - plate / 2, cy - plate / 2), (cx + plate / 2, cy - plate / 2)))
    return [(np.asarray(a, dtype=float), np.asarray(b, dtype=float)) for a, b in walls]


def clear_of_walls(point, walls, clearance: float) -> bool:
    """True if a disc of radius ``clearance`` at ``point`` touches no wall."""

    p = np.asarray(point, dtype=float)[:2]
    for a, b in walls:
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
        if float(np.linalg.norm(p - (a + t * ab))) < clearance:
            return False
    return True


class RouteGeometry:
    """Arc-length view of the planned route, for progress and back-tracking."""

    def __init__(self, route):
        self.points = [np.asarray(q, dtype=float)[:2] for q in route]
        self.cum = [0.0]
        for a, b in zip(self.points[:-1], self.points[1:]):
            self.cum.append(self.cum[-1] + float(np.linalg.norm(b - a)))
        self.length = self.cum[-1]

    def project(self, position):
        """(distance off the route, progress along it, unit tangent there)."""

        p = np.asarray(position, dtype=float)[:2]
        best = (math.inf, 0.0, np.array([0.0, 1.0]))
        for i, (a, b) in enumerate(zip(self.points[:-1], self.points[1:])):
            edge = b - a
            length = float(np.linalg.norm(edge))
            if length < 1e-12:
                continue
            u = edge / length
            t = float(np.clip(np.dot(p - a, u), 0.0, length))
            d = float(np.linalg.norm(p - (a + u * t)))
            if d < best[0]:
                best = (d, self.cum[i] + t, u)
        return best

    def at(self, progress):
        """Point and tangent at a given progress along the route."""

        progress = float(np.clip(progress, 0.0, self.length))
        for i, (a, b) in enumerate(zip(self.points[:-1], self.points[1:])):
            if progress <= self.cum[i + 1] or i == len(self.points) - 2:
                edge = b - a
                length = float(np.linalg.norm(edge)) or 1e-12
                u = edge / length
                return a + u * (progress - self.cum[i]), u
        return self.points[-1], np.array([0.0, 1.0])


@dataclass
class MPCSafety:
    """Fault detector and fail-safes for the learned-dynamics MPC.

    The Lab 4 brief asks for fail safes and safety layers that keep the learned
    controller in line. On the robot two things made simple rules fail: the
    position is wheel odometry, so a jammed ball whose wheels slip reports
    progress it never made; and recovering by reversing drove the ball off the
    back of the start plate and, at a floor bump, into an endless loop.

    Signals, per step:
      off    distance from the route
      stuck  commanded to move AND the estimate barely moving
      jolt   the robot's IMU collision flag (Robot.step), which odometry cannot
             fake: it is the only independent evidence of contact
      short  the ball ended well short of where the learned model predicted
      wall   within ``near_wall`` of a physical wall

    Rules, in priority order:
      1. non-finite command                      -> A* + PID expert
      2. a push or back-off in progress          -> carry it on
      3. (jolt OR short) AND wall                -> "collision": back to route
      4. off > hard limit, or > limit for a time -> "off corridor"
      5. moving AND short for scrape_s           -> "scrape"
      6. jolts repeated within imu_window_s,
         OR stuck for stall_s after start-up     -> PUSH, then CHECK
      7. every MPC candidate hits a wall         -> stop; escalate to a push

    Push and check. Instead of reversing, drive forward along the route at
    ``push_speed`` for ``push_steps``. Then check: real progress and no jolt
    means it is free, so the MPC resumes. Otherwise it is confirmed stuck, and
    the detector works out where the ball can actually be:

      * a jolt means something was in front of the ball. If the map has a wall
        ahead of the estimate, the estimate explains it and is kept;
      * if not - open floor ahead of the estimate - the estimate cannot be
        right. Walk back along the route, through any corner passed, to the
        first position where a wall in the push direction would explain the
        jolt, within ``backtrack_limit``. That is where the ball is;
      * if nothing explains it (a floor bump rather than a wall), take the
        "no real progress" explanation: the ball is where the push began.

    The estimate is moved there (and the odometry frame with it, in lab3) only
    when that evidence is complete; otherwise it is left alone. Then the ball
    backs off away from the obstacle - only if there is route behind it, never
    off the start plate - and pushes again with a small alternating heading
    offset to get round a bump.

    Every takeover is counted and its reason logged.
    """

    route: Sequence[Sequence[float]]
    recovery_speed: float
    push_speed: float = 0.042
    near_wall: Optional[Callable[[Sequence[float]], bool]] = None
    walls: Optional[list] = None
    wall_clearance: float = 0.0415
    estimate_override: Optional[dict] = None
    corridor_limit: float = 0.07
    hard_corridor_limit: float = 0.08
    off_track_s: float = 0.4
    scrape_s: float = 0.4
    #: 1.5 s rather than 1.0: a high-friction start cell takes longer than a
    #: second to get going, and the old setting fired during normal starts.
    stall_s: float = 1.5
    start_grace_s: float = 1.5
    stall_distance: float = 0.004
    min_moving_cmd: float = 0.015
    shortfall: float = 0.02
    imu_window_s: float = 2.0
    imu_hits_needed: int = 2
    push_steps: int = 3
    backoff_steps: int = 2
    check_progress: float = 0.02
    #: Backing off needs route behind the ball; the start plate has none.
    backoff_min_progress: float = 0.12
    backtrack_limit: float = 0.30
    push_lookahead: float = 0.08
    contact_reach: float = 0.15
    wall_stop_limit: int = 3
    #: Known wrong turns on this maze, from the robot runs: estimate plate ->
    #: where the ball really is. The top straight is open on the south side of
    #: (2,0), so the ball drops into (2,1) and jams in (3,1) while the
    #: odometry still has it on the straight; corner 2 is open to the east, so
    #: the ball overshoots into (2,2) and hits its wall while the estimate has
    #: it climbing the column in (1,1). Plates are (col, row), row 0 at the top.
    traps: Dict = field(default_factory=lambda: {(2, 0): (3, 1), (3, 0): (3, 1), (1, 1): (2, 2)})
    replan_request: Optional[dict] = None
    #: Push at most this many times, then hand back to the MPC re-planned for
    #: the goal; stuck again starts the cycle again, so it never gives up.
    push_limit: int = 2
    #: "Too slow": less than slow_progress along the route in slow_window_s
    #: while driving at at least half cruise. Braking for a corner is below
    #: half cruise, so it does not count.
    slow_window_s: float = 2.0
    slow_progress: float = 0.04
    _pushes: int = 0
    _slow: List[tuple] = field(default_factory=list)
    #: Held at zero while stopped and not at the goal. The MPC now refuses to
    #: stay stopped itself; this catches anything else that holds it.
    idle_s: float = 0.8
    goal_radius: float = 0.06
    _idle_t: float = 0.0
    takeovers: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)
    relocations: List[dict] = field(default_factory=list)
    _geo: Optional[RouteGeometry] = None
    _previous_position: Optional[np.ndarray] = None
    _expected: Optional[np.ndarray] = None
    _elapsed: float = 0.0
    _off_t: float = 0.0
    _scrape_t: float = 0.0
    _stuck_t: float = 0.0
    _hit_times: List[float] = field(default_factory=list)
    _wall_stops: int = 0
    _mode: str = "normal"
    _left: int = 0
    _push_heading: float = 0.0
    _push_anchor: Optional[np.ndarray] = None
    _push_s0: float = 0.0
    _push_hits: int = 0
    _attempt: int = 0
    _back_heading: float = 0.0
    last_relocated: Optional[np.ndarray] = None

    def __post_init__(self):
        self._geo = RouteGeometry(self.route)

    # ------------------------------------------------------------- helpers
    def _back_to_route(self, position: np.ndarray) -> np.ndarray:
        d, prog, u = self._geo.project(position)
        if d < 0.01:
            # On the line already: carry on along it, never backwards.
            return np.array([self.recovery_speed, wrap_angle(math.atan2(u[0], u[1]))], dtype=float)
        target, _ = self._geo.at(prog)
        return np.array([
            self.recovery_speed,
            wrap_angle(math.atan2(target[0] - position[0], target[1] - position[1])),
        ], dtype=float)

    def _note(self, reason: str) -> str:
        self.takeovers += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        return reason

    def _reset_timers(self):
        self._off_t = self._scrape_t = self._stuck_t = 0.0
        self._wall_stops = 0

    def _wall_ahead(self, point, heading) -> bool:
        """Is a mapped wall within contact reach in the push direction?"""

        if not self.walls:
            return False
        u = np.array([math.sin(heading), math.cos(heading)])
        for k in range(1, 6):
            q = np.asarray(point, dtype=float)[:2] + u * (self.contact_reach * k / 5.0)
            if not clear_of_walls(q, self.walls, self.wall_clearance):
                return True
        return False

    def _locate(self, estimate: np.ndarray, heading: float) -> tuple:
        """Where can the ball really be, given a confirmed contact?"""

        if self._wall_ahead(estimate, heading):
            return None, "estimate explains the contact"
        _d, prog, _u = self._geo.project(estimate)
        # Search outward in BOTH directions and take the nearest explanation.
        # The odometry can be wrong either way: slip over-reports progress,
        # and a bounce can throw the ball forward of where it was believed.
        step = 0.02
        k = step
        while k <= self.backtrack_limit + 1e-9:
            for sign, word in ((-1.0, "back"), (1.0, "ahead")):
                candidate, _ = self._geo.at(prog + sign * k)
                if self._wall_ahead(candidate, heading):
                    return candidate, f"wall explains it {100*k:.0f} cm {word} along the route"
            k += step
        if self._push_anchor is not None:
            return self._push_anchor.copy(), "no wall explains it: no real progress, back to the push start"
        return None, "no explanation found"

    def _start_push(self, position: np.ndarray, reason: str):
        _d, prog, u = self._geo.project(position)
        # Aim at a point a little AHEAD along the route, not along the tangent
        # at the nearest point: sitting on a corner, that tangent is the
        # incoming leg and pushed the ball straight past the corner into a
        # dead end instead of up the next leg.
        target, _ = self._geo.at(prog + self.push_lookahead)
        offset = 0.0 if self._attempt == 0 else (0.25 if self._attempt % 2 else -0.25)
        self._push_heading = wrap_angle(
            math.atan2(target[0] - position[0], target[1] - position[1]) + offset)
        self._push_anchor = position.copy()
        self._push_s0 = prog
        self._push_hits = 0
        self._mode, self._left = "push", self.push_steps
        self._reset_timers()
        self._hit_times.clear()
        return np.array([self.push_speed, self._push_heading], dtype=float), self._note(reason)

    @staticmethod
    def plate(position) -> tuple:
        x, y = float(position[0]), float(position[1])
        return (int(np.clip(math.floor((x + 0.625) / 0.25), 0, 4)),
                int(np.clip(math.floor((0.625 - y) / 0.25), 0, 4)))

    @staticmethod
    def plate_centre(plate) -> np.ndarray:
        return np.array([(plate[0] - 2) * 0.25, (2 - plate[1]) * 0.25], dtype=float)

    def set_route(self, route) -> None:
        """Adopt a re-planned route (after a relocation or a refocus)."""

        self.route = route
        self._geo = RouteGeometry(route)
        self._slow.clear()

    def _relocate(self, where, why: str):
        where = np.asarray(where, dtype=float)[:2]
        if self.estimate_override is not None:
            self.estimate_override["xy"] = where
        self.last_relocated = where
        self.relocations.append({"to": where.tolist(), "why": why, "t": round(self._elapsed, 2)})

    def _replan(self, start):
        if self.replan_request is not None:
            self.replan_request["from"] = np.asarray(start, dtype=float)[:2]

    def _stuck(self, position: np.ndarray, reason: str, action):
        """Stuck by any measure: wrong-turn pocket, else push, else refocus."""

        here = self.plate(position)
        if here in self.traps:
            true = self.traps[here]
            centre = self.plate_centre(true)
            self._relocate(centre, f"stuck with the estimate in {here}: wrong turn into {true}")
            self._replan(centre)
            self._mode, self._attempt, self._pushes = "normal", 0, 0
            self._reset_timers(); self._hit_times.clear(); self._slow.clear()
            return np.array([0.0, action[1]], dtype=float), self._note(f"wrong turn: in {true}, re-planning")
        if self._pushes < self.push_limit:
            self._pushes += 1
            return self._start_push(position, reason)
        # pushed enough: refocus on the goal from here and let the MPC drive
        self._pushes = 0
        self._replan(position)
        self._mode, self._attempt = "normal", 0
        self._reset_timers(); self._hit_times.clear(); self._slow.clear()
        return np.array([0.0, action[1]], dtype=float), self._note("refocus on the goal")

    def at_goal(self, position) -> bool:
        return float(np.linalg.norm(np.asarray(position, dtype=float)[:2]
                                    - self._geo.points[-1])) <= self.goal_radius

    def expect(self, predicted_xy) -> None:
        self._expected = (None if predicted_xy is None
                          else np.asarray(predicted_xy, dtype=float)[:2].copy())

    # ----------------------------------------------------------------- main
    def apply(self, state, action, expert_action, wall_predicted: bool,
              dt: float = 0.18, collision_flag: bool = False):
        state = np.asarray(state, dtype=float)
        position = state[:2]
        action = np.asarray(action, dtype=float).reshape(-1)[:2].copy()
        dt = float(dt) if dt else 0.18
        self._elapsed += dt
        self.last_relocated = None

        moved = (float(np.linalg.norm(position - self._previous_position))
                 if self._previous_position is not None else None)
        short = False
        if self._expected is not None and self._previous_position is not None:
            want = self._expected - self._previous_position
            got = position - self._previous_position
            if float(np.linalg.norm(want)) > 0.01:
                short = (float(np.linalg.norm(got - want)) > self.shortfall
                         and float(np.linalg.norm(got)) < float(np.linalg.norm(want)))
        self._previous_position = position.copy()
        self._expected = None
        off, prog, _u = self._geo.project(position)
        wall = bool(self.near_wall is not None and self.near_wall(position))
        jolt = bool(collision_flag)
        if jolt:
            self._hit_times.append(self._elapsed)
        self._hit_times = [t for t in self._hit_times if self._elapsed - t <= self.imu_window_s]

        if not np.all(np.isfinite(action)):
            self._note("non-finite command")
            return np.asarray(expert_action, dtype=float).copy(), "non-finite command"

        # ---- a push in progress, then the check
        if self._mode == "push":
            self._push_hits += int(jolt)
            if self._left > 0:
                self._left -= 1
                return np.array([self.push_speed, self._push_heading], dtype=float), self._note("pushing")
            # Judge "stuck" by whether the ball actually moved, not by progress
            # along the route: a ball moving across the route is free, and
            # calling it stuck sent it backing off and pushing down the wrong
            # corridor. Contact during the push still overrides.
            moved_by_push = float(np.linalg.norm(position - self._push_anchor))
            if moved_by_push >= self.check_progress and self._push_hits == 0:
                self._mode, self._attempt, self._pushes = "normal", 0, 0
                self._reset_timers()
                return action, None                                   # free: MPC resumes
            # confirmed stuck. A known wrong-turn pocket takes priority.
            if self.plate(position) in self.traps:
                self._mode = "normal"
                return self._stuck(position, "stuck after push", action)
            # pushed the limit: refocus on the goal rather than pushing again
            if self._pushes >= self.push_limit:
                self._mode = "normal"
                return self._stuck(position, "stuck after push", action)
            # otherwise work out where the ball really is
            where, why = (self._locate(position, self._push_heading)
                          if self._push_hits > 0 else (None, "no contact: bump or friction"))
            self._attempt += 1
            if where is not None and self.estimate_override is not None:
                self.estimate_override["xy"] = where
                self.last_relocated = where
                self.relocations.append({"from": position.tolist(), "to": list(map(float, where)),
                                         "why": why, "t": round(self._elapsed, 2)})
                position = where
                off, prog, _u = self._geo.project(position)
            if prog > self.backoff_min_progress:
                self._pushes += 1
                _d, _p, u = self._geo.project(position)
                self._back_heading = wrap_angle(math.atan2(u[0], u[1]))
                self._mode, self._left = "backoff", self.backoff_steps - 1
                # A negative speed is sent as the opposite heading by the env.
                return np.array([-self.recovery_speed, self._back_heading], dtype=float), self._note("stuck: backing off")
            self._pushes += 1
            return self._start_push(position, "stuck: pushing again")

        if self._mode == "backoff":
            if self._left > 0:
                self._left -= 1
                return np.array([-self.recovery_speed, self._back_heading], dtype=float), self._note("stuck: backing off")
            return self._start_push(position, "stuck: pushing again")

        # ---- ordinary rules
        if (jolt or short) and wall:
            self._reset_timers()
            return self._back_to_route(position), self._note("collision")

        if off > self.hard_corridor_limit:
            self._reset_timers()
            return self._back_to_route(position), self._note("off corridor")
        self._off_t = self._off_t + dt if off > self.corridor_limit else 0.0
        if self._off_t >= self.off_track_s:
            self._reset_timers()
            return self._back_to_route(position), self._note("off corridor")

        moving = moved is not None and moved >= self.stall_distance
        self._scrape_t = self._scrape_t + dt if (moving and short) else 0.0
        if self._scrape_t >= self.scrape_s:
            self._reset_timers()
            return self._back_to_route(position), self._note("scrape")

        if len(self._hit_times) >= self.imu_hits_needed:
            return self._stuck(position, "repeated contact: push and check", action)

        # too slow: little route progress over a window while driving
        if action[0] >= 0.5 * self.push_speed:
            self._slow.append((self._elapsed, prog))
            # Keep one step beyond the window: trimming at exactly the window
            # meant the oldest kept sample was always a step short of it, so
            # the check below could never be met.
            self._slow = [(t, pg) for t, pg in self._slow
                          if self._elapsed - t <= self.slow_window_s + dt + 1e-9]
            if (self._slow and self._elapsed - self._slow[0][0] >= self.slow_window_s - 1e-9
                    and prog - self._slow[0][1] < self.slow_progress
                    and self._elapsed > self.start_grace_s):
                self._slow.clear()
                return self._stuck(position, "too slow: push and check", action)
        else:
            self._slow.clear()

        commanded = action[0] > self.min_moving_cmd
        if self._elapsed > self.start_grace_s:
            self._stuck_t = (self._stuck_t + dt
                             if (commanded and moved is not None and not moving) else 0.0)
        if self._stuck_t >= self.stall_s:
            return self._stuck(position, "stall: push and check", action)

        # Backstop: told to stand still, not moving, for idle_s -> push.
        not_moving = moved is not None and moved < self.stall_distance
        self._idle_t = (self._idle_t + dt
                        if (action[0] <= self.min_moving_cmd and not_moving
                            and self._elapsed > self.start_grace_s and not self.at_goal(position))
                        else 0.0)
        if self._idle_t >= self.idle_s:
            self._idle_t = 0.0
            return self._stuck(position, "idle: push and check", action)

        if wall_predicted:
            self._wall_stops += 1
            if self._wall_stops > self.wall_stop_limit:
                return self._stuck(position, "wall stops: push and check", action)
            self._note("predicted wall contact")
            return np.array([0.0, action[1]], dtype=float), "predicted wall contact"
        self._wall_stops = 0
        return action, None


#: Archive label per controller, so runs of the two can be told apart later.
#: Every run was labelled behaviour-cloning-safe before, including MPC runs.
CONTROLLER_LABELS = {"bc": "behaviour-cloning-safe", "mpc": "learned-dynamics-mpc"}


def archive_run(result: dict) -> Path:
    label = CONTROLLER_LABELS.get(
        str(result.get("report", {}).get("controller", "bc")), "behaviour-cloning-safe")
    run_dir = lab3_stack.runlog.new_run_dir(result["mode"], "lab4", label)
    writer = lab3_stack.runlog.RunWriter(run_dir, {
        "source": result["mode"],
        "experiment": "lab4",
        "model": label,
        "controller": str(result.get("report", {}).get("controller", "bc")),
        "weights": result.get("weights"),
        "start_pose": [*result["start"], 0.0, 0.0],
        "goal": result["goal"],
        "optimal_path": result["optimal_path"],
        "waypoints": result["waypoints"],
        "model_contract": model_contract(),
    })
    for row in result["records"]:
        writer.append({
            "step": row["step"], "t": row["t"], "dt": row["dt"],
            "speed_cmd": row["speed_cmd"], "heading_cmd": row["heading_cmd"],
            "x": row["sim_x"], "y": row["sim_y"],
            "heading": row["heading"], "speed": row["speed"],
            "odom_x": row["real_x"], "odom_y": row["real_y"],
            "setpoint_x": row["target_x"], "setpoint_y": row["target_y"],
            "dist_to_goal": row["dist_to_goal"], "collision": row["collision"],
        })
    writer.close(
        status="stopped" if result["report"].get("stopped_by_user") else "complete",
        metrics=result["report"],
    )
    lab3_stack.plot_run(result, out_path=run_dir / "plots" / "lab4_route.png")
    (run_dir / "policy_decisions.json").write_text(
        json.dumps([
            {
                "step": row["step"],
                "policy_speed_cmd": row["policy_speed_cmd"],
                "policy_heading_cmd": row["policy_heading_cmd"],
                "safety_takeover": row["safety_takeover"],
                "safety_reason": row["safety_reason"],
            }
            for row in result["records"]
        ], indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def write_csv(result: dict, student_id: str = STUDENT_ID, out_dir=None):
    records = result.get("records") or []
    if not records:
        raise ValueError("run produced no steps")
    destination = Path(out_dir) if out_dir else EXPORTS
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{student_id}_lab4.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    verdict = destination / f"{student_id}_lab4_verdict.json"
    verdict.write_text(json.dumps({
        "csv": str(path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(records),
        "mode": result.get("mode"),
        "report": result.get("report", {}),
    }, indent=2) + "\n", encoding="utf-8")
    return path, verdict


def plot_run(result: dict, student_id: str = STUDENT_ID) -> Path:
    path = PLOTS / f"{student_id}_lab4.png"
    return lab3_stack.plot_run(result, out_path=path)


def format_report(report: dict) -> str:
    base = lab3_stack.format_report(report).replace("Lab 3", "Lab 4")
    return (
        f"{base}\n"
        f"  learned-policy steps={report.get('policy_steps', 0)} | "
        f"safety takeovers={report.get('safety_takeovers', 0)} "
        f"({100.0 * report.get('safety_takeover_fraction', 0.0):.1f}%)"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Lab 4 - learned policy with Lab 1-3 model, EKF and safety"
    )
    parser.add_argument("--sim", action="store_true", help="run without BLE hardware")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--steps", type=int, default=lab3_stack.MAX_STEPS)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hold-steps", type=int, default=12)
    parser.add_argument("--weights", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--toy", default=None)
    parser.add_argument("--aim", dest="aim", action="store_true", default=None)
    parser.add_argument("--no-aim", dest="aim", action="store_false")
    parser.add_argument("--power-scale", type=float,
                        default=lab3_stack.lab2_stack.LAB2_PHYSICAL_MAX_SPEED)
    parser.add_argument("--deadline", type=float, default=lab3_stack.DEADLINE_S)
    parser.add_argument("--odom-scale", type=float, default=1.0)
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="multiply the learned speed command (1.0 = as trained)")
    parser.add_argument("--no-relocate", action="store_true",
                        help="safety never moves the position estimate")
    parser.add_argument("--no-wedge-guard", action="store_true",
                        help="no restart kick in slow zones, no push cap near the goal")
    parser.add_argument("--goal-slow", type=float, default=GOAL_SLOW,
                        help="fraction of top speed on the last 12 cm to the goal, robot only")
    parser.add_argument("--corner2-slow", type=float, default=CORNER2_SLOW,
                        help="speed multiplier into corner 2, (0,2)->(1,2), robot only")
    parser.add_argument("--corner3-slow", type=float, default=CORNER3_SLOW,
                        help="speed multiplier into corner 3, (1,1)->(1,0), robot only")
    parser.add_argument("--controller", choices=("bc", "mpc"), default="bc",
                        help="bc: behaviour-cloned policy with the safety layer "
                             "(default, the tested stack). mpc: model-predictive "
                             "control against the learned dynamics model")
    parser.add_argument("--no-brake", dest="brake", action="store_false",
                        default=True,
                        help="disable the stopping-distance speed cap "
                             "(it is what stops corners and the goal being "
                             "overshot on hardware)")
    parser.add_argument("--corner-radius", type=float, default=None,
                        help="radius of the corner-pivot zone, metres (default 0.09). "
                             "At the real dt of 0.68 s the ball covers 4-10 cm per "
                             "step, so a zone tuned in simulation can be stepped "
                             "over entirely; widen it if corners are being cut")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dt = float(args.dt if args.dt is not None else (
        lab3_stack.SIM_DT if args.sim else lab3_stack.REAL_DT
    ))
    render = not args.no_render
    _planner, _optimal, waypoints = lab3_stack.plan_route(dt=dt)
    first_waypoint_local = np.asarray(waypoints[0]) - lab3_stack.START

    with lab3_stack.managed_sim_env(
        dt=dt, render=render, max_steps=args.steps
    ) as sim_env:
        if args.sim:
            result = run_mission(
                sim_env, weights_path=args.weights, max_steps=args.steps,
                seed=args.seed, hold_steps=args.hold_steps, render=render,
                archive=not args.no_archive, deadline_s=args.deadline,
                speed_scale=args.speed_scale, corner_radius=args.corner_radius,
                brake=args.brake, controller=args.controller,
            )
        else:
            do_aim = True if args.aim is None else bool(args.aim)
            with lab3_stack.managed_robot_env(
                dt=dt, toy_name=args.toy, aim=do_aim,
                first_waypoint_local=first_waypoint_local,
                power_scale=args.power_scale,
            ) as robot_env:
                result = run_mission(
                    sim_env, robot_env=robot_env, weights_path=args.weights,
                    max_steps=args.steps, seed=args.seed,
                    hold_steps=args.hold_steps, render=render,
                    archive=not args.no_archive, deadline_s=args.deadline,
                    odom_scale=args.odom_scale, speed_scale=args.speed_scale,
                    corner3_slow=args.corner3_slow, corner2_slow=args.corner2_slow,
                    goal_slow=args.goal_slow, wedge_guard=not args.no_wedge_guard,
                    relocate=not args.no_relocate,
                    corner_radius=args.corner_radius, brake=args.brake,
                    controller=args.controller,
                )

    if not args.no_export:
        csv_path, verdict_path = write_csv(result)
        figure = plot_run(result)
        print(f"wrote {csv_path}")
        print(f"wrote {verdict_path}")
        print(f"wrote {figure}")
    return result


if __name__ == "__main__":
    main()
