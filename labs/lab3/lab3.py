"""Lab 3 mission runner: optimal A*, Lab 2 localisation and path following.

The global layer plans a shortest route through the official plate maze. The
lower layer deliberately reuses the measured Lab 2 EKF, PID and
``WaypointFollower`` rather than introducing another controller. A run records
the simulator truth and the EKF estimate in the four columns required by the
Lab 3 automarker.

Examples:

    python labs/lab3/lab3.py --sim --no-render
    python labs/lab3/lab3.py                 # real robot, with aim handshake
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from sphero_env.envs import SpheroEnv
from sphero_env.envs.custom_maze_full import (
    GRID_RESOLUTION,
    MAZE_HEIGHT,
    MAZE_WIDTH,
    build_occupancy_grid,
    cell_to_world,
    goal_world,
    plot_custom_maze,
    start_world,
)
from sphero_unsw.sphero_edu import SpheroEduAPI

try:
    from Planner import Planner
except ImportError:  # package import, e.g. ``from labs.lab3 import lab3``
    from .Planner import Planner


HERE = Path(__file__).resolve().parent
LABS_DIR = HERE.parent
LAB2_DIR = LABS_DIR / "lab2"
for dependency_dir in (LABS_DIR, LAB2_DIR):
    if str(dependency_dir) not in sys.path:
        # Append so a broken/new lab cannot shadow the dashboard's Lab 1 code.
        sys.path.append(str(dependency_dir))

try:
    import lab2 as lab2_stack
    from controller import WaypointFollower
    from EKF import EKF, dynamics, set_dt, wrap_angle
    from labcore import runlog
except Exception as exc:  # noqa: BLE001 - give one useful import error
    raise ImportError(
        "Lab 3 reuses labs/lab2/{lab2,controller,EKF}.py; "
        f"that stack could not be imported: {exc}"
    ) from exc


STUDENT_ID = "35301856"
MAP = build_occupancy_grid()
START = start_world().astype(float)
GOAL = goal_world().astype(float)
WORLD_SIZE = 1.25

SIM_DT = 0.1
REAL_DT = 0.68
MAX_STEPS = 500
GOAL_TOLERANCE = 0.10
PATH_TOLERANCE = 0.20
# The goal gets the same lead as every other waypoint, and for the same
# reason: the arrival latch fires about 2 cm out on hardware.
#
# It used to be 0.08, set when a run stopped at the entry edge of the goal
# plate. That was over-correction, and it shows: run 20260911T022627 finished
# at (0.556, 0.503) against a (0.5, 0.5) goal - 5.6 cm PAST plate centre,
# almost exactly the 8 cm lead minus the 2 cm latch. The ball was not failing
# to reach the centre, it was being aimed past it.
# The goal takes NO lead. Leading it fought the goal creep, which aims at the
# true plate centre: the follower drove at 0.520 while the creep pulled toward
# 0.500, and run 20260911T024954 settled around x 0.53, past centre.
REAL_GOAL_LEAD_M = 0.0
# A waypoint counts as reached only inside the follower's corner capture
# radius. Used when a replan rebuilds the route: see the drop loop in
# run_mission for what a looser value did to the 11 Sep hardware runs.
WAYPOINT_REACHED_M = 0.05
# Median distance still to run when the follower left a corner, measured over
# every corner of every real run on 11 Sep. See real_drive_waypoints.
# Re-measured after cruise dropped to 0.120 m/s and dt to 0.21. With a 0.02
# lead the corners are now crossed 1.3-2.1 cm PAST centre (run
# 20260911T024954), where at 0.160 m/s and dt 0.68 the same lead left them
# 1.8-2.2 cm short. Less cruise means less coast, and arrive_radius fell from
# 0.041 to 0.030 with dt, so most of the bias the lead was cancelling is gone.
CORNER_LEAD_M = 0.005


def _path_heading_into(path, point):
    """Unit direction of the route arriving AT ``point``."""

    path = [np.asarray(q, dtype=float)[:2] for q in path]
    point = np.asarray(point, dtype=float)[:2]
    if len(path) < 2:
        return None
    index = min(range(len(path)), key=lambda i: float(np.linalg.norm(path[i] - point)))
    if index == 0:
        index = 1
    incoming = path[index] - path[index - 1]
    length = float(np.linalg.norm(incoming))
    return None if length < 1e-9 else incoming / length


def _waypoint_is_behind(waypoint, path, position) -> bool:
    """True when the ball is already PAST ``waypoint`` along its INCOMING edge.

    A plain radius cannot decide whether a replan should keep a corner: 5 cm
    short of it and 5 cm beyond it look identical, and both values tried have
    failed in opposite ways (0.064 m deleted a corner the ball was 0.054 m
    short of; 0.05 m kept one the 12 cm coast dead band made unreachable).

    The projection must be taken against the edge the ball ARRIVES on, not
    the one it leaves on. Projecting onto the outgoing edge is what broke run
    20260911T022627: at the (-0.25, 0.52) corner the outgoing edge runs
    almost pure +x, so a ball at (-0.239, 0.372) - 14.8 cm short in y, but
    1.1 cm past in x - projected positive and the corner was deleted. The
    follower turned a full cell early, the commanded heading disagreed with
    the ball's actual course by 45 degrees for four powered steps, and the
    ball wedged against a wall at (-0.165, 0.45) for 17 steps.

    Against the incoming edge the same position projects to -0.148 and the
    corner is correctly kept.
    """

    heading = _path_heading_into(path, waypoint)
    if heading is None:
        return False
    offset = np.asarray(position, dtype=float)[:2] - np.asarray(waypoint, dtype=float)[:2]
    return float(np.dot(offset, heading)) > 0.0


CSV_COLUMNS = ["sim_x", "sim_y", "real_x", "real_y"]

LOGS = HERE / "logs"
EXPORTS = HERE / "exports"
PLOTS = HERE / "plots"


def goal_heading(start, goal) -> float:
    """Sphero convention: heading 0 is +y and +pi/2 is +x."""

    dx = float(goal[0]) - float(start[0])
    dy = float(goal[1]) - float(start[1])
    return math.atan2(dx, dy)


def robot_to_maze_state(state, start=START, heading_offset=0.0):
    """Transform odometry-frame state into the maze frame.

    Sphero headings increase from +y toward +x, so the corresponding planar
    rotation is clockwise. With ``heading_offset=pi/2``, local forward (+y)
    therefore maps to maze +x.
    """

    state = np.asarray(state, dtype=float).reshape(-1)
    if state.size < 4:
        raise ValueError("state needs [x, y, heading, speed]")
    cosine = math.cos(float(heading_offset))
    sine = math.sin(float(heading_offset))
    x_local, y_local = float(state[0]), float(state[1])
    x_maze = float(start[0]) + cosine * x_local + sine * y_local
    y_maze = float(start[1]) - sine * x_local + cosine * y_local
    return np.array([
        x_maze,
        y_maze,
        wrap_angle(float(state[2]) + float(heading_offset)),
        float(state[3]),
    ])


def robot_observation_to_maze(observation, start=START, heading_offset=0.0):
    observation = np.asarray(observation, dtype=float).reshape(-1)
    transformed = robot_to_maze_state(observation[:4], start, heading_offset)
    if observation.size > 4:
        transformed = np.append(transformed, observation[4])
    return transformed


def maze_action_to_robot(action, heading_offset=0.0):
    """Keep speed but express an absolute maze heading in robot odometry."""

    action = np.asarray(action, dtype=float).reshape(-1)
    return np.array([action[0], wrap_angle(action[1] - heading_offset)], dtype=float)


def make_sim_env(dt=SIM_DT, render=True, max_steps=MAX_STEPS):
    """Official maze simulator driven by the measured Lab 1 lag model."""

    set_dt(dt)
    # EKF.dynamics keeps its historical two-argument marker signature by using
    # a module-level dt. A Robot + sim run changes that value to the measured
    # Bluetooth period, which used to make the blue shadow jump at the real
    # robot's coarse timestep and collide with walls. Bind this simulator to
    # its own clock so the real filter cannot change the plotted simulation.
    dynamics_dt = [float(dt)]

    def simulator_dynamics(state, action):
        return dynamics(state, action, dt=dynamics_dt[0])

    env = SpheroEnv(
        dt=float(dt),
        max_steps=int(max_steps),
        vel_limit=0.15,
        world_width=WORLD_SIZE,
        world_height=WORLD_SIZE,
        goal_pos=GOAL,
        goal_tolerance=GOAL_TOLERANCE,
        occupancy_grid=MAP,
        grid_resolution=GRID_RESOLUTION,
        dynamics=simulator_dynamics,
        obs_noise_std_pos=0.05,
        process_noise_std_speed=0.005,
        process_noise_std_heading=0.01,
        obs_noise_std_vel=0.025,
        render_mode="human" if render else None,
        window_size=(800, 800),
        initial_heading=0.0,
    )
    env._lab3_dynamics_dt = dynamics_dt
    return env


def _set_sim_dt(env, dt):
    """Set both clocks owned by the Lab 3 simulator."""

    value = float(dt)
    env.dt = value
    env._lab3_dynamics_dt[0] = value


def make_real_env(api, dt=REAL_DT):
    """Calibrated real environment with synthetic sensor noise disabled."""

    # Robot odometry is local to the reset pose, not centred in the maze. A
    # 1.25 m observation box therefore clips a valid bottom-left-to-top-right
    # journey at +/-0.625 m. The latest physical run reached 1.67 m in the raw
    # locator while the controller saw the clipped 0.625 m value and continued
    # driving. Keep a larger local sensor box; maze bounds remain in the
    # planner and simulator.
    env = lab2_stack.make_real_env(api, dt=float(dt), vel_limit=0.6, world=5.0)
    env.goal_pos = GOAL.copy()
    env.goal_tolerance = GOAL_TOLERANCE
    return env


@contextmanager
def managed_sim_env(dt=SIM_DT, render=True, max_steps=MAX_STEPS):
    env = make_sim_env(dt=dt, render=render, max_steps=max_steps)
    LOGS.mkdir(parents=True, exist_ok=True)
    env.set_log_path(str(LOGS / "lab3_sim.csv"))
    env.start_logging()
    try:
        yield env
    finally:
        env.stop_logging()
        env.emergency_stop()
        env.close()


@contextmanager
def managed_robot_env(
    *,
    dt=REAL_DT,
    toy_name=None,
    aim=True,
    first_waypoint_local=None,
    aim_fn=None,
    power_scale=lab2_stack.LAB2_PHYSICAL_MAX_SPEED,
):
    """Connect, aim and configure hardware on the same BLE session."""

    selected_toy = lab2_stack.connect_named(toy_name)
    print("Connected.")
    with SpheroEduAPI(selected_toy) as api:
        if aim:
            (aim_fn or lab2_stack.aim_drive_unit)(api, first_waypoint_local)
        env = make_real_env(api, dt=dt)
        lab2_stack.apply_robot_config(
            env,
            auto_aim=False,
            power_scale=power_scale,
        )
        LOGS.mkdir(parents=True, exist_ok=True)
        env.set_log_path(str(LOGS / "lab3_robot.csv"))
        env.reset()
        env.start_logging()
        try:
            yield env
        finally:
            env.stop_logging()
            env.emergency_stop()
            env.close()


def normalise_start(start):
    """Snap a world position to the nearest physical 25 cm maze plate."""

    point = np.asarray(start, dtype=float).reshape(-1)
    if point.size < 2 or not np.all(np.isfinite(point[:2])):
        raise ValueError("start must provide finite x and y coordinates")
    half_width = MAZE_WIDTH * GRID_RESOLUTION
    half_height = MAZE_HEIGHT * GRID_RESOLUTION
    if abs(float(point[0])) > half_width or abs(float(point[1])) > half_height:
        raise ValueError("start is outside the physical maze")

    plate_size = 2.0 * GRID_RESOLUTION
    cell_x = int(np.rint(float(point[0]) / plate_size + (MAZE_WIDTH - 1) / 2.0))
    cell_y = int(np.rint((MAZE_HEIGHT - 1) / 2.0 - float(point[1]) / plate_size))
    if not (0 <= cell_x < MAZE_WIDTH and 0 <= cell_y < MAZE_HEIGHT):
        raise ValueError("start does not map to a maze plate")
    return cell_to_world((cell_x, cell_y)).astype(float)


def place_simulator_at_start(env, seed=0, start=START):
    """Reset, then place truth and odometry at the selected maze start."""

    env.reset(seed=int(seed))
    selected_start = normalise_start(start)
    state = np.array(
        [selected_start[0], selected_start[1], 0.0, 0.0], dtype=np.float32,
    )
    env.state_true = state.copy()
    env.state_odom = state.copy()
    env.last_collision_flag = 0.0
    observation = env._get_observation()
    env.last_obs = observation.copy()
    return observation


def simplify_waypoints(planner, full_path):
    simplify = getattr(planner, "simplify_waypoints", Planner.simplify_waypoints)
    return simplify(full_path)


def plan_route(planner_class=Planner, dt=SIM_DT, state=None, goal=GOAL):
    state = np.array([START[0], START[1], 0.0, 0.0]) if state is None else state
    planner = planner_class(MAP, dt=float(dt))
    full_path = planner.plan(state, goal)
    if not full_path:
        raise RuntimeError("A* found no path through the official maze")
    control_path = simplify_waypoints(planner, full_path)
    # The starting point is not a target to drive to.
    while control_path and np.linalg.norm(np.asarray(control_path[0]) - np.asarray(state[:2])) \
            <= GRID_RESOLUTION * 0.51:
        control_path = control_path[1:]
    if not control_path:
        control_path = [np.asarray(goal, dtype=float)]
    return planner, [np.asarray(point, dtype=float) for point in full_path], control_path


def point_to_path_distance(point, path):
    point = np.asarray(point, dtype=float)[:2]
    path = np.asarray(path, dtype=float).reshape(-1, 2)
    if len(path) == 1:
        return float(np.linalg.norm(point - path[0]))
    best = math.inf
    for start, end in zip(path, path[1:]):
        segment = end - start
        denominator = float(segment @ segment)
        fraction = 0.0 if denominator < 1e-12 else float(
            np.clip((point - start) @ segment / denominator, 0.0, 1.0)
        )
        best = min(best, float(np.linalg.norm(point - (start + fraction * segment))))
    return best


def trajectory_metrics(records, optimal_path):
    """Return the four Lab 3 checks plus conservative path diagnostics.

    The brief does not publish how its single path-distance number is reduced
    over a trajectory. We therefore pass/fail on the maximum deviation. If
    every sample is within 0.20 m, mean, RMS and percentile reductions also
    pass.
    """

    if not records:
        return {"pass": False, "error": "empty trajectory", "checks": {}}

    sim = np.array([[row["sim_x"], row["sim_y"]] for row in records], dtype=float)
    real = np.array([[row["real_x"], row["real_y"]] for row in records], dtype=float)
    sim_deviation = np.array([point_to_path_distance(point, optimal_path) for point in sim])
    real_deviation = np.array([point_to_path_distance(point, optimal_path) for point in real])

    final_sim = float(np.linalg.norm(sim[-1] - GOAL))
    final_real = float(np.linalg.norm(real[-1] - GOAL))
    max_sim = float(np.max(sim_deviation))
    max_real = float(np.max(real_deviation))

    checks = {
        "final_distance_real": {
            "value": final_real, "threshold": GOAL_TOLERANCE,
            "pass": bool(final_real <= GOAL_TOLERANCE),
        },
        "distance_to_optimal_path_real": {
            "value": max_real, "threshold": PATH_TOLERANCE,
            "pass": bool(max_real <= PATH_TOLERANCE),
        },
        "final_distance_sim": {
            "value": final_sim, "threshold": GOAL_TOLERANCE,
            "pass": bool(final_sim <= GOAL_TOLERANCE),
        },
        "distance_to_optimal_path_sim": {
            "value": max_sim, "threshold": PATH_TOLERANCE,
            "pass": bool(max_sim <= PATH_TOLERANCE),
        },
    }
    return {
        "pass": all(item["pass"] for item in checks.values()),
        "checks": checks,
        "path_metric": "maximum point-to-optimal-polyline distance (conservative)",
        "mean_deviation_sim": float(np.mean(sim_deviation)),
        "mean_deviation_real": float(np.mean(real_deviation)),
        "rms_deviation_sim": float(np.sqrt(np.mean(sim_deviation ** 2))),
        "rms_deviation_real": float(np.sqrt(np.mean(real_deviation ** 2))),
    }


def make_follower(waypoints, dt, real, start=START):
    # Slow the hardware approach down. make_controller ignores the vel_limit
    # argument on hardware - it overrides max_speed with LAG_PARAMS vel_limit,
    # 0.160 m/s - so the cap has to be passed explicitly to have any effect.
    #
    # Overshoot is now the binding failure, not the early turn. On run
    # 20260911T023721 the ball crossed the third corner 6 cm PAST it and
    # finished at (0.603, 0.438), 0.120 m from the goal and 8 cm beyond the
    # commanded target. Stopping distance is 0.12 + 0.75*v, so dropping cruise
    # from 0.160 to 0.120 m/s takes it from 0.240 m to 0.210 m, and cuts the
    # speed being carried into each corner by a quarter.
    controller = lab2_stack.make_controller(
        float(dt), 0.6 if real else 0.15, real=real,
        max_speed_ms=0.12 if real else None,
    )
    # Staff advice: keep feedback active inside a goal deadzone so a bulged goal
    # tile cannot roll the ball away after an arrival latch turns the controller off.
    controller.hold_at_goal = False
    follower = WaypointFollower(
        waypoints,
        controller,
        capture_radius=0.05,
        corner_capture=0.05,
        settle_steps=2,
        sharp=True,
        turn_steps=2,
        stop_speed_eps=0.025,
        settle_max=8,
        follow_path=True,
        lookahead=0.10,
    )
    selected_start = normalise_start(start)
    follower.reset(start=np.array([
        selected_start[0], selected_start[1], 0.0, 0.0,
    ]))
    return follower


def real_drive_waypoints(waypoints, full_path):
    """Lead every hardware target slightly along its incoming edge.

    The ball turns early at every corner, and always has. Measured at each
    waypoint switch across the sixteen real runs of 11 Sep, the distance
    still to run when the follower moved on was 0.003-0.031 m with a median
    of 0.020 m. It never once reached a corner before turning, at any q, with
    or without replanning.

    The cause is in the PID, not the planner:

        arrive_radius = max(self.arrive_radius, 1.5 * min_speed * dt)

    which is 0.030 m in simulation at dt 0.1 and 0.041 m on hardware at
    dt 0.68. The latch is therefore dt-driven, and hardware latches roughly
    4 cm out by construction. That is also why simulation never shows this.

    Tightening the latch is not an option - controller.py documents at length
    how a capture radius inside the 12 cm coast dead band deadlocks the ball
    - so the bias is cancelled where it is cheap to cancel. Pushing each
    corner 2 cm further along the edge the ball arrives on means latching
    2 cm early puts it on the true corner, not short of it.

    It matters more here than the numbers suggest. Corridors are one 0.125 m
    cell wide and the ball is 0.073 m across, leaving 2.6 cm of clearance per
    side, so a 2 cm early turn clips the inside corner by most of the
    available room. 2 cm also stays well inside the corner's own cell, so the
    shifted target is never placed in a wall.
    """

    adjusted = [np.asarray(point, dtype=float).copy() for point in waypoints]
    path = [np.asarray(point, dtype=float) for point in full_path]
    if not adjusted or len(path) < 2:
        return adjusted

    def _lead(point, distance):
        """Shift ``point`` along the direction it is approached from."""

        index = min(range(len(path)),
                    key=lambda i: float(np.linalg.norm(path[i] - point)))
        if index == 0:
            return point
        incoming = path[index] - path[index - 1]
        length = float(np.linalg.norm(incoming))
        if length <= 1e-9:
            return point
        return point + distance * incoming / length

    for i in range(len(adjusted) - 1):
        adjusted[i] = _lead(adjusted[i], CORNER_LEAD_M)
    adjusted[-1] = _lead(np.asarray(path[-1], dtype=float), REAL_GOAL_LEAD_M)
    return adjusted


def _record(step, elapsed, dt, sim_xy, estimate, action, follower, collision,
            replans, path_deviation):
    target = follower.target
    return {
        "step": int(step),
        "t": float(elapsed),
        "dt": float(dt),
        "sim_x": float(sim_xy[0]),
        "sim_y": float(sim_xy[1]),
        "real_x": float(estimate[0]),
        "real_y": float(estimate[1]),
        "heading": float(estimate[2]),
        "speed": float(estimate[3]),
        "speed_cmd": float(action[0]),
        "heading_cmd": float(action[1]),
        "waypoint_index": int(follower.index),
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "dist_to_goal": float(np.linalg.norm(np.asarray(estimate[:2]) - GOAL)),
        "path_deviation": float(path_deviation),
        "collision": bool(collision),
        "replans": int(replans),
    }


def run_mission(
    sim_env,
    *,
    robot_env=None,
    planner_class=Planner,
    start=None,
    max_steps=MAX_STEPS,
    seed=0,
    hold_steps=12,
    replan_deviation=0.18,
    replan_confirm_steps=3,
    replan_cooldown=12,
    render=True,
    on_step=None,
    should_stop=None,
    on_env=None,
    verbose=True,
    archive=True,
):
    """Plan, drive, localise, replan if needed, and return a run report."""

    real = robot_env is not None
    dt = float(robot_env.dt if real else sim_env.dt)
    set_dt(dt)
    selected_start = normalise_start(START if start is None else start)
    start_state = np.array([
        selected_start[0], selected_start[1], 0.0, 0.0,
    ])
    sim_observation = place_simulator_at_start(
        sim_env, seed=seed, start=selected_start,
    )
    planner, optimal_path, control_path = plan_route(
        planner_class, dt=dt, state=start_state,
    )
    first_heading = goal_heading(selected_start, control_path[0])

    ekf = EKF(
        dt=dt,
        # q=1 with the full measurement is the configuration behind every
        # good hardware run recorded on 11 Sep: 0.021/0.018, 0.010/0.023,
        # 0.024/0.029 and 0.021/0.036 (final distance / distance to optimal).
        # Raising it did not help. q=128 produced 0.134/0.216 and q=512
        # produced 0.094/0.181 and 0.832/0.022 - the three worst results of
        # the day.
        #
        # The argument for raising q was that a high q pins the estimate to a
        # frozen locator when the ball wedges, instead of letting the process
        # model walk it along the planned route. That symptom is real, but the
        # cure cost more than the disease, and the plausibility gate below now
        # handles the locator resyncs independently of q.
        #
        # A lateral-bias theory was also floated: that high q rode the ball
        # closer to a corridor wall. Measured across all 17 real runs it does
        # not hold - bias on the x=-0.25 corridor spans 0.01-2.28 cm at q=1
        # and 0.96-1.59 cm at q=128/512, so it tracks aim and slip, not q.
        q_scale=1.0,
        r_scale=1.0,
        measurement="full",
        hardware=real,
    )
    initial_state = np.array([
        selected_start[0], selected_start[1], first_heading, 0.0,
    ])
    ekf.reset(initial_state)
    drive_path = (real_drive_waypoints(control_path, optimal_path)
                  if real else control_path)
    follower = make_follower(drive_path, dt, real, start=selected_start)
    shadow_follower = None
    shadow_time = 0.0
    if real:
        # The blue line is an independent simulation of the same A* route, not
        # the physical controller's commands replayed on a different plant.
        # Replaying real commands made the simulation turn when the robot did,
        # even when the simulated ball had not reached that corner yet.
        _set_sim_dt(sim_env, SIM_DT)
        shadow_follower = make_follower(
            control_path, SIM_DT, False, start=selected_start,
        )

    if real:
        robot_observation = np.asarray(robot_env.last_obs, dtype=float)
        # The aim step makes robot heading 0 point along the first planned edge.
        maze_heading_offset = first_heading
        transformed = robot_observation_to_maze(
            robot_observation, selected_start, maze_heading_offset,
        )
        ekf.update(transformed)
    else:
        maze_heading_offset = 0.0
        ekf.update(sim_observation)

    if on_env is not None:
        on_env(robot_env if real else sim_env)

    if verbose:
        turns = [[round(float(v), 3) for v in point] for point in control_path]
        print(f"A*: {len(optimal_path)} grid nodes, "
              f"{(len(optimal_path) - 1) * GRID_RESOLUTION:.3f} m, "
              f"{len(control_path)} controller waypoints")
        print(f"waypoints: {turns}")

    records = []
    replans = 0
    collisions = 0
    off_path_steps = 0
    last_replan_step = -int(replan_cooldown)
    goal_hold = 0
    stopped = False
    start_clock = time.perf_counter()
    last_clock = start_clock
    period_samples = []
    dt_settled = not real

    # Automatic aim correction, learned from the first genuinely moving robot
    # samples exactly as in Lab 2. It corrects motor heading only; the maze
    # frame remains anchored at the known start pose.
    previous_robot_xy = None
    heading_samples = []
    heading_corrected = False
    last_driven_heading = None
    # Wheel odometry is dead reckoning with no external reference, and on the
    # lab surface it fails in two opposite ways that both corrupt the EKF.
    #
    # JAM: the ball wedges against a wall, the motors keep turning, and the
    # locator either freezes or keeps integrating. Measured on the 11 Sep
    # official run, steps 86-105: the estimate sat at (-0.017, 0.510) for 20
    # consecutive steps with speed_cmd 0.040 held on, and `collision` was
    # logged 0 for the whole run. Nothing in the stack noticed.
    #
    # TELEPORT: the locator then resynchronises in one sample. Same run, step
    # 107 -> 108, the reported position moved 0.315 m in a single 0.22 s step.
    # The ball's own ceiling is vel_limit * dt = 0.035 m, so that jump is a
    # factor of nine beyond anything physical, and it is what carried the run
    # 13 cm past the goal plate.
    previous_measurement_xy = None
    jam_steps = 0
    rejected_measurements = 0
    consecutive_rejections = 0
    #: Consecutive rejections before the gate concedes the ball really has
    #: moved. Three samples is ~2 s of BLE, long enough to outlast a single
    #: locator resync and short enough not to strand the filter.
    REANCHOR_AFTER_REJECTIONS = 3
    #: A step is "jammed" when the drive is commanded on but the reported
    #: position has barely moved. 5 mm is comfortably above locator noise
    #: (millimetre scale on the logs) and well below one driven step.
    JAM_MOVE_EPS = 0.005
    #: Consecutive jammed steps before the route is replanned. The two real
    #: jams above ran 12 and 20 steps, so 8 fires inside both while staying
    #: clear of the 2-step square-corner pivot, which commands zero speed and
    #: is therefore never counted here in the first place.
    JAM_CONFIRM_STEPS = 8
    # --- goal creep ---------------------------------------------------- #
    # The ball settles a few centimetres off plate centre and stays there.
    # Measured on run 20260911T024251, the last 18 steps: y held 0.499 against
    # a 0.500 goal, but x sat at 0.467 in a bang-bang limit cycle - one 0.0400
    # command nudged it +3 mm, the next two steps commanded zero and it rolled
    # back 3-4 mm down the lip on the start plate. Net zero, indefinitely.
    #
    # 0.0400 is already min_speed, i.e. 0.116 m/s, so the pulse is not weak -
    # it is SHORT. Speed lag is tau 0.51 s and a step is 0.21 s, so one pulse
    # reaches barely a third of the commanded speed before the envelope cuts
    # it again. Three consecutive steps is ~1.2 tau, enough to actually roll
    # the 3 cm.
    #
    # A global ki_pos would not do this. integral_limit is 0.5 and the term is
    # reset per leg, so on a 0.75 m leg the integral saturates within about
    # four steps: ki_pos 0.1 would then add +0.05 to a command band of
    # 0.040-0.054, doubling the drive on every leg, permanently. The reset
    # that would drain it (controller.py:237) sits behind hold_at_goal, which
    # Lab 3 sets False. Hence a bounded nudge near the goal instead of an
    # integral everywhere - the deadzone-with-live-controller shape Michael
    # describes, rather than constant reactive control.
    GOAL_CREEP_RADIUS = 0.12
    GOAL_CREEP_SETTLED = 0.02
    GOAL_CREEP_STALL = 4
    GOAL_CREEP_STEPS = 3
    goal_creep_stalled = 0
    goal_creep_owed = 0
    goal_creep_nudges = 0
    previous_goal_distance = None
    #: A confirmed jam is RECORDED but not driven out of.
    #:
    #: A reverse-thrust escape was tried and removed: fighting a wedge after
    #: the fact treats the symptom, and the run is already off its optimal
    #: path by then. The counter stays because it is the only instrument that
    #: sees a wedge at all - before it existed, a run that spent 99 of 178
    #: steps against a wall still reported 0 collisions.
    #: Reject a measurement that implies more travel than the ball can do.
    #: 1.6x the kinematic ceiling plus 2 cm. At dt 0.68 that is 0.194 m,
    #: which passes every honest sample seen on hardware (largest legitimate
    #: step measured: 0.056 m) and rejects both observed resyncs: 0.315 m on
    #: run 20260911T013041Z and 0.453 m on run 20260911T015444Z. The earlier
    #: 3x/3 cm ceiling was 0.357 m and let the second one through.
    def _measurement_is_plausible(previous, current, step_dt):
        if previous is None:
            return True
        # A floor, because the ceiling cannot be dt alone. At dt 0.68 the
        # kinematic term dominates; at dt 0.21 it collapses to 0.074 m, but
        # BLE does not deliver one sample per loop at that rate - it batches,
        # and a single sample then covers several periods of honest travel.
        # Measured on run 20260911T023721 (dt 0.21): median step 0.016 m,
        # p90 0.056 m, max 0.104 m, and the gate threw away 19 of 102
        # measurements, leaving the estimate on the process model a fifth of
        # the time. 0.15 m passes all of that and still rejects both real
        # resyncs, 0.315 m and 0.453 m.
        ceiling = max(
            1.6 * float(lab2_stack.LAG_PARAMS["vel_limit"]) * float(step_dt) + 0.02,
            0.15,
        )
        return float(np.linalg.norm(current - previous)) <= ceiling

    start_row = _record(
        0, 0.0, dt, selected_start, ekf.state_est,
        np.array([0.0, first_heading]), follower, False, 0, 0.0,
    )
    records.append(start_row)
    if on_step is not None:
        on_step(start_row)

    for step in range(1, int(max_steps) + 1):
        if should_stop is not None and should_stop():
            stopped = True
            break

        now = time.perf_counter()
        wall_dt = now - last_clock
        last_clock = now
        if real and step > 1 and not dt_settled:
            period_samples.append(wall_dt)
            if len(period_samples) >= 8:
                measured_dt = float(np.median(period_samples))
                if 0.05 < measured_dt < 2.0:
                    if abs(measured_dt - dt) > 0.05:
                        dt = measured_dt
                        set_dt(dt)
                        ekf.dt = dt
                        follower.controller.dt = dt
                        if verbose:
                            print(f"measured loop period {dt:.3f} s; "
                                  "model, EKF and controller updated")
                    dt_settled = True

        action_maze, target, _distance = follower(ekf.state_est)

        # Creep the last few centimetres onto plate centre. Confined to the
        # goal neighbourhood and to the case where the ball has stopped
        # converging, so no leg can ever see it.
        if real and follower._is_final():
            goal_error = GOAL - np.asarray(ekf.state_est[:2], dtype=float)
            goal_distance = float(np.linalg.norm(goal_error))
            # Progress is measured against the CLOSEST the ball has been, not
            # against the previous step. The limit cycle oscillates +/-3.5 mm
            # with a three step period, so a step-to-step test sees the ball
            # "converging" every third step and resets the counter forever:
            # replayed against run 20260911T024251 it fired zero nudges in
            # eighteen stalled steps.
            converging = (previous_goal_distance is None
                          or goal_distance < previous_goal_distance - 0.003)
            if GOAL_CREEP_SETTLED < goal_distance <= GOAL_CREEP_RADIUS:
                # Do not accumulate stall while a nudge is still being
                # delivered, or the counter re-arms mid-nudge and the nudges
                # run back to back. This holds the minimum period at
                # GOAL_CREEP_STEPS + GOAL_CREEP_STALL steps.
                if goal_creep_owed > 0:
                    goal_creep_stalled = 0
                else:
                    goal_creep_stalled = 0 if converging else goal_creep_stalled + 1
                if goal_creep_owed <= 0 and goal_creep_stalled >= GOAL_CREEP_STALL:
                    goal_creep_stalled = 0
                    goal_creep_owed = GOAL_CREEP_STEPS
                    goal_creep_nudges += 1
            else:
                goal_creep_stalled = 0
                goal_creep_owed = 0
                previous_goal_distance = None
            previous_goal_distance = (goal_distance if converging
                                      else previous_goal_distance)
            if goal_creep_owed > 0:
                goal_creep_owed -= 1
                # Aim at the TRUE plate centre, not the led target.
                action_maze = np.array([
                    lab2_stack.LAG_PARAMS["deadband"] + 0.04,
                    goal_heading(ekf.state_est, GOAL),
                ], dtype=float)
        sim_env.set_current_setpoint(target)

        # Do not spin the drive unit at zero speed except during an intentional
        # square-corner pivot.
        if getattr(follower, "turning", 0):
            last_driven_heading = float(action_maze[1])
        elif abs(float(action_maze[0])) < 1e-6 and last_driven_heading is not None:
            action_maze = np.array([action_maze[0], last_driven_heading])
        elif abs(float(action_maze[0])) >= 1e-6:
            last_driven_heading = float(action_maze[1])

        ekf.predict(action_maze)

        if should_stop is not None and should_stop():
            stopped = True
            break
        robot_info = {}
        if real:
            if should_stop is not None and should_stop():
                stopped = True
                break
            action_robot = maze_action_to_robot(action_maze, maze_heading_offset)
            robot_observation, _rr, robot_terminated, robot_truncated, robot_info = \
                robot_env.step(action_robot)
            transformed = robot_observation_to_maze(
                robot_observation, selected_start, maze_heading_offset,
            )
            measured_xy = np.asarray(transformed[:2], dtype=float)
            if _measurement_is_plausible(previous_measurement_xy, measured_xy, dt):
                ekf.update(transformed)
                moved = (0.0 if previous_measurement_xy is None
                         else float(np.linalg.norm(measured_xy - previous_measurement_xy)))
                previous_measurement_xy = measured_xy
                consecutive_rejections = 0
                # Only a step that actually asked the ball to move can be a
                # jam. The square-corner pivot commands zero speed on purpose.
                if abs(float(action_maze[0])) >= 1e-6 and moved < JAM_MOVE_EPS:
                    jam_steps += 1
                else:
                    jam_steps = 0
            else:
                # Skip the update rather than the predict: the process model
                # still carries the state forward, so one bad sample costs
                # accuracy instead of teleporting the estimate.
                #
                # Do NOT re-anchor on the rejected reading. Doing that made the
                # gate a one-step delay and nothing more: on run
                # 20260911T015444Z the locator jumped 0.45 m at step 105, the
                # gate rejected it and moved the anchor there, and step 106's
                # reading sat 0.006 m from the new anchor - so it was accepted
                # and the estimate teleported anyway, one step later than
                # before. The anchor stays where the filter believes the ball
                # is, so a jump has to be justified against reality.
                rejected_measurements += 1
                consecutive_rejections += 1
                if consecutive_rejections >= REANCHOR_AFTER_REJECTIONS:
                    # A relocation the locator insists on for several samples
                    # in a row is more likely real than a one-off resync, and
                    # refusing it forever would strand the filter. Accept it.
                    previous_measurement_xy = measured_xy
                    consecutive_rejections = 0
                jam_steps = 0
                message = (f"rejected implausible odometry jump at step {step}")
                if verbose:
                    print(f"WARNING: {message}")
                if on_step is not None:
                    on_step({"step": -1, "warning": message})

            # Advance the independent 0.1 s simulation by the real wall-clock
            # time consumed by this robot update. It therefore completes at
            # roughly the same time without inheriting BLE jitter or the real
            # controller's waypoint switches.
            # Two fixed simulator ticks per robot sample is the minimum needed
            # for the 0.1 s controller to finish a route before the faster
            # hardware follower. Wall time is used when a slow BLE round trip
            # represents more simulated time than that.
            shadow_time += max(time.perf_counter() - now, 2.0 * SIM_DT)
            sim_info = {}
            while shadow_time >= SIM_DT and not shadow_follower.finished:
                shadow_state = sim_env.get_true_state()
                shadow_action, shadow_target, _shadow_distance = \
                    shadow_follower(shadow_state)
                sim_env.set_current_setpoint(shadow_target)
                sim_observation, _reward, _terminated, _truncated, sim_info = \
                    sim_env.step(shadow_action)
                shadow_time -= SIM_DT
        else:
            sim_observation, _reward, sim_terminated, sim_truncated, sim_info = \
                sim_env.step(action_maze)
            robot_terminated = robot_truncated = False
            ekf.update(sim_observation)

        sim_xy = sim_env.get_true_state()[:2].astype(float)
        # In Robot mode the simulator is only a shadow trajectory. Its fitted
        # dynamics can run ahead of a slower physical ball and touch a future
        # wall; that is not evidence that the ball collided and must not cause
        # a physical replan. Each mode therefore trusts its own collision
        # source exclusively.
        collision = (bool(robot_info.get("collision")) if real
                     else bool(sim_info.get("collision")))
        collisions += int(collision)

        if real and not heading_corrected:
            robot_xy = np.asarray(robot_env.get_odom_state()[:2], dtype=float)
            if previous_robot_xy is not None:
                displacement = robot_xy - previous_robot_xy
                if float(np.linalg.norm(displacement)) > 0.03:
                    course = math.atan2(float(displacement[0]), float(displacement[1]))
                    command_robot = float(
                        maze_action_to_robot(action_maze, maze_heading_offset)[1]
                    )
                    heading_samples.append(wrap_angle(course - command_robot))
            previous_robot_xy = robot_xy
            if len(heading_samples) >= 5:
                error = float(np.median(heading_samples))
                error_degrees = math.degrees(error)
                heading_corrected = True
                if abs(error_degrees) <= 45.0 and abs(error_degrees) > 1.0:
                    robot_env.heading_offset_deg = (
                        float(getattr(robot_env, "heading_offset_deg", 0.0))
                        - error_degrees
                    )
                    message = f"heading auto-corrected by {-error_degrees:+.1f} deg"
                    if verbose:
                        print(message)
                    if on_step is not None:
                        on_step({"step": -1, "warning": message})
                elif abs(error_degrees) > 45.0:
                    message = (
                        f"measured heading offset {error_degrees:+.0f} deg exceeds "
                        "the 45 deg sanity limit; correction not applied"
                    )
                    if verbose:
                        print(f"WARNING: {message}")
                    if on_step is not None:
                        on_step({"step": -1, "warning": message})

        path_deviation = point_to_path_distance(ekf.state_est[:2], optimal_path)
        if path_deviation > float(replan_deviation):
            off_path_steps += 1
        else:
            off_path_steps = 0

        # A jam is a collision the hardware detector missed. The 11 Sep run
        # logged 0 collisions across two jams of 12 and 20 steps, so treating
        # a confirmed jam as a replan trigger is the only thing that gets the
        # route moving again.
        jammed = jam_steps >= JAM_CONFIRM_STEPS
        may_replan = step - last_replan_step >= int(replan_cooldown)
        needs_replan = (collision or jammed
                        or off_path_steps >= int(replan_confirm_steps))
        if jammed:
            jam_steps = 0
            collisions += 1
        if needs_replan and may_replan and not follower.finished:
            fresh_path = planner.plan(ekf.state_est, GOAL)
            if fresh_path:
                fresh_control = simplify_waypoints(planner, fresh_path)
                # Drop only a waypoint the ball has genuinely REACHED.
                #
                # GRID_RESOLUTION * 0.51 is 0.064 m, and a replan triggered
                # near a corner therefore deletes that corner from up to
                # 6.4 cm away. Measured on the 11 Sep run 20260911T015444Z,
                # step 73: a jam replan fired while the ball sat 0.054 m short
                # of the (-0.25, 0.5) corner, the corner was dropped, and the
                # follower turned straight for the goal from there. The ball
                # entered the +x corridor at y = 0.448 instead of 0.500 - 5 cm
                # below centre - and then ground along the bottom wall for the
                # rest of the route: four further jams of 24, 22 and 32 steps,
                # with y never recovering above 0.480.
                #
                # Every jam in that run descends from that one dropped corner,
                # so the threshold is now the follower's own corner capture.
                # A corner still ahead of the ball stays in the route and gets
                # driven to; only one already satisfied is discarded.
                while fresh_control and (
                    np.linalg.norm(np.asarray(fresh_control[0])
                                   - ekf.state_est[:2]) <= WAYPOINT_REACHED_M
                    or _waypoint_is_behind(
                        fresh_control[0], fresh_path, ekf.state_est,
                    )
                ):
                    fresh_control = fresh_control[1:]
                if fresh_control:
                    fresh_drive = (real_drive_waypoints(fresh_control, fresh_path)
                                   if real else fresh_control)
                    follower = make_follower(
                        fresh_drive, dt, real, start=ekf.state_est[:2],
                    )
                    follower.reset(start=ekf.state_est)
                    replans += 1
                    last_replan_step = step
                    off_path_steps = 0
                    message = (
                        f"replanned at step {step} after "
                        f"{'collision' if collision else f'{path_deviation:.3f} m deviation'}"
                    )
                    if verbose:
                        print(message)
                    if on_step is not None:
                        on_step({"step": -1, "warning": message})

        elapsed = time.perf_counter() - start_clock if real else step * dt
        row = _record(
            step, elapsed, dt, sim_xy, ekf.state_est, action_maze, follower,
            collision, replans, path_deviation,
        )
        records.append(row)
        if on_step is not None:
            on_step(row)

        sim_trace = np.array([[item["sim_x"], item["sim_y"]] for item in records])
        estimate_trace = np.array([[item["real_x"], item["real_y"]] for item in records])
        sim_env.set_belief(ekf.state_est, ekf.P)
        sim_env.set_overlay_trajectories(
            real_traj=estimate_trace,
            odom_traj=sim_trace,
            true_traj=np.asarray(optimal_path),
        )
        if render:
            sim_env.render()

        if follower.finished:
            goal_hold += 1
            if goal_hold >= int(hold_steps):
                break
        else:
            goal_hold = 0

        # Do not stop merely because either environment entered its goal
        # tolerance. The follower owns arrival and deliberately keeps feedback
        # active for ``hold_steps`` so the ball can correct a roll on the
        # bulged goal plate. The explicit step limit remains the safety bound.

    report = trajectory_metrics(records, optimal_path)
    report.update({
        "steps": len(records),
        "elapsed_s": float(records[-1]["t"] if records else 0.0),
        "collisions": int(collisions),
        "replans": int(replans),
        # Odometry health, so a run can be judged on whether the estimate
        # was trustworthy rather than only on where it claimed to finish.
        "rejected_measurements": int(rejected_measurements),
        "goal_creep_nudges": int(goal_creep_nudges),

        "stopped_by_user": bool(stopped),
        "goal_hold_steps": int(goal_hold),
        "planner_nodes": len(optimal_path),
        "optimal_length_m": float((len(optimal_path) - 1) * GRID_RESOLUTION),
        "control_waypoints": [[float(v) for v in point] for point in control_path],
        "real_goal_lead_m": float(REAL_GOAL_LEAD_M if real else 0.0),
    })

    result = {
        "lab": 3,
        "mode": "real" if real else "sim",
        "records": records,
        "report": report,
        "optimal_path": [[float(v) for v in point] for point in optimal_path],
        "waypoints": [[float(v) for v in point] for point in control_path],
        "start": selected_start.tolist(),
        "goal": GOAL.tolist(),
    }

    if archive:
        result["run_dir"] = str(archive_run(result))

    if verbose:
        print(format_report(report))
    return result


def archive_run(result):
    """Store the run and its route plot in a unique archive directory."""

    run_dir = runlog.new_run_dir(result["mode"], "lab3", "astar-lab2-pid")
    writer = runlog.RunWriter(
        run_dir,
        {
            "source": result["mode"],
            "experiment": "lab3",
            "model": "astar-lab2-pid",
            "start_pose": [*result["start"], 0.0, 0.0],
            "goal": GOAL.tolist(),
            "optimal_path": result["optimal_path"],
            "waypoints": result["waypoints"],
        },
    )
    for row in result["records"]:
        writer.append({
            "step": row["step"],
            "t": row["t"],
            "dt": row["dt"],
            "speed_cmd": row["speed_cmd"],
            "heading_cmd": row["heading_cmd"],
            "x": row["sim_x"],
            "y": row["sim_y"],
            "heading": row["heading"],
            "speed": row["speed"],
            "odom_x": row["real_x"],
            "odom_y": row["real_y"],
            "setpoint_x": row["target_x"],
            "setpoint_y": row["target_y"],
            "dist_to_goal": row["dist_to_goal"],
            "collision": row["collision"],
        })
    writer.close(status="stopped" if result["report"]["stopped_by_user"] else "complete",
                 metrics=result["report"])
    plot_run(result, out_path=run_dir / "plots" / "lab3_route.png")
    return run_dir


def write_csv(result, student_id=STUDENT_ID, out_dir=None):
    """Write every recorded step using exactly the four required columns."""

    records = result.get("records") or []
    if not records:
        raise ValueError("run produced no steps")
    out_dir = Path(out_dir) if out_dir else EXPORTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{student_id}_lab3.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    verdict_path = out_dir / f"{student_id}_lab3_verdict.json"
    payload = {
        "csv": str(path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(records),
        "mode": result.get("mode"),
        "report": result.get("report", {}),
        "optimal_path": result.get("optimal_path", []),
    }
    verdict_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path, verdict_path


def plot_run(result, out_path=None):
    """Plot the maze, optimal path, simulation truth and EKF/real estimate."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = result.get("records") or []
    if not records:
        raise ValueError("run produced no steps")
    out_path = Path(out_path) if out_path else PLOTS / f"{STUDENT_ID}_lab3.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sim_path = [[row["sim_x"], row["sim_y"]] for row in records]
    real_path = [[row["real_x"], row["real_y"]] for row in records]

    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    plot_custom_maze(
        axis,
        optimal_path=result.get("optimal_path"),
        sim_path=sim_path,
        real_path=real_path,
        start=result.get("start"),
    )
    report = result.get("report", {})
    axis.text(
        0.01, 0.01,
        f"four checks: {'PASS' if report.get('pass') else 'FAIL'}  |  "
        f"steps: {report.get('steps', '?')}  |  collisions: {report.get('collisions', '?')}  |  "
        f"replans: {report.get('replans', '?')}",
        transform=axis.transAxes,
        fontsize=9,
        color="#27313c",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.92,
                  edgecolor="#c9d1da"),
        zorder=10,
    )
    handles, labels = axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axis.legend(unique.values(), unique.keys(), loc="upper left", framealpha=0.95)
    figure.savefig(out_path, dpi=170)
    plt.close(figure)
    return out_path


def format_report(report):
    labels = (
        ("final_distance_real", "final distance to goal (real/estimate)"),
        ("distance_to_optimal_path_real", "max distance to optimal (real/estimate)"),
        ("final_distance_sim", "final distance to goal (simulation)"),
        ("distance_to_optimal_path_sim", "max distance to optimal (simulation)"),
    )
    lines = ["Lab 3 conservative submission check"]
    for key, label in labels:
        item = report.get("checks", {}).get(key, {})
        value = item.get("value", math.inf)
        threshold = item.get("threshold", math.nan)
        lines.append(
            f"  {'PASS' if item.get('pass') else 'FAIL'}  {label:<44} "
            f"{value:.4f} m  (<= {threshold:.2f})"
        )
    lines.append(
        f"  overall: {'PASS' if report.get('pass') else 'FAIL'} | "
        f"steps={report.get('steps', 0)} collisions={report.get('collisions', 0)} "
        f"replans={report.get('replans', 0)}"
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Lab 3 - optimal A* with the measured Lab 2 navigation stack"
    )
    parser.add_argument("--sim", action="store_true", help="run without BLE hardware")
    parser.add_argument("--no-render", action="store_true", help="disable pygame rendering")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="maximum control steps")
    parser.add_argument("--dt", type=float, default=None, help="override the mode's loop period")
    parser.add_argument("--seed", type=int, default=0, help="simulation noise seed")
    parser.add_argument("--hold-steps", type=int, default=12,
                        help="active feedback steps after automatic arrival")
    parser.add_argument("--replan-deviation", type=float, default=0.18,
                        help="sustained path error that triggers replanning")
    parser.add_argument("--toy", default=None, help="preferred Sphero name")
    parser.add_argument("--power-scale", type=float,
                        default=lab2_stack.LAB2_PHYSICAL_MAX_SPEED,
                        help="raw motor mapping scale; 0.60 is the measured default")
    parser.add_argument("--aim", dest="aim", action="store_true", default=None,
                        help="aim the drive unit before a real run")
    parser.add_argument("--no-aim", dest="aim", action="store_false",
                        help="skip manual aiming")
    parser.add_argument("--no-export", action="store_true", help="do not write submission files")
    parser.add_argument("--no-archive", action="store_true", help="do not add this run to the archive")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dt = float(args.dt if args.dt is not None else (SIM_DT if args.sim else REAL_DT))
    render = not args.no_render
    _planner, _optimal, control = plan_route(dt=dt)
    first_waypoint_local = np.asarray(control[0]) - START

    with managed_sim_env(dt=dt, render=render, max_steps=args.steps) as sim_env:
        if args.sim:
            result = run_mission(
                sim_env,
                max_steps=args.steps,
                seed=args.seed,
                hold_steps=args.hold_steps,
                replan_deviation=args.replan_deviation,
                render=render,
                archive=not args.no_archive,
            )
        else:
            do_aim = True if args.aim is None else bool(args.aim)
            with managed_robot_env(
                dt=dt,
                toy_name=args.toy,
                aim=do_aim,
                first_waypoint_local=first_waypoint_local,
                power_scale=args.power_scale,
            ) as robot_env:
                result = run_mission(
                    sim_env,
                    robot_env=robot_env,
                    max_steps=args.steps,
                    seed=args.seed,
                    hold_steps=args.hold_steps,
                    replan_deviation=args.replan_deviation,
                    render=render,
                    archive=not args.no_archive,
                )

    if not args.no_export:
        csv_path, verdict_path = write_csv(result)
        plot_path = plot_run(result)
        print(f"wrote {csv_path}")
        print(f"wrote {verdict_path}")
        print(f"wrote {plot_path}")
    return result


if __name__ == "__main__":
    main()
