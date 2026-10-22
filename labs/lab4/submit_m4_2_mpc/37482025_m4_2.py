"""M4-2 policy (student 37482025): model-predictive control on a LEARNED
dynamics model, with its own state estimator, speed calibration and
fail-safes, in one self-contained file.

This is the controller used for the Lab 4 robot runs and demo (Ed #140 option
"learn a dynamics model, and use your existing navigation stack to optimise
with respect to this"), packaged so the marker can run it unchanged.

How the marker runs it (Ed #140, #164, #200)::

    policy = Policy()                     # no arguments; loads weight.pth itself
    action = policy(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
    obs, ... = env.step(action)

``obs`` is the raw output of ``SpheroEnv.step`` or ``Robot.step``:
``[x, y, heading, speed]`` with an optional collision flag (length 4 or 5).
The action is ``[speed_cmd, heading_cmd]`` with an absolute heading (0 is +y,
+pi/2 is +x). Only ``math``, ``heapq``, ``os``, ``time``, ``numpy`` and
``torch`` are imported. ``weight.pth`` (found next to this file or in the
working directory) is this module's state_dict: the learned dynamics network
and its input/output normalisation.

What is learned
---------------
* ``DynamicsModel``: a 6-64-64-4 network trained on 230 logged real-robot runs
  (184 train / 46 held out). Inputs, in the commanded-heading frame: forward
  and lateral velocity, the speed command, the previous command resolved into
  this frame (it carries the actuation latency and the turn), and dt.
  Outputs: forward and lateral displacement over the step and the next
  forward and lateral velocity. Six-step rollout error on held-out runs:
  2.1 cm, against 4.8 cm for constant velocity and 5.4 cm for command x dt.
* ``SpeedAdapter``: the ground speed per unit command of whatever plant is
  being driven, learned online.

How an action is chosen (``LearnedMPC``)
----------------------------------------
A* plans the route on the known maze. Each step, 30 candidate commands (six
speeds x five headings around a pure-pursuit aim point) are rolled through
the learned model in one batch: held for 1.2 s to score progress and
cross-track error, and applied once then cut to predict where the ball would
come to rest. Predicted rest points beyond the next corner or the goal, and
any predicted point inside a wall, are priced heavily; the cheapest is sent.
The coast the model learned is what sets the braking.

Around the MPC
--------------
1. Frame: both step functions start the ball at (0, 0); the maze puts the
   start plate at (-0.5, -0.5), so a first observation near the origin is
   shifted onto it.
2. ``StateEKF``: both template environments add 5 cm of noise to every
   position, so the MPC runs on a filtered estimate, never the raw obs.
3. Units: the model speaks the command units of the robot it was trained on.
   Its command is turned into a ground speed through that robot's measured
   motor map and then into this plant's command by ``SpeedAdapter`` (the
   template Robot sends ``cmd / 0.15 * 255`` as motor power, the simulator
   takes m/s). The Robot's speed channel echoes the command; that is detected
   and not trusted.
4. Slow zones from the robot runs: into corner 2 (0.6), into corner 3 (0.8),
   and a ramp to 0.5 over the last 30 cm to the goal; never driving east past
   the goal; a start from rest inside a slow zone gets at least 0.8.
5. Jams: no motion, a collision flag or a stopped estimate under a drive
   command for five steps triggers a short drive back to the route.
6. Goal hold inside 5 cm of the goal on the final leg (the 25 cm plate
   counts), released only past 8.5 cm.
7. Speed calibration on the first straight (a steady target until the gain
   is measured), and on the template Robot never more than raw 77.

Tested in closed loop with this exact calling pattern on the official maze:
the lab simulator, and emulations of the template Robot (its power mapping
and echoed speed channel) and of the calibrated robot, at the template's
0.1 s step and at the 0.22 s a step takes over Bluetooth. Final checks:
35 of 36 runs on the goal plate at step 500 at 0.22 s, 34 of 36 at 0.1 s.
"""

from __future__ import annotations

import heapq
import math
import os
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - the dashboard explains the dependency
    torch = None


# --------------------------------------------------------------------- maze

#: The official 5 x 5 plate maze as the 11 x 11 occupancy grid the environment
#: consumes: 1 is an obstacle, 0 is free, world origin at the centre of the
#: grid.  Held as a literal so the file depends on nothing else.
OCCUPANCY_GRID = (
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    (1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1),
    (1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1),
    (1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1),
    (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1),
    (1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1),
    (1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1),
    (1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1),
    (1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
)

PLATE_SIZE = 0.25
GRID_RESOLUTION = PLATE_SIZE / 2.0
START = (-0.5, -0.5)
GOAL = (0.5, 0.5)

#: Measured on the robot: ball diameter, and the environment's speed ceiling.
BALL_DIAMETER = 0.073
VEL_LIMIT = 0.15

def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for Lab 4. Install the project requirements "
            "with: python -m pip install -r requirements.txt"
        )


def wrap_angle(angle: float) -> float:
    """Normalise an angle to [-pi, pi)."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def heading_towards(origin: Sequence[float], target: Sequence[float]) -> float:
    """Absolute heading from ``origin`` to ``target``.

    Heading 0 is world +y and +pi/2 is world +x, so x and y swap relative to
    the usual ``atan2(dy, dx)``.
    """

    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    return wrap_angle(math.atan2(dx, dy))


# ------------------------------------------------------------------ geometry

def world_to_cell(point: Sequence[float]) -> Tuple[int, int]:
    """World position to (row, col), matching ``SpheroEnv._is_collision``."""

    height, width = len(OCCUPANCY_GRID), len(OCCUPANCY_GRID[0])
    col = int(math.floor(float(point[0]) / GRID_RESOLUTION + width / 2.0))
    row = int(math.floor(-float(point[1]) / GRID_RESOLUTION + height / 2.0))
    return row, col


def cell_to_world(cell: Sequence[int]) -> np.ndarray:
    """Centre of (row, col) in world coordinates."""

    height, width = len(OCCUPANCY_GRID), len(OCCUPANCY_GRID[0])
    x = (float(cell[1]) + 0.5 - width / 2.0) * GRID_RESOLUTION
    y = -(float(cell[0]) + 0.5 - height / 2.0) * GRID_RESOLUTION
    return np.array([x, y], dtype=float)


def cell_is_free(cell: Sequence[int]) -> bool:
    row, col = int(cell[0]), int(cell[1])
    if not (0 <= row < len(OCCUPANCY_GRID) and 0 <= col < len(OCCUPANCY_GRID[0])):
        return False
    return OCCUPANCY_GRID[row][col] == 0


def nearest_free_cell(cell: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Snap a noisy pose to the closest free planning node."""

    if cell_is_free(cell):
        return int(cell[0]), int(cell[1])
    best = None
    best_distance = math.inf
    for row, line in enumerate(OCCUPANCY_GRID):
        for col, value in enumerate(line):
            if value:
                continue
            distance = (row - cell[0]) ** 2 + (col - cell[1]) ** 2
            if distance < best_distance:
                best_distance = distance
                best = (row, col)
    return best


def astar(start_cell: Tuple[int, int], goal_cell: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Optimal four-neighbour path, Manhattan heuristic (admissible, consistent)."""

    if not cell_is_free(start_cell) or not cell_is_free(goal_cell):
        return []

    def heuristic(cell):
        return abs(cell[0] - goal_cell[0]) + abs(cell[1] - goal_cell[1])

    frontier = [(heuristic(start_cell), 0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0}
    while frontier:
        _f, cost, current = heapq.heappop(frontier)
        # Lazy deletion: a cheaper route to this cell was already expanded.
        if cost != g_score.get(current):
            continue
        if current == goal_cell:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for d_row, d_col in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            neighbour = (current[0] + d_row, current[1] + d_col)
            if not cell_is_free(neighbour):
                continue
            tentative = cost + 1
            if tentative >= g_score.get(neighbour, math.inf):
                continue
            came_from[neighbour] = current
            g_score[neighbour] = tentative
            heapq.heappush(frontier, (tentative + heuristic(neighbour), tentative, neighbour))
    return []


def turning_points(path: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Start, goal and every direction change of a grid path."""

    points = [np.asarray(p, dtype=float)[:2] for p in path]
    if len(points) < 3:
        return [p.copy() for p in points]
    kept = [points[0].copy()]
    for before, here, after in zip(points, points[1:], points[2:]):
        incoming = here - before
        outgoing = after - here
        if abs(incoming[0] * outgoing[1] - incoming[1] * outgoing[0]) > 1e-12:
            kept.append(here.copy())
    kept.append(points[-1].copy())
    return kept


def point_to_path_distance(point: Sequence[float], path: Sequence[np.ndarray]) -> float:
    """Shortest distance from a point to a polyline."""

    p = np.asarray(point, dtype=float)[:2]
    best = math.inf
    for a, b in zip(path[:-1], path[1:]):
        edge = b - a
        length_sq = float(np.dot(edge, edge))
        if length_sq < 1e-12:
            best = min(best, float(np.linalg.norm(p - a)))
            continue
        t = float(np.clip(np.dot(p - a, edge) / length_sq, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + t * edge))))
    return best


def nearest_path_point(point: Sequence[float], path: Sequence[np.ndarray]) -> np.ndarray:
    """Closest point on a polyline."""

    p = np.asarray(point, dtype=float)[:2]
    best = np.asarray(path[0], dtype=float)
    best_distance = float(np.linalg.norm(p - best))
    for a, b in zip(path[:-1], path[1:]):
        edge = b - a
        length_sq = float(np.dot(edge, edge))
        if length_sq < 1e-12:
            continue
        t = float(np.clip(np.dot(p - a, edge) / length_sq, 0.0, 1.0))
        candidate = a + t * edge
        distance = float(np.linalg.norm(p - candidate))
        if distance < best_distance:
            best_distance = distance
            best = candidate
    return best


def step_dynamics(state: Sequence[float], action: Sequence[float], dt: float) -> np.ndarray:
    """Measured Lab 1 plant model, used only to roll an action forward."""

    x, y, heading, speed = (float(v) for v in np.asarray(state, dtype=float)[:4])
    speed_cmd, heading_cmd = float(action[0]), wrap_angle(float(action[1]))
    # The drive unit turns far faster than the ball accelerates, so the heading
    # setpoint is reached within a control period and the speed lags it.
    heading_new = heading_cmd
    speed_new = speed + (speed_cmd - speed) * min(1.0, dt / 0.25)
    x_new = x + speed_new * math.sin(heading_new) * dt
    y_new = y + speed_new * math.cos(heading_new) * dt
    return np.array([x_new, y_new, heading_new, speed_new], dtype=float)


# ----------------------------------------------------------------------- EKF

class StateEKF:
    """Extended Kalman filter on [x, y, heading, speed], inside the policy.

    Both the simulator and the robot add measurement noise to every
    observation (the template's ``obs_noise_std_pos`` is 0.05 m), so the policy
    must not act on the raw observation: a 5 cm jitter is a fifth of a plate.
    The motion model is the Sphero heading convention with the commanded
    heading reached quickly and the speed treated as a slowly varying state,
    so the filter does not assume how a speed command maps to ground speed
    (that mapping differs between the simulator and the robot; see
    ``SpeedAdapter``).
    """

    #: Measurement noise (std): position, heading, speed. The template's
    #: simulator and Robot both default to 0.05 m and 0.05 rad / m/s. The
    #: speed channel is trusted far less than its noise suggests: the template
    #: Robot reports ``get_speed() / 255 * vel_limit``, which is the command
    #: echoed back rather than a measurement, so ground speed is taken from
    #: the positions and the reported speed only nudges it.
    R_DIAG = (0.05 ** 2, 0.05 ** 2, 0.08 ** 2, 0.05 ** 2)
    #: Used instead for the speed channel once it is found to be an echo of
    #: the command (see SpeedAdapter.speed_is_echo).
    R_SPEED_ECHO = 0.08 ** 2
    #: Process noise per second: position, heading, speed.
    Q_POS = 0.02 ** 2
    Q_HEADING = 0.6 ** 2
    Q_SPEED = 0.25 ** 2
    #: Time constant of the drive unit turning to a commanded heading.
    HEADING_TAU = 0.15

    def __init__(self):
        self.x: Optional[np.ndarray] = None
        self.P = np.eye(4)
        self.speed_is_echo = False

    def reset(self, z: np.ndarray) -> None:
        self.x = np.array([z[0], z[1], z[2], max(float(z[3]), 0.0)], dtype=float)
        self.P = np.diag((0.05 ** 2, 0.05 ** 2, 0.3 ** 2, 0.05 ** 2))

    def predict(self, heading_cmd: Optional[float], dt: float) -> None:
        x, y, h, v = self.x
        if heading_cmd is not None:
            h = wrap_angle(h + wrap_angle(heading_cmd - h) * min(1.0, dt / self.HEADING_TAU))
        x_new = x + v * math.sin(h) * dt
        y_new = y + v * math.cos(h) * dt
        F = np.eye(4)
        F[0, 2] = v * math.cos(h) * dt
        F[0, 3] = math.sin(h) * dt
        F[1, 2] = -v * math.sin(h) * dt
        F[1, 3] = math.cos(h) * dt
        Q = np.diag((self.Q_POS * dt, self.Q_POS * dt, self.Q_HEADING * dt, self.Q_SPEED * dt))
        self.x = np.array([x_new, y_new, h, v], dtype=float)
        self.P = F @ self.P @ F.T + Q

    def update(self, z: np.ndarray, collision: bool = False) -> None:
        R = np.diag(self.R_DIAG)
        # The reported heading is not used: the robot reports it in the drive
        # unit's own frame, which the aim rotates away from the frame the
        # commands are in, so on the robot it disagreed with the direction of
        # travel and the prediction pushed the estimate the wrong way. The
        # ball drives along the commanded heading, which predict() follows.
        R[2, 2] = 1.0e6
        if self.speed_is_echo:
            R[3, 3] = self.R_SPEED_ECHO
        if collision:
            # A collision zeroes the ball's speed in both environments.
            R[3, 3] = 0.005 ** 2
            z = z.copy()
            z[3] = 0.0
        innovation = z - self.x
        innovation[2] = wrap_angle(innovation[2])
        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.x[2] = wrap_angle(self.x[2])
        self.P = (np.eye(4) - K) @ self.P
        self.P = 0.5 * (self.P + self.P.T)


# ------------------------------------------------------------- speed adapter

class SpeedAdapter:
    """Turn a ground-speed target (m/s) into the step function's speed command.

    The same command means different ground speeds on different plants. In the
    template simulator the action is metres per second. The template Robot
    sends ``cmd / vel_limit * 255`` as motor power, so with vel_limit 0.15 a
    command of 0.026 is already raw 45, about 0.16 m/s on the floor, a factor
    of about 6. A fixed command is therefore either far too slow in one or
    fast enough to hit the walls in the other.

    The adapter learns the factor online. While a steady command is held it
    measures how far the filtered position actually moved, keeps a running
    estimate of ``gain = ground speed / command``, and divides every target by
    it. Ground speed comes from displacement, never from the reported speed:
    the template Robot's speed channel is the command echoed back, and
    trusting it drove the estimated gain down and the command up to full
    power. It
    starts from the robot's factor, so the first commands are gentle on
    hardware and are raised within about a second in the simulator. A command
    the ball does not move under (motor stiction) raises a learned floor.
    """

    GAIN_INITIAL = 6.0
    #: The template simulator's gain is 1 (its action is m/s), the lowest of
    #: the plants this has to drive.
    GAIN_MIN, GAIN_MAX = 0.8, 15.0
    #: A gain window needs at least this much travel, well clear of the 1-2 cm
    #: the filtered position wanders on sensor noise alone.
    MIN_TRAVEL = 0.05
    SMOOTHING = 0.35
    MIN_MOVING = 0.03          # m/s; below this the speed is noise
    SETTLE_STEPS = 3           # steps a command must be held before measuring
    WINDOW = 8                 # steps of displacement per gain measurement
    STALL_SECONDS = 0.5
    #: The stiction floor never rises past this: a ball pinned against a wall
    #: does not move whatever it is told, and raising the floor without limit
    #: would end at full motor power (raw 255 on the template Robot).
    FLOOR_MAX = 0.035
    #: Stall check: commanded to move for this long without the filtered
    #: position getting this far.
    STALL_DISTANCE = 0.01
    #: A gain window only counts when every command in it is within this
    #: fraction of the window's mean: right after a cut the ball is still
    #: coasting, and speed / command then reads far too high.
    STEADY_SPREAD = 0.35
    #: Window for checking the reported speed against displacement.
    CHANNEL_SECONDS = 1.5

    def __init__(self, vel_limit: float):
        self.vel_limit = float(vel_limit)
        self.gain = self.GAIN_INITIAL
        self.floor = 0.0
        self._held = 0
        self._last_cmd = 0.0
        self._stalled_for = 0.0
        self._history: List[Tuple[np.ndarray, float, float, float]] = []
        #: Votes for "the reported speed is the command echoed back" (+1) or
        #: "it is a real measurement" (-1), one per measurement window.
        self.echo_votes = 0
        self.updates = 0
        self.gain_measured = self.GAIN_INITIAL
        self._channel: List[Tuple[np.ndarray, float, float]] = []
        self._since_vote = 0.0
        self._stall: List[Tuple[np.ndarray, float, float]] = []

    @property
    def speed_is_echo(self) -> bool:
        return self.echo_votes >= 3

    def observe(self, position: np.ndarray, heading_ok: bool, collision: bool, dt: float,
                reported_speed: float = 0.0, heading_cmd: Optional[float] = None) -> None:
        """Update the gain from the motion produced by the previous command."""

        cmd = self._last_cmd
        self._check_stall(position, cmd, dt)
        self._check_speed_channel(position, dt, reported_speed, heading_ok and not collision)
        # Any window of driving in a straight line counts; the command may vary
        # inside it, so the displacement is divided by the mean command.
        steady = cmd > 1e-6 and not collision and heading_ok
        if not steady:
            self._history.clear()
            self._stalled_for = 0.0
            return
        self._history.append((np.asarray(position, dtype=float)[:2].copy(), float(dt), cmd,
                              float(reported_speed),
                              0.0 if heading_cmd is None else float(heading_cmd)))
        if len(self._history) > self.WINDOW + 1:
            self._history.pop(0)
        if len(self._history) <= self.WINDOW:
            return
        elapsed = sum(h[1] for h in self._history[1:])
        displacement = self._history[-1][0] - self._history[0][0]
        moved = float(np.linalg.norm(displacement))
        # Only motion along the commanded direction is evidence: a ball that
        # is held or pinned still shows a filtered position wandering on
        # sensor noise, and reading that as "slow" walked the gain down until
        # the command reached full power.
        mean_heading = math.atan2(sum(math.sin(h[4]) for h in self._history[1:]),
                                  sum(math.cos(h[4]) for h in self._history[1:]))
        along = float(displacement[0] * math.sin(mean_heading)
                      + displacement[1] * math.cos(mean_heading))
        if moved < self.MIN_TRAVEL or along < 0.85 * moved:
            return
        speed_est = moved / max(elapsed, 1e-6)
        mean_cmd = float(np.mean([h[2] for h in self._history[1:]]))
        if any(abs(h[2] - mean_cmd) > self.STEADY_SPREAD * mean_cmd for h in self._history[1:]):
            return
        mean_reported = float(np.mean([h[3] for h in self._history[1:]]))
        if mean_cmd > 1e-6:
            if speed_est >= self.MIN_MOVING:
                measured = float(np.clip(speed_est / mean_cmd, self.GAIN_MIN, self.GAIN_MAX))
                # A falling gain raises every command, so it may fall at most
                # 25 % per window; rising (gentler commands) is not limited.
                self.gain = max(self.gain + self.SMOOTHING * (measured - self.gain),
                                0.75 * self.gain)
                self.updates += 1
                self.gain_measured = self.gain

    def _check_speed_channel(self, position, dt: float, reported: float, straight: bool) -> None:
        """Is the reported speed a measurement at all?

        Compared with the ball's actual displacement over the last ~1.5 s,
        whatever the command is doing. On the template Robot the channel is
        ``get_speed() / 255 * vel_limit``, the command echoed back: on the
        robot it read about 0.017 m/s while the ball covered 0.15 m/s, the
        filter believed it, and the estimate fell 10-25 cm behind the ball. A
        test that needed a steady command never got to vote, because the MPC
        changes its command every step.
        """

        if not straight:
            self._channel.clear()
            return
        self._channel.append((np.asarray(position, dtype=float)[:2].copy(), float(dt), float(reported)))
        elapsed = sum(h[1] for h in self._channel[1:])
        while len(self._channel) > 2 and elapsed - self._channel[1][1] >= self.CHANNEL_SECONDS:
            self._channel.pop(0)
            elapsed = sum(h[1] for h in self._channel[1:])
        self._since_vote += dt
        if elapsed < 0.8 * self.CHANNEL_SECONDS or self._since_vote < 0.5:
            return
        moved = float(np.linalg.norm(self._channel[-1][0] - self._channel[0][0])) / elapsed
        if moved < 0.06:
            return
        ratio = float(np.mean([abs(h[2]) for h in self._channel[1:]])) / moved
        inconsistent = ratio < 0.5 or ratio > 2.0
        self.echo_votes = int(np.clip(self.echo_votes + (1 if inconsistent else -1), -5, 5))
        self._since_vote = 0.0

    def _check_stall(self, position: np.ndarray, cmd: float, dt: float) -> None:
        """Raise the command floor, and lower the gain, when the ball will not
        move under the commands it is given.

        Looks at the last STALL_SECONDS as a whole, so a command that toggles
        through zero (the MPC does) still counts. No motion while mostly told
        to move means the commands are under the motor's stiction, i.e. the
        gain is over-estimated: on the template Robot an over-estimated gain
        of 10 sent raw 15 against a stiction of about 21 and the ball sat
        still for 400 steps.
        """

        self._stall.append((np.asarray(position, dtype=float)[:2].copy(), float(dt), cmd))
        elapsed = sum(h[1] for h in self._stall[1:])
        while len(self._stall) > 2 and elapsed - self._stall[1][1] >= self.STALL_SECONDS:
            self._stall.pop(0)
            elapsed = sum(h[1] for h in self._stall[1:])
        if elapsed < self.STALL_SECONDS:
            return
        driven = [h[2] for h in self._stall if h[2] > 1e-6]
        if len(driven) < 0.6 * len(self._stall):
            return
        spread = max(float(np.linalg.norm(h[0] - self._stall[0][0])) for h in self._stall)
        if spread < self.STALL_DISTANCE:
            stuck_cmd = float(np.mean(driven))
            self.floor = min(max(self.floor, stuck_cmd) * 1.3, self.FLOOR_MAX)
            # Bounded: a ball that is pinned or held does not move whatever it
            # is told, and unbounded cuts walked the command to full power.
            self.gain = max(0.7 * self.gain, 0.5 * self.gain_measured, self.GAIN_MIN)
            self._stall.clear()

    def predicted_speed(self) -> float:
        """Ground speed the last command should produce, from the learned gain."""

        cmd = self._last_cmd
        if cmd <= 1e-6 or cmd < 0.999 * self.floor:
            return 0.0
        return cmd * self.gain

    def command(self, target: float) -> float:
        if abs(target) <= 1e-9:
            cmd = 0.0
        else:
            cmd = abs(target) / self.gain
            cmd = max(cmd, self.floor)
            cmd = min(cmd, self.vel_limit) * (1.0 if target > 0 else -1.0)
        if abs(cmd - self._last_cmd) <= 0.2 * max(abs(self._last_cmd), 1e-6):
            self._held += 1
        else:
            self._held = 0
        self._last_cmd = abs(cmd) if cmd > 0 else 0.0
        return cmd



# ------------------------------------------------------------- physical walls

def plate_walls(plate: float = PLATE_SIZE) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Wall segments of the real maze: the outer boundary plus every line
    between two plates whose connector cell in the grid is occupied. The
    grid's free corridor is only half a plate wide, so the ball is checked
    against the real plate edges."""

    occ = OCCUPANCY_GRID
    n = (len(occ) - 1) // 2
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


WALLS = plate_walls()
_WALL_A = np.array([a for a, _b in WALLS])
_WALL_AB = np.array([b - a for a, b in WALLS])
_WALL_LEN2 = np.maximum(np.einsum("ij,ij->i", _WALL_AB, _WALL_AB), 1e-12)
_GRID = np.asarray(OCCUPANCY_GRID)
#: Ball radius plus 5 mm, measured against the real plate edges.
WALL_CLEARANCE = BALL_DIAMETER / 2.0 + 0.005


def points_free(points: np.ndarray) -> np.ndarray:
    """For (N, 2) points: clear of every physical wall, and the ball centre in
    a free cell of the 0.125 m grid (the simulator collides on the centre
    against that skeleton)."""

    p = np.asarray(points, dtype=float).reshape(-1, 2)
    rel = p[:, None, :] - _WALL_A[None, :, :]
    t = np.clip(np.einsum("nwk,wk->nw", rel, _WALL_AB) / _WALL_LEN2[None, :], 0.0, 1.0)
    closest = _WALL_A[None, :, :] + t[:, :, None] * _WALL_AB[None, :, :]
    clear = np.min(np.linalg.norm(p[:, None, :] - closest, axis=2), axis=1) >= WALL_CLEARANCE
    height, width = _GRID.shape
    cols = np.floor(p[:, 0] / GRID_RESOLUTION + width / 2.0).astype(int)
    rows = np.floor(-p[:, 1] / GRID_RESOLUTION + height / 2.0).astype(int)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    cell_free = np.zeros(len(p), dtype=bool)
    cell_free[inside] = _GRID[rows[inside], cols[inside]] == 0
    return clear & cell_free


# ------------------------------------------------------ model command units

#: Motor map of the robot the model was trained on (Lab 1 calibration):
#: raw = 12 + cmd / 0.6 * 243, capped at 45; the ball moves at about
#: 0.0052 m/s per raw step above 15 (measured: raw 45 -> 0.16 m/s,
#: raw 29 -> 0.074 m/s) and not at all under raw 21.
def model_cmd_to_ground_speed(cmd: float) -> float:
    if cmd <= 0.0:
        return 0.0
    raw = min(12.0 + cmd / 0.6 * 243.0, 45.0)
    return 0.0052 * (raw - 15.0) if raw >= 21.0 else 0.0


def ground_speed_to_model_cmd(speed: float) -> float:
    if speed <= 0.0:
        return 0.0
    raw = min(speed / 0.0052 + 15.0, 45.0)
    return max(raw - 12.0, 0.0) / 243.0 * 0.6


# ------------------------------------------------- learned dynamics + MPC

if torch is not None:

    class DynamicsModel(torch.nn.Module):
        """Learned one-step dynamics (see the module docstring)."""

        def __init__(self, hidden: int = 64):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(6, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                torch.nn.Linear(hidden, 4),
            )
            self.register_buffer("x_mean", torch.zeros(6))
            self.register_buffer("x_std", torch.ones(6))
            self.register_buffer("y_mean", torch.zeros(4))
            self.register_buffer("y_std", torch.ones(4))

        def forward(self, x):
            return self.net((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean

    def batch_step(model, pos, vel, speed_cmd, heading_cmd, prev_speed, prev_heading, dt):
        """Advance N candidates one step through the learned model, in the
        commanded-heading frame the model was trained in."""

        s, c = torch.sin(heading_cmd), torch.cos(heading_cmd)
        vf = vel[:, 0] * s + vel[:, 1] * c
        vl = vel[:, 0] * c - vel[:, 1] * s
        turn = torch.remainder(prev_heading - heading_cmd + math.pi, 2 * math.pi) - math.pi
        pf, pl = prev_speed * torch.cos(turn), prev_speed * torch.sin(turn)
        x = torch.stack((vf, vl, speed_cmd, pf, pl, torch.full_like(vf, float(dt))), dim=1)
        out = model(x)
        df, dl, vf2, vl2 = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        dpos = torch.stack((df * s + dl * c, df * c - dl * s), dim=1)
        nvel = torch.stack((vf2 * s + vl2 * c, vf2 * c - vl2 * s), dim=1)
        return pos + dpos, nvel


class LearnedMPC:
    """Choose ``[speed_cmd, heading_cmd]`` (model units) by rolling candidates
    through the learned model. Same costs and settings as the robot runs."""

    HEADING_OFFSETS = (-0.45, -0.22, 0.0, 0.22, 0.45)
    SPEED_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    #: From rest only "stop" or a real push: the model has little data on
    #: starting from standstill and believes a bigger command moves the ball
    #: less there.
    REST_FRACTIONS = (0.0, 0.8, 1.0)
    REST_SPEED = 0.03
    LOOKAHEAD = 0.10
    HORIZON_S = 1.2
    STOP_S = 2.2
    VEL_SMOOTH = 0.5
    ARRIVE_RADIUS = 0.03
    SETTLE_RADIUS = 0.06
    SETTLE_SPEED = 0.03
    GOAL_RUN_ON = 0.0
    W_GOAL_OVERSHOOT_LIN = 3000.0
    #: Aim to come to rest 4 cm short of the goal: on the robot the ball
    #: consistently finished 3.7-5.8 cm past without it.
    GOAL_AIM_SHORT = 0.04
    W_PROGRESS = 30.0
    W_CROSS = 600.0
    W_OVERSHOOT = 8000.0
    W_WALL = 1.0e4
    W_TURN = 0.02
    #: With more than this much straight left before the next stop point the
    #: candidates are limited to the upper speed levels. The learned model is
    #: weakest at low speed, where it believes a bigger command moves the
    #: ball less, and a ball that started slowly could settle into choosing
    #: zero; on the robot the MPC cruised at the top levels here anyway.
    OPEN_STRAIGHT = 0.25
    #: Off the current segment by more than this -> check the others.
    RESYNC_OFF = 0.08
    OPEN_FRACTIONS = (0.6, 0.8, 1.0)

    def __init__(self, model, waypoints, start, cruise, dt, speed_factor=None):
        self.model = model
        self.points = [np.asarray(start, dtype=float)[:2]] + [
            np.asarray(w, dtype=float)[:2] for w in waypoints]
        self.cruise = float(cruise)
        self.dt = float(dt)
        self.speed_factor = speed_factor
        self.at_rest = True
        self.index = 1
        self.prev_pos: Optional[np.ndarray] = None
        self.prev_dt: Optional[float] = None
        self.last_heading: Optional[float] = None
        self.last_speed = 0.0
        self.vel_est = np.zeros(2)
        self.last_best_hits_wall = False
        self.done = False

    def _segment(self):
        return self.points[self.index - 1], self.points[self.index]

    @staticmethod
    def _along(p, a, b):
        d = b - a
        L = float(np.linalg.norm(d)) or 1e-9
        u = d / L
        n = np.array([u[1], -u[0]])
        rel = p - a
        return rel @ u, rel @ n, L

    def _aim(self, pos):
        a, b = self._segment()
        s, _lat, L = self._along(pos[None, :], a, b)
        u = (b - a) / (L or 1e-9)
        return a + u * min(max(float(s[0]), 0.0) + self.LOOKAHEAD, L)

    def _velocity(self, pos):
        if self.prev_pos is None or not self.prev_dt:
            return self.vel_est
        raw = (pos - self.prev_pos) / max(self.prev_dt, 1e-3)
        self.vel_est = self.VEL_SMOOTH * raw + (1.0 - self.VEL_SMOOTH) * self.vel_est
        return self.vel_est

    def _advance(self, pos, vel):
        while self.index < len(self.points):
            a, b = self._segment()
            s, _lat, L = self._along(pos[None, :], a, b)
            dist = float(np.linalg.norm(pos - b))
            speed = float(np.linalg.norm(vel))
            reached = (dist <= self.ARRIVE_RADIUS
                       or (dist <= self.SETTLE_RADIUS and speed <= self.SETTLE_SPEED)
                       or float(s[0]) >= L)
            if not reached:
                return
            if self.index == len(self.points) - 1:
                self.done = True
                return
            self.index += 1

    def _resync(self, pos) -> None:
        """Re-select the route segment when the ball is clearly on another one.

        A jam recovery or a bounce can carry the ball back past a corner it
        had already reached; aiming on along the later segment then points
        through a wall and every candidate is priced as a collision.
        """

        def seg_dist(i):
            a, b = self.points[i - 1], self.points[i]
            d = b - a
            t = float(np.clip(np.dot(pos - a, d) / max(float(np.dot(d, d)), 1e-12), 0.0, 1.0))
            return float(np.linalg.norm(pos - (a + t * d)))

        here = seg_dist(self.index)
        if here <= self.RESYNC_OFF:
            return
        best = min(range(1, len(self.points)), key=seg_dist)
        if best != self.index and seg_dist(best) < here - 0.03:
            self.index = best
            self.done = False

    def act(self, position, dt, velocity=None):
        pos = np.asarray(position, dtype=float)[:2]
        self._resync(pos)
        if velocity is not None:
            # The filter's velocity. Differencing filtered positions 0.1 s
            # apart turned ~1 cm of filter wander into ~0.1 m/s of phantom
            # speed, and the model then braked for a coast that was not there.
            self.vel_est = np.asarray(velocity, dtype=float)[:2].copy()
            vel = self.vel_est
        else:
            vel = self._velocity(pos)
        self.at_rest = float(np.linalg.norm(vel)) < self.REST_SPEED
        self._advance(pos, vel)
        self.prev_pos, self.prev_dt = pos.copy(), dt
        a, b = self._segment()
        if self.done:
            heading = self.last_heading if self.last_heading is not None else heading_towards(pos, b)
            return np.array([0.0, heading])

        base = heading_towards(pos, self._aim(pos))
        fractions = self.REST_FRACTIONS if self.at_rest else self.SPEED_FRACTIONS
        s_now, _lat_now, L_now = self._along(pos[None, :], a, b)
        if L_now - float(s_now[0]) > self.OPEN_STRAIGHT:
            fractions = self.OPEN_FRACTIONS
        cruise = self.cruise * (float(self.speed_factor(pos, b)) if self.speed_factor else 1.0)
        cands = [(f * cruise, wrap_angle(base + o)) for f in fractions for o in self.HEADING_OFFSETS]
        N = len(cands)
        sp = torch.tensor([c[0] for c in cands], dtype=torch.float32)
        hd = torch.tensor([c[1] for c in cands], dtype=torch.float32)
        p0 = torch.tensor(np.repeat(pos[None, :], N, 0), dtype=torch.float32)
        v0 = torch.tensor(np.repeat(vel[None, :], N, 0), dtype=torch.float32)
        last_h = self.last_heading if self.last_heading is not None else base
        prev_sp0 = torch.full_like(sp, float(self.last_speed))
        prev_hd0 = torch.full_like(hd, float(last_h))
        zero = torch.zeros_like(sp)
        horizon = max(3, int(round(self.HORIZON_S / dt)))
        stop_steps = max(6, int(round(self.STOP_S / dt)))
        with torch.no_grad():
            p, v = p0, v0
            held = []
            prev_s, prev_h = prev_sp0, prev_hd0
            for _ in range(horizon):
                p, v = batch_step(self.model, p, v, sp, hd, prev_s, prev_h, dt)
                prev_s, prev_h = sp, hd
                held.append(p.numpy().astype(float))
            p1, v1 = batch_step(self.model, p0, v0, sp, hd, prev_sp0, prev_hd0, dt)
            ps, vs = p1, v1
            prev_s = sp
            rest_path = [p1.numpy().astype(float)]
            for _ in range(stop_steps):
                ps, vs = batch_step(self.model, ps, vs, zero, hd, prev_s, hd, dt)
                prev_s = zero
                rest_path.append(ps.numpy().astype(float))

        s0 = float(self._along(pos[None, :], a, b)[0][0])
        L = float(np.linalg.norm(b - a)) or 1e-9
        cost = np.zeros(N)
        s_h, _lh, _ = self._along(held[-1], a, b)
        cost -= self.W_PROGRESS * (np.minimum(s_h, L) - s0)
        for q in held:
            _s, lat, _ = self._along(q, a, b)
            cost += self.W_CROSS * lat ** 2
        final_leg = self.index == len(self.points) - 1
        target_rest = L + self.GOAL_RUN_ON - (self.GOAL_AIM_SHORT if final_leg else 0.0)
        s_rest, _lr, _ = self._along(rest_path[-1], a, b)
        over = np.maximum(s_rest - target_rest, 0.0)
        cost += self.W_OVERSHOOT * (over ** 2) + self.W_GOAL_OVERSHOOT_LIN * over
        # walls, on every predicted point, all candidates at once
        pts = np.stack(held + rest_path[::3], axis=1)                 # (N, T, 2)
        free = points_free(pts.reshape(-1, 2)).reshape(N, -1)
        wall = ~np.all(free, axis=1)
        cost += self.W_WALL * wall
        if self.last_heading is not None:
            cost += self.W_TURN * np.array([abs(wrap_angle(c[1] - self.last_heading)) for c in cands])

        best = int(np.argmin(cost))
        if self.at_rest and cands[best][0] <= 0.0:
            # At rest and not at the goal: never choose to stay stopped.
            moving = [i for i, c in enumerate(cands) if c[0] > 0.0]
            if moving:
                best = min(moving, key=lambda i: cost[i])
        speed, heading = cands[best]
        self.last_best_hits_wall = bool(wall[best])
        self.last_heading = heading
        self.last_speed = float(speed)
        return np.array([float(speed), float(heading)])

    def note_applied(self, model_speed: float, heading: float) -> None:
        """Record what was really sent when something overrode the MPC: the
        model takes the previous command as an input."""

        self.last_speed = float(model_speed)
        self.last_heading = wrap_angle(float(heading))


# -------------------------------------------------------------------- policy

if torch is not None:

    class Policy(torch.nn.Module):
        """``policy = Policy(); action = policy(obs)``: see the module docstring."""

        #: Cruise in the model's units: raw 45, the robot's motor cap, which is
        #: where the robot runs' cruise (speed scale 20) ended up.
        CRUISE = 0.0815
        CORNER2_SLOW = 0.6
        CORNER3_SLOW = 0.8
        GOAL_SLOW = 0.5
        GOAL_SLOW_FAR = 0.30
        GOAL_SLOW_NEAR = 0.12
        SLOW_ZONE_START = 0.8
        GOAL_LATCH = 0.05
        GOAL_RELEASE = 0.085
        #: Jam handling (ground speeds in m/s).
        STALL_DISTANCE = 0.004
        WEDGE_TRIGGER = 5
        WEDGE_RECOVERY_STEPS = 8
        RECOVERY_SPEED = 0.08
        WEIGHT_FILES = ("weight.pth", "37482025_weight.pth", "37482025_m4_2.pth")
        #: Speed calibration on the first straight: hold this ground-speed
        #: target until the plant's gain has been measured this many times
        #: (or for at most CAL_MAX_STEPS). The MPC switches between speed
        #: levels every step, which no steady-command window survives, so
        #: without this the gain stayed at its cautious prior and the ball
        #: crawled at 3 cm/s in the simulator.
        #: Command ceiling on the template Robot (see _send): raw 77.
        ECHO_CMD_MAX = 0.045
        CAL_SPEED = 0.10
        CAL_UPDATES = 2
        CAL_MAX_STEPS = 40

        def __init__(self, d_in: int = 5, hidden: int = 64, d_out: int = 2, *,
                     goal: Sequence[float] = GOAL, dt: float = 0.1):
            super().__init__()
            self.dynamics = DynamicsModel(hidden=64)
            self.dynamics.eval()
            self.goal_xy = np.array([float(goal[0]), float(goal[1])], dtype=float)
            self.dt = float(dt)
            self.reset()
            self.weights_loaded = self._load_default_weights()

        # ----------------------------------------------------------- set-up

        def _load_default_weights(self) -> Optional[str]:
            folders = []
            try:
                folders.append(os.path.dirname(os.path.abspath(__file__)))
            except NameError:
                pass
            folders.append(os.getcwd())
            for folder in folders:
                for name in self.WEIGHT_FILES:
                    path = os.path.join(folder, name)
                    if not os.path.isfile(path):
                        continue
                    try:
                        try:
                            state = torch.load(path, map_location="cpu", weights_only=True)
                        except TypeError:
                            state = torch.load(path, map_location="cpu")
                        self.load_state_dict(state)
                        return path
                    except Exception:
                        continue
            return None

        def load_state_dict(self, state_dict, strict: bool = True, *args, **kwargs):
            # Also accept the bare dynamics model's own state_dict.
            keys = list(state_dict.keys())
            if keys and not any(str(k).startswith("dynamics.") for k in keys):
                state_dict = {"dynamics." + k: v for k, v in state_dict.items()}
            return super().load_state_dict(state_dict, strict=strict)

        def reset(self) -> None:
            self.route: List[np.ndarray] = []
            self.waypoints: List[np.ndarray] = []
            self.mpc: Optional[LearnedMPC] = None
            self.ekf = StateEKF()
            self.speed = SpeedAdapter(VEL_LIMIT)
            # The MPC's command varies step to step; accept wider windows.
            self.speed.STEADY_SPREAD = 0.6
            self.speed.MIN_TRAVEL = 0.03
            self.speed.WINDOW = 10
            self.frame_offset: Optional[np.ndarray] = None
            self._last_call: Optional[float] = None
            self._last_heading_cmd: Optional[float] = None
            self._at_goal = False
            self._previous_position: Optional[np.ndarray] = None
            self._wedge_steps = 0
            self._wall_steps = 0
            self._recent: List[Tuple[np.ndarray, float]] = []
            self._recovery_left = 0
            self.takeovers = 0
            self.reasons = {}
            self.steps = 0

        def plan(self, position: Sequence[float]) -> None:
            start_cell = nearest_free_cell(world_to_cell(position))
            goal_cell = nearest_free_cell(world_to_cell(self.goal_xy))
            cells = astar(start_cell, goal_cell) if start_cell and goal_cell else []
            self.route = [cell_to_world(c) for c in cells] or [
                np.asarray(position, dtype=float)[:2].copy(), self.goal_xy.copy()]
            self.waypoints = turning_points(self.route)
            while (len(self.waypoints) > 1
                   and float(np.linalg.norm(self.waypoints[0] - np.asarray(position)[:2]))
                   <= GRID_RESOLUTION * 0.51):
                self.waypoints.pop(0)
            self.mpc = LearnedMPC(self.dynamics, self.waypoints, start=position,
                                  cruise=self.CRUISE, dt=self.dt, speed_factor=self._zone_factor)

        @property
        def waypoint_index(self) -> int:
            return (self.mpc.index - 1) if self.mpc is not None else 0

        # ------------------------------------------------------- slow zones

        def _zone_factor(self, position, target) -> float:
            tx, ty = float(target[0]), float(target[1])
            x, y = float(position[0]), float(position[1])
            factor = 1.0
            if (abs(tx - self.goal_xy[0]) < 0.05 and abs(ty - self.goal_xy[1]) < 0.05
                    and abs(y - self.goal_xy[1]) < 0.125):
                d = math.hypot(x - self.goal_xy[0], y - self.goal_xy[1])
                ramp = (d - self.GOAL_SLOW_NEAR) / (self.GOAL_SLOW_FAR - self.GOAL_SLOW_NEAR)
                factor = self.GOAL_SLOW + (1.0 - self.GOAL_SLOW) * min(max(ramp, 0.0), 1.0)
            elif (abs(tx + 0.25) < 0.125 and abs(ty - 0.5) < 0.125
                    and abs(x + 0.25) < 0.125 and 0.125 <= y <= 0.625):
                factor = self.CORNER3_SLOW            # up column 1 into corner 3
            elif (abs(tx + 0.25) < 0.125 and abs(ty) < 0.125
                    and -0.625 <= x <= -0.125 and abs(y) < 0.125):
                factor = self.CORNER2_SLOW            # along row 2 into corner 2
            if factor < 1.0 and self.mpc is not None and self.mpc.at_rest:
                factor = max(factor, self.SLOW_ZONE_START)
            return factor

        # ------------------------------------------------------------ safety

        def _note(self, reason: str) -> None:
            self.takeovers += 1
            self.reasons[reason] = self.reasons.get(reason, 0) + 1

        def _recovery_action(self, position: np.ndarray, heading: float) -> np.ndarray:
            """Back onto the route centreline, or back along it if already on it."""

            target = nearest_path_point(position, self.route)
            if float(np.linalg.norm(position - target)) < 0.01:
                return np.array([-self.RECOVERY_SPEED, wrap_angle(heading)], dtype=float)
            return np.array([self.RECOVERY_SPEED, heading_towards(position, target)], dtype=float)

        # ----------------------------------------------------------- forward

        def forward(self, observation) -> "torch.Tensor":
            if not torch.is_tensor(observation):
                observation = torch.tensor(np.asarray(observation, dtype=np.float32),
                                           dtype=torch.float32)
            observation = observation.to(dtype=torch.float32)
            if observation.ndim > 1:
                # A batch carries no trajectory: answer each row on its own.
                return torch.stack([self.__class__()(row) for row in observation])

            state = observation.detach().cpu().numpy().astype(float).reshape(-1)
            if state.size < 4 or not np.all(np.isfinite(state[:4])):
                return torch.zeros(2, dtype=torch.float32)

            # Start-relative frame (both step functions reset to (0, 0)).
            if self.frame_offset is None:
                self.frame_offset = (np.asarray(START, dtype=float)
                                     if float(np.linalg.norm(state[:2])) < 0.25 else np.zeros(2))
            state = state.copy()
            state[:2] = state[:2] + self.frame_offset
            collision = bool(state.size >= 5 and float(state[4]) > 0.5)

            # Control period: measured when steps take real time (robot),
            # otherwise the template's nominal 0.1 s.
            now = time.perf_counter()
            dt = self.dt
            if self._last_call is not None:
                gap = now - self._last_call
                if 0.03 <= gap <= 1.0:
                    dt = gap
            self._last_call = now

            z = np.array(state[:4], dtype=float)
            z[2] = wrap_angle(z[2])
            reported_speed = float(z[3])
            if self.speed.speed_is_echo:
                z[3] = self.speed.predicted_speed()
            if self.ekf.x is None:
                self.ekf.reset(z)
            else:
                self.ekf.predict(self._last_heading_cmd, dt)
                self.ekf.update(z, collision)
            estimate = self.ekf.x.copy()
            heading_ok = (self._last_heading_cmd is None
                          or abs(wrap_angle(estimate[2] - self._last_heading_cmd)) < 0.4)
            self.speed.observe(estimate[:2], heading_ok, collision, dt, reported_speed,
                               self._last_heading_cmd)
            self.ekf.speed_is_echo = self.speed.speed_is_echo
            position = estimate[:2]

            if self.mpc is None:
                self.plan(position)
            self.steps += 1

            # Goal hold: the whole 25 cm plate counts.
            to_goal = float(np.linalg.norm(position - self.goal_xy))
            on_final_leg = self.mpc.index >= len(self.mpc.points) - 1
            if self._at_goal and to_goal > self.GOAL_RELEASE:
                self._at_goal = False
            elif not self._at_goal and on_final_leg and to_goal < self.GOAL_LATCH:
                self._at_goal = True
            if self._at_goal:
                self.mpc.prev_pos = position.copy()
                self.mpc.note_applied(0.0, float(estimate[2]))
                return self._send(0.0, float(estimate[2]))

            # Jams: a drive command and no motion, a collision flag, or a
            # stopped estimate, for several steps running.
            previous = self._previous_position if self._previous_position is not None else position
            self._previous_position = position.copy()
            if self._recovery_left > 0:
                self._recovery_left -= 1
                self._note("jam recovery")
                ground, heading = self._recovery_action(position, float(estimate[2]))
                self.mpc.prev_pos = None
                self.mpc.vel_est = np.zeros(2)
                self.mpc.note_applied(math.copysign(ground_speed_to_model_cmd(abs(ground)), ground),
                                      heading)
                return self._send(ground, heading)
            driving = self.mpc.last_speed > 0.0
            stuck = driving and (float(np.linalg.norm(position - previous)) < self.STALL_DISTANCE
                                 or collision or abs(float(estimate[3])) < 0.01)
            self._wedge_steps = self._wedge_steps + 1 if stuck else 0
            if self._wedge_steps >= self.WEDGE_TRIGGER:
                self._wedge_steps = 0
                self._recovery_left = self.WEDGE_RECOVERY_STEPS - 1
                self._note("jam recovery")
                ground, heading = self._recovery_action(position, float(estimate[2]))
                self.mpc.note_applied(math.copysign(ground_speed_to_model_cmd(abs(ground)), ground),
                                      heading)
                return self._send(ground, heading)

            velocity = float(estimate[3]) * np.array([math.sin(estimate[2]), math.cos(estimate[2])])
            # Displacement overrules the filter's speed: with an echoed speed
            # channel and an over-estimated gain the filter can believe the
            # ball is rolling while it sits still, and the MPC then "brakes"
            # for good. Not having moved 1.5 cm in a second is at rest.
            self._recent.append((position.copy(), dt))
            while len(self._recent) > 2 and sum(h[1] for h in list(self._recent)[1:]) > 1.0:
                self._recent.pop(0)
            if (sum(h[1] for h in self._recent[1:]) >= 0.8
                    and max(float(np.linalg.norm(h[0] - self._recent[0][0])) for h in self._recent) < 0.015):
                velocity = np.zeros(2)
            model_speed, heading = self.mpc.act(position, dt, velocity)
            parked = self.mpc.last_best_hits_wall and float(np.linalg.norm(position - previous)) < 0.01
            self._wall_steps = self._wall_steps + 1 if parked else 0
            if self._wall_steps >= self.WEDGE_TRIGGER:
                # Every option was predicted to end in a wall: get back onto
                # the route centreline rather than wait.
                self._wall_steps = 0
                self._recovery_left = self.WEDGE_RECOVERY_STEPS - 1
                self._note("all options into a wall")
                ground, heading = self._recovery_action(position, float(estimate[2]))
                self.mpc.note_applied(math.copysign(ground_speed_to_model_cmd(abs(ground)), ground),
                                      heading)
                return self._send(ground, heading)
            ground = model_cmd_to_ground_speed(float(model_speed))
            if (self.speed.updates < self.CAL_UPDATES and self.steps <= self.CAL_MAX_STEPS
                    and self.mpc.index == 1):
                # Calibration: a steady target up the first straight.
                ground = self.CAL_SPEED
                self.mpc.note_applied(ground_speed_to_model_cmd(ground), heading)
            # Never drive east past the goal centre: the coast already carries
            # the ball on, and the plate's east edge is the end of the map.
            if (abs(position[1] - self.goal_xy[1]) <= 0.125 and position[0] > self.goal_xy[0] + 0.02
                    and ground > 0.0 and math.sin(heading) > 0.3):
                ground = 0.0
                self.mpc.note_applied(0.0, heading)
                self._note("past goal: no eastward drive")
            return self._send(ground, heading)

        def _send(self, ground_speed: float, heading: float) -> "torch.Tensor":
            """Ground speed (m/s) -> this plant's command, through the adapter."""

            ground_speed = float(np.clip(ground_speed, -VEL_LIMIT, VEL_LIMIT))
            cmd = self.speed.command(ground_speed)
            if self.speed.speed_is_echo:
                # The speed channel echoes the command, i.e. this is the
                # template Robot (raw = cmd / 0.15 * 255). Never more than raw
                # 77 there, whatever the learned gain says.
                cmd = float(np.clip(cmd, -self.ECHO_CMD_MAX, self.ECHO_CMD_MAX))
            heading = wrap_angle(heading)
            self._last_heading_cmd = heading if cmd >= 0 else wrap_angle(heading + math.pi)
            return torch.tensor([cmd, heading], dtype=torch.float32)

        @property
        def contract(self) -> dict:
            return {
                "input": "[x, y, heading, speed] (+ optional collision flag)",
                "output": "[speed_cmd, absolute heading_cmd]",
                "learned": "DynamicsModel (offline, real runs) + SpeedAdapter (online)",
                "controller": "MPC over the learned model on an EKF estimate",
            }

else:

    class Policy:  # pragma: no cover - only used on machines without torch
        def __init__(self, *args, **kwargs):
            require_torch()
