"""Model-predictive control against a learned dynamics model.

This is the Ed #140 option "learn a dynamics model, and use your existing
navigation stack to optimise with respect to this". A* still chooses the route;
every command is chosen by predicting the ball through a model learned from
logged runs (dynamics_learning.py) and taking the best prediction.

Each control step
-----------------
1. Velocity is estimated from the last two positions, the same way the model's
   training data was built.
2. A small set of candidate commands is formed: headings around a pure-pursuit
   aim point on the current route segment, at a range of speeds.
3. Every candidate is rolled through the learned model, all in one batch:
     - held for a short horizon, to score progress and cross-track error;
     - applied for one step and then cut to zero, to predict where the ball
       would come to rest if the drive were cut next step.
4. The cost rewards progress along the segment and penalises cross-track error,
   any predicted rest point beyond the segment's end, and any predicted point
   inside a wall. The cheapest candidate is sent.

Step 3's "stop test" is what replaces the hand-fitted brake. The ball is
allowed to go as fast as the learned coast says it can and still stop at the
corner, which is a statement about this ball on this floor rather than a
constant fitted to two points.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def heading_towards(origin, target) -> float:
    """Heading 0 is world +y and +pi/2 is world +x."""

    return wrap(math.atan2(float(target[0]) - float(origin[0]),
                           float(target[1]) - float(origin[1])))


class DynamicsModel(torch.nn.Module):
    """Same layout as dynamics_learning.DynamicsModel, so its weights load."""

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
    """Advance N candidates one step through the learned model.

    pos, vel: (N, 2) world; the commands are (N,) tensors; dt is a scalar.
    The previous command is resolved into the current command's frame, which
    is how the model was trained: it carries both the actuation latency and
    the size of the turn.
    """

    s, c = torch.sin(heading_cmd), torch.cos(heading_cmd)
    # world -> commanded-heading frame (forward along the command)
    vf = vel[:, 0] * s + vel[:, 1] * c
    vl = vel[:, 0] * c - vel[:, 1] * s
    turn = torch.remainder(prev_heading - heading_cmd + math.pi, 2 * math.pi) - math.pi
    pf, pl = prev_speed * torch.cos(turn), prev_speed * torch.sin(turn)
    x = torch.stack((vf, vl, speed_cmd, pf, pl, torch.full_like(vf, float(dt))), dim=1)
    out = model(x)
    df, dl, vf2, vl2 = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    # frame -> world
    dpos = torch.stack((df * s + dl * c, df * c - dl * s), dim=1)
    nvel = torch.stack((vf2 * s + vl2 * c, vf2 * c - vl2 * s), dim=1)
    return pos + dpos, nvel


class LearnedMPC:
    """Choose ``[speed_cmd, heading_cmd]`` by rolling candidates through a model."""

    #: Heading offsets around the pure-pursuit aim, radians.
    HEADING_OFFSETS = (-0.45, -0.22, 0.0, 0.22, 0.45)
    #: Speed candidates as fractions of the cruise command.
    SPEED_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    REST_FRACTIONS = (0.0, 0.8, 1.0)
    REST_SPEED = 0.03
    LOOKAHEAD = 0.10
    #: Look-ahead in SECONDS, converted to steps with the live dt. In steps it
    #: meant 0.4 s in the simulator but 0.72 s on the robot, which is shorter
    #: than the robot's start-up latency, so a faster command did not look
    #: faster inside the horizon and the ball crawled away from every start.
    HORIZON_S = 1.2
    #: Long enough to cover the whole coast on either plant.
    STOP_S = 2.2
    #: Velocity comes from differencing positions. At 0.18 s a 3 mm position
    #: error is a 0.02 m/s velocity error, as large as the speeds at a start,
    #: so the estimate is smoothed before it reaches the model.
    VEL_SMOOTH = 0.5
    ARRIVE_RADIUS = 0.03   # a turning point is reached inside this ...
    SETTLE_RADIUS = 0.06   # ... or inside this once the ball has nearly stopped
    SETTLE_SPEED = 0.03

    #: Overshooting the goal is costly in a way a corner is not: the ball has
    #: to reverse onto the plate. On the first hardware run the model predicted
    #: the 11 cm run-on from the cut almost exactly (11.9 cm against 10.9), yet
    #: the ball still finished 5 cm past, because with progress capped at the
    #: goal, arriving at full speed scored as well as arriving slowly and a few
    #: centimetres of predicted overshoot cost only about 2 per cm. On the final
    #: leg the ball may not run on at all, and each centimetre past is priced
    #: above the progress it buys.
    GOAL_RUN_ON = 0.0
    W_GOAL_OVERSHOOT_LIN = 3000.0
    #: On the final leg, aim to come to rest this far SHORT of the goal.
    #: Measured on three robot runs: the controller ramps the command down to
    #: land the ball exactly on the goal, and at the moment of each cut the
    #: model predicted the run-on to about a centimetre, but the small error
    #: compounds over the ramp and the result was consistently 3.7, 5.8 and
    #: 5.5 cm past. A consistent bias earns a constant correction. Short of
    #: the goal costs nothing - 4 cm is well inside the 8.85 cm plate
    #: half-width and the creep finishes it - whereas past it is the end wall.
    #: Corners are not biased: stopping short of a corner turns early and cuts
    #: the inside wall, which is no better than running past it.
    GOAL_AIM_SHORT = 0.04
    W_PROGRESS = 30.0
    W_CROSS = 600.0
    W_OVERSHOOT = 8000.0
    W_WALL = 1.0e4
    W_TURN = 0.02

    def __init__(
        self,
        model: DynamicsModel,
        waypoints: Sequence[Sequence[float]],
        start: Sequence[float],
        cruise: float,
        dt: float,
        is_free: Optional[Callable[[Sequence[float]], bool]] = None,
        goal_aim_short: Optional[float] = None,
        speed_factor: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ):
        self.model = model.eval()
        self.points = [np.asarray(start, dtype=float)[:2]] + [
            np.asarray(w, dtype=float)[:2] for w in waypoints
        ]
        self.cruise = float(cruise)
        self.dt = float(dt)
        self.is_free = is_free
        # speed_factor(position, target) -> multiplier on cruise, for stretches
        # of the route that have to be driven slower than the rest.
        self.speed_factor = speed_factor
        self.at_rest = True
        # The bias correction is a property of the plant: the simulator barely
        # coasts, so there it would simply leave the ball short of the goal.
        if goal_aim_short is not None:
            self.GOAL_AIM_SHORT = float(goal_aim_short)
        self.index = 1                 # current target is self.points[index]
        self.prev_pos: Optional[np.ndarray] = None
        self.prev_dt: Optional[float] = None
        self.last_heading: Optional[float] = None
        self.last_speed = 0.0
        self.vel_est = np.zeros(2)
        self.last_best_hits_wall = False
        self._prev_sent = (0.0, 0.0)
        self.done = False

    # ---------------------------------------------------------------- helpers

    def _segment(self):
        a = self.points[self.index - 1]
        b = self.points[self.index]
        return a, b

    @staticmethod
    def _along(p, a, b):
        """(distance along a->b, signed lateral) for points p (N, 2)."""

        d = b - a
        L = float(np.linalg.norm(d)) or 1e-9
        u = d / L
        n = np.array([u[1], -u[0]])
        rel = p - a
        return rel @ u, rel @ n, L

    def _aim(self, pos):
        a, b = self._segment()
        s, _lat, L = self._along(pos[None, :], a, b)
        s = float(s[0])
        u = (b - a) / (L or 1e-9)
        target_s = min(max(s, 0.0) + self.LOOKAHEAD, L)
        return a + u * target_s

    def _velocity(self, pos, dt):
        if self.prev_pos is None or not self.prev_dt:
            return self.vel_est
        raw = (pos - self.prev_pos) / max(self.prev_dt, 1e-3)
        a = self.VEL_SMOOTH
        self.vel_est = a * raw + (1.0 - a) * self.vel_est
        return self.vel_est

    def _advance(self, pos, vel):
        """Move to the next turning point once the current one is reached."""

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

    # ------------------------------------------------------------------- act

    def predict_next(self, position, command, dt: Optional[float] = None) -> np.ndarray:
        """Where the learned model expects the ball one step after ``command``.

        Used by the safety layer as a collision sensor: a ball that ends up well
        short of where the model said it would go has run into something. The
        previous command is the one sent before this step, which is what the
        model was trained to condition on.
        """

        dt = float(dt) if dt else self.dt
        pos = np.asarray(position, dtype=float)[:2]
        prev_s, prev_h = self._prev_sent
        with torch.no_grad():
            p, _v = batch_step(
                self.model,
                torch.tensor(pos[None], dtype=torch.float32),
                torch.tensor(self.vel_est[None], dtype=torch.float32),
                torch.tensor([float(command[0])], dtype=torch.float32),
                torch.tensor([float(command[1])], dtype=torch.float32),
                torch.tensor([float(prev_s)], dtype=torch.float32),
                torch.tensor([float(prev_h)], dtype=torch.float32), dt)
        return p[0].numpy().astype(float)

    def act(self, position, dt: Optional[float] = None) -> np.ndarray:
        # The command sent on the previous step, kept for predict_next().
        self._prev_sent = (self.last_speed,
                           self.last_heading if self.last_heading is not None else 0.0)
        pos = np.asarray(position, dtype=float)[:2]
        dt = float(dt) if dt else self.dt
        vel = self._velocity(pos, dt)
        # Read by speed_factor callers: a start from rest needs more push.
        self.at_rest = float(np.linalg.norm(vel)) < self.REST_SPEED
        self._advance(pos, vel)
        self.prev_pos, self.prev_dt = pos.copy(), dt

        a, b = self._segment()
        if self.done:
            # On the goal: hold, pointing where we last pointed.
            heading = self.last_heading if self.last_heading is not None else heading_towards(pos, b)
            return np.array([0.0, heading])

        aim = self._aim(pos)
        base = heading_towards(pos, aim)
        # From rest, only "stop" or a real push. The learned model has little
        # data on starting from standstill and wrongly believes a bigger command
        # moves the ball less there, so it chose 0.017-0.025 at every start and
        # the ball could not break free of friction or climb a floor bump.
        fractions = (self.SPEED_FRACTIONS if float(np.linalg.norm(vel)) >= self.REST_SPEED
                     else self.REST_FRACTIONS)
        cruise = self.cruise
        if self.speed_factor is not None:
            cruise *= float(self.speed_factor(pos, b))
        cands = [(f * cruise, wrap(base + o))
                 for f in fractions for o in self.HEADING_OFFSETS]
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
            # held rollout: progress and cross-track. The first step follows
            # the command actually sent last; after that the candidate is its
            # own previous command.
            p, v = p0, v0
            held = []
            prev_s, prev_h = prev_sp0, prev_hd0
            for _ in range(horizon):
                p, v = batch_step(self.model, p, v, sp, hd, prev_s, prev_h, dt)
                prev_s, prev_h = sp, hd
                held.append(p.numpy().astype(float))
            # one step at the candidate, then cut: where would it come to rest?
            p1, v1 = batch_step(self.model, p0, v0, sp, hd, prev_sp0, prev_hd0, dt)
            ps, vs = p1, v1
            prev_s = sp
            rest_path = [p1.numpy().astype(float)]
            for _ in range(stop_steps):
                ps, vs = batch_step(self.model, ps, vs, zero, hd, prev_s, hd, dt)
                prev_s = zero
                rest_path.append(ps.numpy().astype(float))

        s0, _l0, L = self._along(pos[None, :], a, b)
        s0 = float(s0[0])
        first = held[0]
        s1, lat1, _ = self._along(first, a, b)
        rest = rest_path[-1]
        s_rest, _lr, _ = self._along(rest, a, b)

        cost = np.zeros(N)
        # progress over the held horizon, capped at the segment's end. Over a
        # single step it would be near zero for every candidate on the robot,
        # whose latency means the first step after a command barely moves.
        s_h, _lh, _ = self._along(held[-1], a, b)
        cost -= self.W_PROGRESS * (np.minimum(s_h, L) - s0)
        # cross-track over the held horizon
        for q in held:
            _s, lat, _ = self._along(q, a, b)
            cost += self.W_CROSS * lat ** 2
        # would cutting the drive next step leave the ball beyond the corner?
        # Resting anywhere inside the arrival radius counts as arriving, so the
        # ball may run on up to that far. Demanding it stop short of the point
        # itself forced a creep over the last few centimetres of every leg and
        # left it stalled short of the goal when the loop period was long.
        # Every turning point is priced like the goal. Straight ahead of a
        # corner is a wall, so running past one is as bad as running past the
        # goal. The goal-only version cut goal overshoot on the robot from
        # 5.1 cm to 0.0-3.5 cm, but corner 3, at the end of the 0.5 m column
        # where the ball reaches 0.20 m/s, kept the cheap pricing and ran
        # 4.1 to 4.6 cm past on all three runs that followed.
        final_leg = self.index == len(self.points) - 1
        target_rest = L + self.GOAL_RUN_ON - (self.GOAL_AIM_SHORT if final_leg else 0.0)
        over = np.maximum(s_rest - target_rest, 0.0)
        cost += self.W_OVERSHOOT * (over ** 2) + self.W_GOAL_OVERSHOOT_LIN * over
        # walls, on everything we predicted
        wall = np.zeros(N, dtype=bool)
        if self.is_free is not None:
            for i in range(N):
                pts = [h[i] for h in held] + [r[i] for r in rest_path[::3]]
                if any(not self.is_free(q) for q in pts):
                    cost[i] += self.W_WALL
                    wall[i] = True
        # mild preference for not swinging the heading about
        if self.last_heading is not None:
            cost += self.W_TURN * np.array([abs(wrap(c[1] - self.last_heading)) for c in cands])

        best = int(np.argmin(cost))
        if float(np.linalg.norm(vel)) < self.REST_SPEED and cands[best][0] <= 0.0:
            # At rest and not at the goal: never choose to stay stopped. On the
            # robot the MPC parked at corners 2 and 3 proposing 0 for up to
            # 18 s; only the occasional safety nudge moved it. Take the best
            # candidate that actually moves.
            moving = [i for i, c in enumerate(cands) if c[0] > 0.0]
            if moving:
                best = min(moving, key=lambda i: cost[i])
        speed, heading = cands[best]
        # True only when even the best candidate hits a wall, i.e. every
        # option the controller had was predicted to end in contact. That is
        # the safety layer's cue, not the controller's.
        self.last_best_hits_wall = bool(wall[best])
        self.last_heading = heading
        self.last_speed = float(speed)
        return np.array([float(speed), float(heading)])

    def note_applied(self, action) -> None:
        """Record the command actually sent, when something else overrode ours.

        The model takes the previous command as an input, so after a safety
        takeover it must start from what the robot was really told to do.
        """

        self.last_speed = float(action[0])
        self.last_heading = wrap(float(action[1]))


def load_model(path) -> DynamicsModel:
    m = DynamicsModel()
    m.load_state_dict(torch.load(str(path)))
    return m.eval()
