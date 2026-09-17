"""Lab 1 PID go-to-goal controller, as tuned on BP-9480, for use in Lab 2.

Standalone copy of `labcore/controller.py` with the tuned gains from
`labs/lab1/config/lab1_config.json` baked in as defaults, so the lab2 folder
runs on its own. Lab 1 scored RMSE 0.0747 and a final error of 0.0380 m with
exactly these numbers; nothing here is a fresh guess.

The action is `[speed_cmd, heading_cmd]` - the second element is a desired
heading ANGLE, not a turn rate. So the two loops are

    heading loop   heading_cmd = heading + Kp_h * e_h
    speed loop     speed_cmd   = Kp_p * e_p + Kd_p * de_p/dt

with Kp_h = 1.0 reducing exactly to "point straight at the goal", which is
deliberate: at Kp_h = 1 the measured heading cancels out of the command
algebraically, so the +-46 deg drift of this ball's heading reference cannot
reach the output. Kp_h is not a tuning parameter.

In Lab 2 this controller is driven by the EKF's estimate rather than the raw
observation. That is the whole point of the lab - closing the loop on a
filtered state instead of a noisy one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def goal_heading(dx: float, dy: float) -> float:
    """Heading pointing along (dx, dy) in this environment's convention.

    heading 0 -> +y and heading +pi/2 -> +x, so the atan2 arguments are
    swapped relative to the textbook atan2(dy, dx).
    """
    return math.atan2(dx, dy)


@dataclass
class ControllerState:
    integral_pos: float = 0.0
    integral_heading: float = 0.0
    prev_pos_error: Optional[float] = None
    prev_heading_error: Optional[float] = None
    arrived: bool = False
    latched_goal: Optional[tuple] = None


@dataclass
class PIDController:
    """Position + heading PID producing `[speed_cmd, heading_cmd]`.

    Defaults are the tuned BP-9480 values.
    """

    # -- tuned gains (chosen by autotune, 40 trials, best loss 0.472) -------- #
    kp_pos: float = 1.186091468853902
    ki_pos: float = 0.0          # deliberately zero: see note below
    kd_pos: float = 0.1281545543882583

    kp_heading: float = 1.0      # NOT a tuning parameter - see module docstring
    ki_heading: float = 0.0
    kd_heading: float = 0.0

    # -- safety limits, set by hand, deliberately not tuned ----------------- #
    max_speed: float = 0.45
    min_speed: float = 0.04      # floor so the command clears the deadband

    align_gate: float = 0.08984898653286887
    align_exponent: float = 1.0
    arrive_radius: float = 0.03
    brake_radius: float = 0.9751677522056599

    # -- measured coast behaviour ------------------------------------------- #
    #: Stopping distance measured as d_stop = 0.12 + 0.75 * v. The constant
    #: does NOT vanish as v -> 0: even crawling, the ball rolls 12 cm after the
    #: motors cut, against a 10 cm goal circle. That is a hardware fact.
    coast_tau: float = 0.75
    coast_bias: float = 0.12
    #: Fraction of the braking limit actually used. `set_speed` is a closed
    #: loop in firmware, so the ball keeps accelerating for a step after being
    #: told to slow. This is the headroom for that.
    brake_margin: float = 0.75
    reverse_brake_angle: float = 3.15

    #: Slowest speed the ball can actually be DRIVEN at, m/s.
    #:
    #: `min_speed` is a command, and the model converts it:
    #: actual = speed_gain * (cmd - deadband) = 3.05 * 0.04 = 0.116 m/s. Below
    #: that the motors are inside the deadband and the ball only coasts. Used
    #: to taper the constant coast term - see `compute`.
    min_drive_speed: float = 0.116

    #: Speed below which the ball counts as STOPPED, m/s.
    #:
    #: The coast model adds the full 0.12 m `coast_bias` whenever the ball is
    #: judged to be rolling. Testing that against 1e-6 means a stationary ball
    #: whose ESTIMATED speed carries a hair of numerical noise - the logs show
    #: `-0.000`, a small negative, not a true zero - is treated as rolling. The
    #: 12 cm stopping distance is then subtracted from a 5.7 cm gap, `remaining`
    #: goes to zero, the min_speed nudge is skipped, and the command sticks at
    #: zero forever.
    #:
    #: Measured: a run parked 5.7 cm from the goal and sat there for the last
    #: 16 steps commanding 0.000 with a measured speed of 0.000.
    #:
    #: 0.01 m/s is well under min_speed (0.04) and far above estimator noise,
    #: so a ball that is actually stopped gets its nudge and creeps in.
    stopped_speed: float = 0.01

    goal_move_tol: float = 0.05
    speed_feedback: bool = False
    speed_margin: float = 1.8

    #: Command-to-speed model, needed ONLY to compare a measured speed with a
    #: command. actual = speed_gain * (cmd - deadband), so the inverse is
    #: cmd = actual / speed_gain + deadband. Same identified values as the
    #: process model in EKF.py.
    speed_gain: float = 3.0498303914012186
    deadband: float = 0.0019852385383325637
    integral_limit: float = 0.5
    hold_at_goal: bool = True

    dt: float = 0.1
    state: ControllerState = field(default_factory=ControllerState)

    # ki_pos is zero on purpose. The position error is a DISTANCE, always >= 0,
    # so an integrator on it can only ever grow - it cannot unwind by changing
    # sign. It winds up and slams the output. Same reason kd_pos must stay well
    # below kp_pos: an early autotune result of kd 1.140 against kp 0.261 made
    # Kd*de/dt (negative on approach) larger than Kp*e, pinning the output at
    # the floor and making the robot crawl the whole way. Current ratio ~9:1.

    def gains(self) -> dict:
        return {name: getattr(self, name)
                for name in self.__dataclass_fields__ if name != "state"}

    def reset(self) -> None:
        self.state = ControllerState()

    def __call__(self, obs: Sequence[float], goal: Sequence[float]) -> np.ndarray:
        return self.compute(obs, goal)[0]

    def compute(self, obs: Sequence[float], goal: Sequence[float],
                stopping: bool = True, brake_distance: float = None,
                speed_cap: float = None):
        """Return (action, diagnostics). `obs` is [x, y, heading, speed, ...].

        `stopping=False` means "pass through this point, do not stop on it".

        The intermediate corners of a route are points to be driven THROUGH.
        Nothing is scored on stopping at them, and braking for each one costs
        twice: the ball slows to the motor deadband on the approach, and the
        follower then advances early - because arrival is a radius test - so
        the corner is rounded off.

        Measured on hardware with braking at every corner: corner 1 cut by
        12 cm, corner 3 by 9 cm, 31% of steps pinned at the min_speed floor,
        and a path of 1.80 m against a 2.00 m route.

        Only the final waypoint is scored on where the ball stops, so only the
        final waypoint gets the braking envelope and the arrival latch.
        """
        x, y, heading = float(obs[0]), float(obs[1]), float(obs[2])
        speed_now = abs(float(obs[3])) if len(obs) > 3 else 0.0
        gx, gy = float(goal[0]), float(goal[1])
        dt = max(self.dt, 1e-6)
        st = self.state

        dx, dy = gx - x, gy - y
        e_pos = math.hypot(dx, dy)
        theta_goal = goal_heading(dx, dy)
        e_heading = wrap_angle(theta_goal - heading)

        # When following a path, `goal` is a carrot ON the edge, a lookahead
        # ahead of the ball - so the distance to it is roughly constant and
        # says nothing about how much road is left. Braking has to be measured
        # against the real corner instead, which is what `brake_distance`
        # carries. Steering uses the carrot; stopping uses the corner.
        #
        # This used to be done by calling compute() a second time with the
        # corner as the goal and taking the smaller speed. That corrupts the
        # controller: compute() mutates prev_pos_error, the integrator and the
        # arrival latch, so the derivative term ended up differenced against
        # alternating references and the path wandered.
        e_brake = e_pos if brake_distance is None else float(brake_distance)

        # --- arrival latch ------------------------------------------------- #
        # The motor deadband means the ball cannot creep: the shortest step it
        # can take is min_speed * dt, so an arrival circle smaller than that is
        # one the robot physically cannot stop inside - it steps over, turns
        # round, steps over again, forever.
        rolling = speed_now > self.stopped_speed
        coast = self.coast_tau * speed_now if self.coast_tau > 0.0 else 0.0

        # The constant part of the stopping distance fades at low speed.
        #
        # `d_stop = 0.12 + 0.75*v` was fitted from stops at DRIVING speed, and
        # the 0.12 m constant is only meaningful over that range. Charging it
        # in full to a ball creeping at 0.02 m/s claims it will roll 12 cm,
        # which is plainly false - and it is what made the last few centimetres
        # take forever: `remaining = 0.05 - 0.12` went negative, the command
        # was cut to zero, and the approach became nudge-drift-stop at 5 mm a
        # cycle. Measured: 38 steps to cover the final 6 cm.
        #
        # So the constant is scaled by how close the ball is to the slowest
        # speed it can actually be driven at. Above that it is charged in full,
        # exactly as measured; below it, it tapers to nothing.
        bias_scale = min(1.0, speed_now / max(self.min_drive_speed, 1e-6))
        if self.coast_bias > 0.0 and rolling:
            # Only while rolling: a stationary ball has no roll to allow for,
            # and charging it 12 cm of coast it is not doing strands it short
            # of the goal with the drive commanded off.
            coast += self.coast_bias * bias_scale

        # "Cut the drive so the roll carries me in" and "I have arrived, hold"
        # are different decisions and must not share a radius.
        arrive_radius = max(self.arrive_radius, 1.5 * self.min_speed * dt)

        goal_moved = (st.latched_goal is None
                      or math.hypot(gx - st.latched_goal[0],
                                    gy - st.latched_goal[1]) > self.goal_move_tol)
        if goal_moved:
            st.arrived = False
        if stopping and e_brake <= arrive_radius:
            st.arrived = True
            st.latched_goal = (gx, gy)

        if stopping and st.arrived and self.hold_at_goal:
            st.integral_pos = 0.0
            st.prev_pos_error = e_pos
            st.prev_heading_error = e_heading
            action = np.array([0.0, wrap_angle(theta_goal)], dtype=float)
            return action, {"e_pos": e_pos, "e_heading": e_heading,
                            "theta_goal": theta_goal, "speed_cmd": 0.0,
                            "heading_cmd": float(action[1]), "gate": 0.0,
                            "arrived": True, "saturated": False}

        # --- derivative terms ---------------------------------------------- #
        d_pos = 0.0 if st.prev_pos_error is None else (e_pos - st.prev_pos_error) / dt
        if st.prev_heading_error is None:
            d_heading = 0.0
        else:
            d_heading = wrap_angle(e_heading - st.prev_heading_error) / dt

        # --- heading loop: output is an absolute heading command ----------- #
        st.integral_heading = float(np.clip(
            st.integral_heading + e_heading * dt,
            -self.integral_limit, self.integral_limit))
        heading_cmd = wrap_angle(heading
                                 + self.kp_heading * e_heading
                                 + self.ki_heading * st.integral_heading
                                 + self.kd_heading * d_heading)

        # --- speed loop ---------------------------------------------------- #
        integral_candidate = st.integral_pos + e_pos * dt
        raw = (self.kp_pos * e_pos
               + self.ki_pos * integral_candidate
               + self.kd_pos * d_pos)

        # Shed momentum before reversing: a rolling ball cannot turn round, and
        # commanding a ~180 deg change while it still carries its old velocity
        # produces an orbit that never converges.
        if (abs(e_heading) > self.reverse_brake_angle
                and speed_now > max(2.0 * self.min_speed, 1e-6)):
            st.prev_pos_error = e_pos
            st.prev_heading_error = e_heading
            action = np.array([0.0, wrap_angle(theta_goal)], dtype=float)
            return action, {"e_pos": e_pos, "e_heading": e_heading,
                            "theta_goal": theta_goal, "speed_cmd": 0.0,
                            "heading_cmd": float(action[1]), "gate": 0.0,
                            "arrived": False, "saturated": False,
                            "braking_to_turn": True}

        # Gate forward drive on alignment so the robot turns before it drives.
        cos_align = math.cos(e_heading)
        gate_raw = max(0.0, cos_align) ** max(self.align_exponent, 1e-6)
        gate = (1.0 - self.align_gate) + self.align_gate * gate_raw
        speed = raw * gate

        # Brake into the goal - only if we mean to stop on it.
        if stopping and self.brake_radius > 1e-9:
            speed = min(speed, self.max_speed * min(1.0, e_brake / self.brake_radius))

        # Never go faster than the speed it could still stop from. Inverting
        # d_stop = coast_bias + coast_tau * v gives the fastest speed that
        # still stops on the mark from here. The dt in the denominator charges
        # the step the ball keeps travelling during this decision.
        if stopping and self.coast_tau > 1e-9:
            stoppable = (max(e_brake - self.coast_bias, 0.0)
                         / (self.coast_tau + dt)) * max(self.brake_margin, 1e-3)
            speed = min(speed, stoppable)
            # Enforce it on the speed the ball HAS, not just the one commanded:
            # this robot runs faster than commanded while its firmware speed
            # loop winds up. Once over, the only useful command is zero.
            if rolling and speed_now > stoppable:
                speed = 0.0

        # Never command a step longer than what is left to travel.
        remaining = max(e_brake - coast, 0.0)
        if stopping:
            speed = min(speed, remaining / dt)

        # Cornering speed limit. A corner is not a place to stop, it is a
        # place to be SLOW - slow enough that the turn fits inside the
        # corridor, but still rolling. Applied before the min_speed floor, so
        # the ball is never driven below what clears the motor deadband.
        if speed_cap is not None:
            speed = min(speed, float(speed_cap))

        saturated = speed > self.max_speed or speed < 0.0
        speed = float(np.clip(speed, 0.0, self.max_speed))

        # Clear the motor deadband while there is still ground to cover. A
        # Sphero steers by rolling, so a stopped ball cannot turn - the floor
        # must NOT be withdrawn when the heading error is large.
        if e_brake > arrive_radius and (remaining > 0.0 or not stopping):
            speed = max(speed, self.min_speed)

        # Speed feedback: cut the drive if the ball is ALREADY running faster
        # than the plan, because on this robot a held command accelerates.
        #
        # Both sides of this test must be in the same units, and they were not.
        # `speed_now` is metres per second, while `speed` and `min_speed` are
        # COMMANDS - and a command is not a speed. With a 0.16 m/s top speed the
        # threshold worked out at about 0.098, which was then compared against a
        # real speed whose FLOOR is 0.116 m/s: the cut fired the instant the ball
        # moved under power. Measured on the 3 Sep hardware run: the command was
        # zero on 46 of 169 steps, the ball averaged 0.037 m/s against a 0.116
        # m/s minimum driven speed, and a 0.53 m leg took 42 steps instead of 5.
        #
        # Converting the measured speed into command units puts the test back on
        # its feet: it now only fires when the ball is genuinely running away.
        speed_now_cmd = speed_now / max(self.speed_gain, 1e-9) + self.deadband
        if (self.speed_feedback and rolling
                and speed_now_cmd > max(speed, self.min_speed) * self.speed_margin):
            speed = 0.0

        # Conditional integration: freeze the integrator while saturated.
        if not saturated:
            st.integral_pos = float(np.clip(integral_candidate,
                                            -self.integral_limit,
                                            self.integral_limit))

        st.prev_pos_error = e_pos
        st.prev_heading_error = e_heading

        action = np.array([speed, heading_cmd], dtype=float)
        return action, {"e_pos": e_pos, "e_heading": e_heading,
                        "theta_goal": theta_goal, "speed_cmd": speed,
                        "heading_cmd": heading_cmd, "gate": gate,
                        "arrived": False, "arrive_radius": arrive_radius,
                        "saturated": bool(saturated),
                        "integral_pos": st.integral_pos, "d_pos": d_pos}


# --------------------------------------------------------------------------- #
# Waypoint following
# --------------------------------------------------------------------------- #


class WaypointFollower:
    """Drive a closed list of waypoints with the PID above.

    The demo asks for a rectangle: four waypoints, visited automatically, with
    the inferred path judged on how rectangular it looks. Two details decide
    that:

    `capture_radius` - how close counts as "reached". Too tight and the ball
    orbits a corner it cannot stop on (12 cm of irreducible coast, remember);
    too loose and the corners round off into an oval and stop looking like a
    rectangle. 0.10 m is the goal tolerance and a sensible start.

    `settle_steps` - hold at each corner for a few steps before advancing. A
    ball that rounds the corner at speed cuts it; pausing lets the coast bleed
    off so the next leg starts pointed the right way. This is what makes the
    path read as a rectangle rather than a racetrack oval.
    """

    def __init__(self, waypoints, controller: PIDController,
                 capture_radius: float = 0.10, settle_steps: int = 3,
                 loop: bool = False, pass_through: bool = False,
                 closest_approach: bool = True, lookahead: float = 0.20,
                 follow_path: bool = True, corner_capture: float = None,
                 corner_speed: float = 0.0, corner_slow_dist: float = 0.35,
                 sharp: bool = False, turn_steps: int = 2,
                 stop_speed_eps: float = 0.025, settle_max: int = 8):
        self.waypoints = [tuple(float(v) for v in wp[:2]) for wp in waypoints]
        self.controller = controller
        self.settle_steps = int(settle_steps)

        #: Square corners. Every waypoint is driven into and STOPPED ON, like
        #: the goal, instead of being flown through.
        #:
        #: The default rounds the corners and cannot help it: the capture
        #: radius is clamped up to the coasting distance (0.15 m on this ball),
        #: so the follower switches to the next waypoint 15 cm BEFORE the
        #: corner and the ball turns early. That is the arc in the demo trace.
        #:
        #: In sharp mode the ball brakes to a halt on the corner - the arrival
        #: latch fires at 4 cm, not 15 - and then spins the drive unit to the
        #: new heading at ZERO SPEED for `turn_steps` steps before driving on.
        #: Spinning at zero speed turns the drive unit inside the shell without
        #: moving the ball, which is the same mechanism the aim step uses, so
        #: the corner comes out square rather than as a fillet.
        #:
        #: It costs time: roughly 4 extra steps per corner, about 11 s over the
        #: four corners at dt 0.68.
        self.sharp = bool(sharp)
        self.turn_steps = int(turn_steps)
        #: Steps of in-place rotation still owed at the corner just reached.
        self.turning = 0

        #: A corner counts as reached only once the ball is actually STOPPED.
        #:
        #: Arrival on distance alone is not enough for a square corner. The
        #: latch fires at 4 cm and at that moment the ball is usually still
        #: rolling, so it advances, the drive unit turns, and the ball arcs
        #: through the corner carrying the speed it had. Measured on the 3 Sep
        #: run: every corner was reached to within 7 mm and the trace still
        #: showed a fillet at each one. Waiting for the roll to die is what
        #: turns that fillet into a pivot.
        self.stop_speed_eps = float(stop_speed_eps)
        #: Never wait longer than this. On a slope the ball may never be
        #: perfectly still, and a corner is not worth hanging the run on.
        self.settle_max = int(settle_max)

        # A capture radius smaller than the stopping distance is not "tighter
        # tracking", it is a deadlock. The ball cannot be placed more precisely
        # than it can stop: measured d_stop = 0.12 + 0.75*v, and the 0.12 does
        # not vanish as v -> 0. Ask it to arrive within 2 cm and it overshoots,
        # turns round, overshoots the other way, and never advances.
        #
        # Seen on hardware: 200 steps spent orbiting the first waypoint at
        # ~(0, 0.47), speed oscillating between -0.174 and +0.144 m/s with a
        # mean of -0.001, heading swinging over 3.7 rad. The run never reached
        # waypoint 2.
        #
        # Simulation does NOT reproduce this, because the sim controller runs
        # with coast_bias = 0 - there is no stiction roll to model - so a value
        # that works in sim can still deadlock the real ball. Hence the clamp
        # is derived from the controller's own coast model rather than tested
        # for empirically.
        #
        # Corners do not need a tight radius anyway: the robot should pass
        # through them, not stop on them. Only the final waypoint needs
        # precision, and the PID's arrival latch handles that separately.
        self.requested_capture = float(capture_radius)
        floor = (getattr(controller, "coast_bias", 0.0)
                 + getattr(controller, "arrive_radius", 0.0))
        self.capture_radius = max(self.requested_capture, floor)
        self.capture_clamped = self.capture_radius > self.requested_capture + 1e-9
        self.loop = bool(loop)
        #: Drive THROUGH intermediate corners without braking for them.
        self.pass_through = bool(pass_through)
        #: Advance at closest approach rather than on entering capture_radius.
        self.closest_approach = bool(closest_approach)
        #: steer at the edge rather than at the corner
        self.follow_path = bool(follow_path)
        #: Capture radius for CORNERS only. The clamp to the coasting distance
        #: is what the FINAL waypoint needs - it has to stop there. Corners are
        #: driven through, and a corner radius of 0.15 m means the turn starts
        #: 15 cm short of the corner: measured, each leg began 11-12 cm before
        #: its corner and cut across, rounding the rectangle by that much.
        #:
        #: A tighter corner radius used to deadlock in the coast dead band, but
        #: the `stopped_speed` fix cures that - a stopped ball now gets its
        #: min_speed nudge and creeps the last few centimetres in.
        self.corner_capture = float(corner_capture) if corner_capture else None
        #: Speed to carry THROUGH an intermediate corner, m/s. 0 disables and
        #: falls back to braking for the corner as though stopping on it.
        #:
        #: A corner has only two bad modes without this. Brake fully and the
        #: ball stops, rotates on the spot, then lurches off in the new
        #: direction from wherever it stopped - measured speed at corners fell
        #: to 0.001-0.005 m/s, and the lurch is what carries it through the
        #: wall. Skip the braking entirely and it arrives at full speed and
        #: swings 9 cm wide.
        #:
        #: The answer is neither: slow to a cornering speed, stay rolling
        #: through the turn, accelerate out. A Sphero steers by rolling, so it
        #: must keep moving to change direction at all.
        self.corner_speed = float(corner_speed)
        #: Distance from the corner at which the slowdown starts, m.
        #:
        #: This must be at least the ball's STOPPING distance at cruise, or the
        #: cap never takes effect before the turn. `set_speed` is a firmware
        #: target, not a brake: measured on hardware, the command had dropped
        #: to 0.040 while the ball was still doing 0.171 m/s, and shedding that
        #: needs 0.12 + 0.75*0.17 = 0.25 m of travel. With the slowdown
        #: starting at 0.22 m there was no road left to slow down in, so the
        #: ball advanced to the next waypoint at full speed and coasted 7 cm
        #: past the old edge. Corner speeds of 0.06 and 0.01 overshot
        #: identically for exactly this reason - neither had taken effect.
        self.corner_slow_dist = float(corner_slow_dist)
        #: Command-to-speed gain, so `corner_speed` can be given in real m/s.
        #: The identified Lab 1 value; actual = speed_gain * (cmd - deadband).
        self.speed_gain = 3.0498303914012186
        self.index = 0
        self.settle = 0
        self.finished = False
        self.prev_dist = None
        self.receding = 0
        self.stalled = 0
        #: start of the edge currently being followed
        self.segment_start = tuple(self.waypoints[-1])
        #: How far ahead on the edge to aim, metres.
        #:
        #: This is the pure-pursuit lookahead and it trades cornering against
        #: line-holding. Long, and the aim point sits well up the edge, so a
        #: ball that is off to one side is steered at a shallow angle and the
        #: cross-track error decays slowly - the leg bows outward. Short, and it
        #: is corrected hard, but too short oscillates: the ball weaves about
        #: the line because it is always steering at a point it is nearly on
        #: top of.
        #:
        #: 0.20 m leaves the long legs bowing - measured on the 3 Sep runs,
        #: leg 0 held the line to 0.1 cm while legs 1-3 ran 0.8-1.4 cm mean and
        #: up to 3.8 cm, and leg 0 is the only one the ball starts already on.
        #: 0.12 m fixes that in simulation (cross-track mean 0.70 -> 0.62 cm,
        #: no weaving, corners and final error unchanged).
        #:
        #: HELD AT 0.20 FOR THE DEMO ANYWAY. Every one of the 16 clean hardware
        #: runs was driven at 0.20; 0.12 has never touched the robot. With a
        #: single scored attempt and no practice run, an untested steering
        #: change is not worth 0.08 cm of mean cross-track. Set it to 0.12
        #: after the demo, or once there is a run to check it against.
        self.lookahead = float(lookahead)
        #: consecutive motionless steps near a corner before giving up on it
        self.stall_steps = 15
        # Ignore millimetre-scale locator noise when detecting that the ball
        # has passed a corner. A real pass grows the distance for more than one
        # sample while the ball is still rolling.
        self.recede_tol = 0.005
        self.recede_steps = 2

    def reset(self, start=None):
        self.index = 0
        self.settle = 0
        self.finished = False
        self.prev_dist = None
        self.receding = 0
        self.stalled = 0
        # The first edge runs from wherever the ball actually is to waypoint 1.
        if start is not None:
            self.segment_start = (float(start[0]), float(start[1]))
        else:
            self.segment_start = tuple(self.waypoints[-1])
        self.controller.reset()

    def _brake_for_target(self) -> bool:
        """Should the controller brake for the point it is currently heading to?

        Always for the final waypoint - that one is scored on where the ball
        stops. For corners it depends on `pass_through`: skipping the brake
        keeps the speed up but the ball then flies well past the corner, since
        it still coasts 12 cm plus 0.75 s of speed. Measured with pass_through
        on: the route came to 4.10 m against a 2.00 m rectangle.
        """
        return self._is_final() or not self.pass_through

    def _carrot(self, px, py):
        """Aim point on the current EDGE, not the corner at the end of it.

        Aiming straight at the next corner is waypoint seeking. The moment the
        ball is off the edge - and it always is, after coasting through the
        previous corner - the straight line to the next corner cuts diagonally
        across the rectangle. In open space that is untidy; in the demo track,
        where the route runs between walls, it is a collision.

        So steer at a point ON the segment, a lookahead ahead of wherever the
        ball projects onto it. Standard pure pursuit: cross-track error decays
        instead of being ignored, and the path hugs the edge.

        Measured without it: mean 9.6 cm off the outline, max 20 cm, and the
        corners missed by 18-30 cm.
        """
        ax, ay = self.segment_start
        bx, by = self.target
        dx, dy = bx - ax, by - ay
        length2 = dx * dx + dy * dy
        if length2 < 1e-9:
            return self.target
        # Where the ball sits along the segment, clamped to it.
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
        # Look ahead along the segment, never past the far end.
        t_goal = min(1.0, t + self.lookahead / math.sqrt(length2))
        return (ax + t_goal * dx, ay + t_goal * dy)

    def _is_final(self) -> bool:
        """True when the current target is the one the run is scored on."""
        return self.index >= len(self.waypoints) - 1

    def _capture_for_target(self) -> float:
        """Capture radius for the point currently being driven to.

        The clamp to the coasting distance applies to EVERY waypoint, corners
        included. It is tempting to let corners use a tighter radius, since
        they are driven through rather than stopped on - but the braking
        envelope commands exactly zero inside `coast_bias` (0.12 m): there
        `stoppable` is 0, and the min_speed floor is skipped because
        `remaining` is 0 too. A corner radius below 0.12 m therefore sits
        inside a dead band the ball cannot cross.

        Measured on hardware with a 0.05 m corner radius: speed_cmd was zero on
        148 of 200 steps, and 140 of those were between 0.05 and 0.12 m from a
        waypoint. The ball stalled at ~8 cm out and never advanced; the run
        reached corner 1 only at step 133 and never finished. Closest approach
        cannot rescue it either, because a stationary ball's distance never
        starts increasing.

        Simulation does not reproduce this - the sim controller runs with
        coast_bias = 0, so there is no dead band there at all.

        Corner accuracy is therefore floored at roughly the coasting distance.
        That is a hardware fact, like the 12 cm stop, not a tuning choice.
        """
        if self._is_final():
            return self.capture_radius        # clamped: the ball stops here
        # Corners use the radius that was actually asked for. Measured on the
        # fake: 0.15 (clamped) put the path 0.061 m off the outline at worst,
        # 0.10 put it 0.017 m off - the turn starts at the corner instead of
        # 15 cm short of it.
        return self.corner_capture or self.requested_capture

    def _advance(self):
        self.settle = 0
        self.prev_dist = None
        self.receding = 0
        self.stalled = 0
        # Square corners: rotate on the spot before taking the next edge.
        if self.sharp and not self.finished:
            self.turning = self.turn_steps
        # The edge we are about to follow starts at the corner we just left.
        self.segment_start = tuple(self.target)
        self.index += 1
        # A new goal must release the arrival latch, or the controller holds at
        # the corner it already reached.
        self.controller.reset()
        if self.index >= len(self.waypoints):
            if self.loop:
                self.index = 0
            else:
                self.index = len(self.waypoints) - 1
                self.finished = True

    @property
    def target(self):
        if self.index >= len(self.waypoints):
            return self.waypoints[-1]
        return self.waypoints[self.index]

    def __call__(self, state):
        """`state` is the EKF estimate - not the raw observation."""
        goal = self.target
        dist = math.hypot(goal[0] - float(state[0]), goal[1] - float(state[1]))
        final = self._is_final()

        # --- square corner: rotate on the spot, then drive ----------------- #
        # Speed stays at zero so the drive unit turns inside the shell and the
        # ball does not move. `run_mission` sees `turning` and lets this
        # heading through instead of holding the last driven one.
        if self.turning > 0:
            self.turning -= 1
            aim = goal_heading(goal[0] - float(state[0]),
                               goal[1] - float(state[1]))
            self.controller.reset()
            return (np.array([0.0, aim], dtype=float), self.target, dist)

        # In sharp mode EVERY waypoint is stopped on, so none of the
        # fly-through machinery below applies: no closest-approach turn, no
        # capture-radius early advance, no corner speed cap.
        if self.sharp and not self.finished:
            # Corners do not need the goal's settle. Waiting `settle_steps` on
            # each of four corners costs 8 s at dt 0.68, and nothing is scored
            # on how long the ball sat on a corner - only on where it finally
            # stops. One confirming step is enough there.
            settle_needed = self.settle_steps if final else 1

            # Stall breaker. Sharp mode advances on ARRIVAL, so a ball that
            # cannot reach the corner - jammed on the barrier, stuck on a lip -
            # never advances and burns the rest of the run standing still.
            # Measured on the 3 Sep run: it overshot corner 2 by 3.6 cm into
            # the wall and sat there for 130 of 201 steps. Give up on a corner
            # that is not getting any closer and take the next leg; the route
            # at least continues, and the goal is still scored.
            if not final:
                if self.prev_dist is not None and abs(dist - self.prev_dist) < 0.004:
                    self.stalled += 1
                else:
                    self.stalled = 0
                self.prev_dist = dist
                # Never abandon a distant waypoint merely because the robot
                # stopped moving. In the two 10 Sep Lab 3 hardware runs this
                # skipped (1,0) from 0.31 m away, then aimed through a wall.
                # The fallback is only legitimate inside the physically
                # achievable corner envelope.
                if (self.stalled >= self.stall_steps
                        and dist <= self.capture_radius):
                    self.stalled = 0
                    self._advance()
                    return (self.controller.compute(state, self.target,
                                                    stopping=True)[0],
                            self.target, dist)

            # Steer at the carrot ON the edge, brake against the real corner.
            # Aiming straight at the corner instead lets the ball cut the
            # diagonal from wherever it is: measured, that took the corridor
            # from 0.099 m to 0.244 m and put it 12 cm off the line. Squaring
            # the corner is about where it STOPS, not about where it steers.
            aim = (self._carrot(float(state[0]), float(state[1]))
                   if self.follow_path else self.target)
            action, _diag = self.controller.compute(state, aim, stopping=True,
                                                    brake_distance=dist)
            if self.controller.state.arrived:
                self.settle += 1
                v = abs(float(state[3])) if len(state) > 3 else 0.0
                stopped = v <= self.stop_speed_eps
                if (self.settle >= settle_needed
                        and (stopped or self.settle >= self.settle_max)):
                    self._advance()
                    # Do NOT return the action computed above. It was aimed at
                    # the corner we have just left, and by the time arrival is
                    # confirmed the ball has usually rolled PAST that corner -
                    # so the stale action points backwards and is issued with
                    # the drive on.
                    #
                    # Measured on the 11 Sep official Lab 3 run, step 40: the
                    # ball sat 4.4 cm beyond the (-0.25, 0) corner, the
                    # follower advanced to (-0.25, 0.5), and the returned
                    # command was heading -94 deg at speed 0.040 - a 190 deg
                    # reversal held for one powered step, immediately before
                    # the leg that departed at an angle and jammed on a wall.
                    #
                    # The next leg begins with a square-corner pivot anyway,
                    # so start it here: zero speed, aimed at the new target.
                    if self.turning > 0:
                        self.turning -= 1
                    aim_next = goal_heading(self.target[0] - float(state[0]),
                                            self.target[1] - float(state[1]))
                    self.controller.reset()
                    return (np.array([0.0, aim_next], dtype=float),
                            self.target, dist)
            else:
                self.settle = 0
            return action, self.target, dist

        # Intermediate corners advance at CLOSEST APPROACH, not on entering a
        # radius. The radius test fires the moment the ball is anywhere within
        # capture_radius - 15 cm on hardware, because that is the coasting
        # distance - so the turn starts early and the corner is rounded off.
        # Measured: corner 1 cut by 12 cm, corner 3 by 9 cm, path 1.80 m of
        # 2.00 m.
        #
        # Waiting until the distance starts growing again means the ball has
        # actually reached the corner before it turns. The neighbourhood test
        # keeps noise on a distant leg from triggering it.
        # Stall breaker. If the ball has stopped moving near a corner it will
        # never satisfy either the radius test or the closest-approach test, so
        # advance anyway rather than burning the whole run on one waypoint.
        if not final and not self.finished:
            if self.prev_dist is not None and abs(dist - self.prev_dist) < 0.002:
                self.stalled += 1
            else:
                self.stalled = 0
            if (self.stalled >= self.stall_steps
                    and dist < self._capture_for_target() * 1.5):
                self.stalled = 0
                self._advance()
                action, _d = self.controller.compute(state, self.target,
                                                     stopping=self._brake_for_target())
                return action, self.target, dist

        speed_now = abs(float(state[3])) if len(state) > 3 else None
        rolling_now = True if speed_now is None else speed_now > 0.02
        if (not final and not self.finished and self.prev_dist is not None
                and dist > self.prev_dist + self.recede_tol):
            self.receding += 1
        else:
            self.receding = 0

        if (self.closest_approach
                and not final and not self.finished and rolling_now
                and self.receding >= self.recede_steps
                and dist < max(self._capture_for_target(), 0.05) * 1.2):
            self._advance()
            action, _d = self.controller.compute(state, self.target,
                                                 stopping=self._brake_for_target())
            return action, self.target, dist
        self.prev_dist = dist

        if not final and dist <= self._capture_for_target() and not self.finished:
            # A corner reached head-on, before any closest-approach turn.
            self._advance()
            action, _d = self.controller.compute(state, self.target,
                                                 stopping=self._brake_for_target())
            return action, self.target, dist

        if final and not self.finished:
            # The GOAL is not a corner, and must not inherit the corner rule.
            #
            # The capture radius is clamped to the coasting distance so the
            # ball can advance past a corner it cannot stop on. Applying that
            # to the goal ends the run 15 cm out: the follower declares arrival,
            # commands zero, and the ball coasts to a halt wherever it lands.
            # Measured: the command went to zero at 0.188 m and the run finished
            # 7.1 cm from the goal, having stopped driving entirely.
            #
            # The goal instead gets driven into until the controller's own
            # arrival latch fires at `arrive_radius` (3 cm). The braking
            # envelope is built for exactly this: cut the drive a stopping
            # distance short and let the roll carry it in, then nudge at
            # min_speed if it stalls outside. Lab 1 reached 0.038 m this way.
            action, _diag = self.controller.compute(state, self.target,
                                                    stopping=True)
            if self.controller.state.arrived:
                self.settle += 1
                if self.settle >= self.settle_steps:
                    self._advance()
            else:
                self.settle = 0
            return action, self.target, dist

        if self.follow_path:
            aim = self._carrot(float(state[0]), float(state[1]))
        else:
            aim = self.target

        # Ease into an intermediate corner rather than braking to a halt on it.
        # The cap ramps from full speed at `corner_slow_dist` down to
        # `corner_speed` at the corner itself, so the ball is already turning
        # while it still has the speed to steer with.
        cap = None
        brake = self._brake_for_target()
        if self.corner_speed > 0.0 and not final:
            brake = False                       # the cap replaces the stop
            # The ramp must REACH corner_speed before the turn starts, not at
            # the corner point itself - the follower advances as soon as the
            # ball is within the capture radius, so the last `capture` metres
            # never happen. Ramping to the corner instead of to the capture
            # circle left the cap at 0.25 m/s where 0.08 was asked for, and
            # corner_speed 0.05 / 0.08 / 0.12 all behaved identically.
            turn_at = self._capture_for_target()
            if dist < self.corner_slow_dist:
                span = max(self.corner_slow_dist - turn_at, 1e-6)
                t = max(0.0, min(1.0, (dist - turn_at) / span))
                # corner_speed is a REAL speed in m/s; the controller works in
                # command units, where actual = speed_gain * (cmd - deadband).
                # Asking for 0.04 in command units is really asking for
                # 3.05 * 0.04 = 0.12 m/s, which is roughly where the ball
                # already was - so the cap looked active and changed nothing.
                cap = self.corner_speed / max(self.speed_gain, 1e-6)
                cap = cap + t * (self.controller.max_speed - cap)

        # ONE call: steer at the carrot, brake against the distance to the
        # real corner. Two calls would mutate the controller state twice.
        action, _diag = self.controller.compute(
            state, aim, stopping=brake, brake_distance=dist, speed_cap=cap)
        return action, self.target, dist


def round_corners(points, radius, per_corner=6, closed=True):
    """Replace each sharp corner with a circular fillet of `radius`.

    A 90-degree turn driven at speed cannot be sharp: the ball coasts 12 cm
    after the motors cut, so it swings WIDE through every corner and crosses
    the outer wall. Measured on hardware, the path passed 7-9 cm from each
    corner point with excursions beyond the corridor.

    Cutting the corner with an arc moves the path INWARD, away from the outer
    wall, and lets the ball carry speed through the turn instead of lurching.
    It also matches the real track, whose centre line is a rounded rectangle,
    not a sharp one.

    Geometry: for a corner B between A and C with interior angle theta, the
    fillet leaves AB at distance `radius / tan(theta/2)` from B and rejoins BC
    at the same distance. For a right angle that is exactly `radius`. The arc
    centre sits on the bisector at `radius / sin(theta/2)`.

    A 90-degree fillet departs from the sharp corner by `radius * (sqrt(2) - 1)`,
    about 0.41 * radius - so the deviation from the ideal rectangle is
    predictable and chosen, rather than whatever the coast happens to produce.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    radius = float(radius)
    if radius <= 0 or len(pts) < 3:
        return list(pts)

    n = len(pts)
    out = []
    span = range(n) if closed else range(1, n - 1)
    if not closed:
        out.append(pts[0])

    for i in span:
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]

        v1 = (a[0] - b[0], a[1] - b[1])
        v2 = (c[0] - b[0], c[1] - b[1])
        l1 = math.hypot(*v1)
        l2 = math.hypot(*v2)
        if l1 < 1e-9 or l2 < 1e-9:
            out.append(b)
            continue
        u1 = (v1[0] / l1, v1[1] / l1)
        u2 = (v2[0] / l2, v2[1] / l2)

        cos_t = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
        theta = math.acos(cos_t)                 # interior angle at B
        if theta < 1e-6 or abs(math.pi - theta) < 1e-6:
            out.append(b)                        # straight through, nothing to fillet
            continue

        # Never eat more than half of either adjoining leg.
        tangent = radius / math.tan(theta / 2.0)
        tangent = min(tangent, 0.5 * l1, 0.5 * l2)
        r_eff = tangent * math.tan(theta / 2.0)

        p1 = (b[0] + u1[0] * tangent, b[1] + u1[1] * tangent)   # arc start, on BA
        p2 = (b[0] + u2[0] * tangent, b[1] + u2[1] * tangent)   # arc end, on BC

        bis = (u1[0] + u2[0], u1[1] + u2[1])
        lb = math.hypot(*bis)
        if lb < 1e-9:
            out.append(b)
            continue
        bis = (bis[0] / lb, bis[1] / lb)
        centre_d = r_eff / math.sin(theta / 2.0)
        cx = b[0] + bis[0] * centre_d
        cy = b[1] + bis[1] * centre_d

        a1 = math.atan2(p1[1] - cy, p1[0] - cx)
        a2 = math.atan2(p2[1] - cy, p2[0] - cx)
        sweep = a2 - a1
        while sweep > math.pi:
            sweep -= 2 * math.pi
        while sweep < -math.pi:
            sweep += 2 * math.pi

        steps = max(2, int(per_corner))
        for k in range(steps + 1):
            ang = a1 + sweep * (k / steps)
            out.append((cx + r_eff * math.cos(ang), cy + r_eff * math.sin(ang)))

    if not closed:
        out.append(pts[-1])
    return out


def rectangle(width: float, height: float, centre=(0.0, 0.0)):
    """Four corners of a rectangle, anticlockwise from the bottom-left.

    Centred on `centre`. Kept for the odd case where a symmetric box is what
    you want; the demo route is anchored, not centred - see below.
    """
    cx, cy = float(centre[0]), float(centre[1])
    hw, hh = float(width) / 2.0, float(height) / 2.0
    return [(cx - hw, cy - hh), (cx + hw, cy - hh),
            (cx + hw, cy + hh), (cx - hw, cy + hh)]


def rectangle_from_origin(width: float, height: float, origin=(0.0, 0.0)):
    """The published M2 route, generalised to any width and height.

        (ox, oy+h) -> (ox+w, oy+h) -> (ox+w, oy) -> (ox, oy)

    Anchored at the start rather than centred, because that is the shape the
    unit actually set: the robot begins at the origin facing +y, drives out
    along +y, across, back down, and finishes where it started. At
    width = height = 0.5 this reproduces the published waypoints exactly, so
    changing the size scales the real route instead of replacing it with a
    different shape.

    The first leg is along +y, which is heading 0 in this environment - so
    "place the robot facing the first waypoint" stays true at any size.
    """
    ox, oy = float(origin[0]), float(origin[1])
    w, h = float(width), float(height)
    return [(ox, oy + h), (ox + w, oy + h), (ox + w, oy), (ox, oy)]
