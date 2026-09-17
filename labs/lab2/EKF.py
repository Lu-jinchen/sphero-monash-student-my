"""Lab 2 - EKF odometry for the Sphero BOLT+ (BP-9480).

Self-contained on purpose: this file imports nothing from `labcore`, so it runs
from the lab2 directory on a lab machine with only numpy available. The values
below are the ones identified and tuned in Lab 1 - they are copied here rather
than imported so the file can be read and marked on its own.

What Lab 2 is scored on
-----------------------
Not accuracy. Consistency. The marker computes, per step,

    d_k = sqrt( e^T P_k^-1 e )        e = truth - estimate

    mean d          <= 4.0
    chi2 pass rate  >= 0.90     of steps with d^2 <= 5.991 (chi2, 2 DoF, 95%)

A filter that tracks perfectly but reports P = 1e-8 FAILS both. A filter that
wanders 5 cm but reports sigma = 5 cm passes. The covariance is the deliverable.

The second threshold binds, not the first: 90% of steps under d = 2.447 is much
tighter than a mean of 4.0. The target is per-axis error roughly equal to
per-axis sigma, which puts d^2 near 2, the mean of a chi-square with 2 DoF.

Process model
-------------
The Lab 1 M2 "lag" model, with the identified parameters. Three effects that a
plain unicycle misses and that dominate this ball:

    speed lag        the shell takes time to spin up (tau_speed)
    turn rate limit  the heading servo cannot snap (max_turn_rate)
    deadband         small commands produce no motion at all

Jacobian
--------
Numerical, by central differences. The model is piecewise - the deadband and
the alignment gate are hard switches - so an analytic derivative is simply
wrong exactly at the moments the filter most needs it right. Eight extra model
evaluations per step is nothing next to a BLE round trip.
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- #
# Identified parameters - Lab 1, robot BP-9480
# --------------------------------------------------------------------------- #

#: M2 lag model, fitted to real runs. These are MEASURED properties of the
#: physical ball, not choices - see labs/lab1/config/lab1_config.json.
LAG_PARAMS = {
    "speed_gain":        3.0498303914012186,   # (m/s) per unit speed_cmd
    "vel_limit":         0.16019408000280883,  # m/s, physical top speed
    "tau_speed":         0.5145514618855662,   # s, speed response
    "deadband":          0.0019852385383325637,
    "max_turn_rate":     12.0,                 # rad/s
    "tau_heading":       0.05,                 # s, heading response
    "heading_tol":       0.35,                 # rad, drive cut above this
    "turn_speed_factor": 0.9999999999,
    "heading_bias":      0.037599846230111825,  # rad, constant offset
}

#: Loop period. The simulator skeleton runs at 0.1 s; the real robot loop is
#: BLE-bound at about 0.68 s. A model fitted at one dt does not describe the
#: same robot sampled at the other, so this is set explicitly by the runner
#: via set_dt() rather than assumed.
DT = 0.1

ANGLE_INDEX = 2     # heading, the one state that must be wrapped
N_STATES = 4        # [x, y, heading, speed]

#: chi-square critical values by degrees of freedom
CHI2_95 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488}
CHI2_05 = {1: 0.004, 2: 0.103, 3: 0.352, 4: 0.711}


def set_dt(dt):
    """Set the loop period used by dynamics(). Call once, before the run."""
    global DT
    DT = float(dt)
    return DT


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def wrap_state_diff(diff):
    """Difference between two states, with the heading component wrapped."""
    out = np.asarray(diff, dtype=float).copy()
    if out.shape[-1] > ANGLE_INDEX:
        out[..., ANGLE_INDEX] = wrap_angle(out[..., ANGLE_INDEX])
    return out


# --------------------------------------------------------------------------- #
# Process model - Lab 1 M2 (first-order lag + deadband + turn rate limit)
# --------------------------------------------------------------------------- #


def dynamics(state, action, dt=None, params=None):
    """Next state from current state and action, using the Lab 1 lag model.

        action = [speed_cmd, heading_cmd]        heading_cmd is an ANGLE
        state  = [x, y, heading, speed]

    Convention, matching SpheroEnv: heading 0 -> +y, heading pi/2 -> +x, so
    dx = v*sin(theta) and dy = v*cos(theta). Do not "fix" this to the textbook
    form; the environment and the locator both use it.

    This is passed to SpheroEnv(dynamics=...) as well as used by the EKF, so
    the filter's process model and the simulator's truth model are the same
    function - which is the point of Lab 1 feeding Lab 2.
    """
    dt = DT if dt is None else float(dt)
    p = LAG_PARAMS if params is None else params

    x, y, heading, speed = (float(v) for v in np.asarray(state, dtype=float)[:4])
    speed_cmd, heading_cmd = float(action[0]), float(action[1])

    desired_heading = wrap_angle(heading_cmd + p["heading_bias"])

    # Heading: first-order response, rate limited.
    heading_error = wrap_angle(desired_heading - heading)
    alpha_h = float(np.clip(dt / max(p["tau_heading"], 1e-6), 0.0, 1.0))
    step_h = float(np.clip(alpha_h * heading_error,
                           -p["max_turn_rate"] * dt, p["max_turn_rate"] * dt))
    heading_new = wrap_angle(heading + step_h)

    # Speed: deadband on the command, alignment gate, first-order lag.
    magnitude = abs(speed_cmd)
    if magnitude <= p["deadband"]:
        target = 0.0
    else:
        effective = math.copysign(magnitude - p["deadband"], speed_cmd)
        target = p["speed_gain"] * effective

    residual = abs(wrap_angle(desired_heading - heading_new))
    if residual > p["heading_tol"]:
        target *= p["turn_speed_factor"]

    target = float(np.clip(target, -p["vel_limit"], p["vel_limit"]))
    alpha_v = float(np.clip(dt / max(p["tau_speed"], 1e-6), 0.0, 1.0))
    speed_new = speed + alpha_v * (target - speed)
    speed_new = float(np.clip(speed_new, -p["vel_limit"], p["vel_limit"]))

    x_new = x + speed_new * math.sin(heading_new) * dt
    y_new = y + speed_new * math.cos(heading_new) * dt
    # float64, NOT float32. The EKF differentiates this function by central
    # differences with a step of 1e-5; float32 resolves about 1.2e-7, so the
    # subtraction of two nearly-equal float32 values keeps only ~1% relative
    # precision. Measured: F[2,2] came out 0.9894 where the exact value is 1.0,
    # and that 1% error entered P on every single step. The environment casts
    # back to float32 itself, so nothing downstream cares.
    return np.array([x_new, y_new, heading_new, speed_new], dtype=float)


# --------------------------------------------------------------------------- #
# Measurement models
# --------------------------------------------------------------------------- #


class MeasurementModel:
    """One h(x) plus the R that goes with it.

    `indices` selects which state elements the sensor observes, which makes H
    a constant selection matrix. That is correct here because every sensor on
    this robot observes a state directly: position from the locator, heading
    from the fused IMU, speed from the wheel encoders. A genuinely non-linear
    h would need its own Jacobian.
    """

    def __init__(self, name, indices, noise, angle_indices=()):
        self.name = name
        self.indices = tuple(int(i) for i in indices)
        self.noise = tuple(float(v) for v in noise)     # VARIANCE per element
        self.angle_indices = tuple(int(i) for i in angle_indices)

    @property
    def dof(self):
        return len(self.indices)

    def h(self, state):
        return np.asarray(state, dtype=float)[list(self.indices)]

    def H(self, n=N_STATES):
        H = np.zeros((len(self.indices), n))
        for row, idx in enumerate(self.indices):
            H[row, idx] = 1.0
        return H

    def R(self):
        return np.diag(np.asarray(self.noise, dtype=float))

    def innovation(self, z, state):
        y = np.asarray(z, dtype=float).ravel() - self.h(state)
        for i in self.angle_indices:
            y[i] = wrap_angle(y[i])
        return y

    def scaled(self, factor):
        """Same sensor, R multiplied by `factor`.

        Tuning R means asking "how much do I trust this sensor", and the answer
        is nearly always a multiple of the nominal spec rather than a fresh set
        of variances. Scaling the whole matrix keeps the relative weighting
        between channels while moving overall trust, which is the axis the NIS
        plot responds to.
        """
        factor = max(float(factor), 1e-9)
        return MeasurementModel(self.name, self.indices,
                                [v * factor for v in self.noise],
                                self.angle_indices)


# The simulator generates its observation noise with these standard deviations
# (SpheroEnv obs_noise_std_pos / obs_noise_std_vel, and note the environment
# applies the *vel* std to heading as well). Starting R at the generator's own
# values is the principled starting point: R is measurable, so measure it and
# do not guess. Q is the knob that gets tuned.
SIM_SIGMA_POS = 0.05        # m
SIM_SIGMA_VEL = 0.025       # m/s, and rad on heading

# --------------------------------------------------------------------------- #
# The REAL robot's sensors are nothing like the simulator's
# --------------------------------------------------------------------------- #
#
# The simulator applies `obs_noise_std_vel` to heading as well as to speed, so
# in sim the heading measurement really is good to 0.025 rad. On hardware that
# number is a fiction, and believing it is catastrophic.
#
# Measured on this robot, the heading reference wanders by up to +-46 degrees
# over a run (Lab 1, trap 7). Telling the filter it is accurate to 2 degrees
# makes it chase heading noise, and because Q couples heading to position - a
# heading error moves you sideways - the position estimate gets dragged with
# it through the off-diagonal terms of the Kalman gain.
#
# Seen on a real 200-step run: the ball travelled 19 cm in total while the
# filter's estimate swung over a metre in y, reporting sigma of 2 cm the whole
# time. Replaying that log with sigma_heading = 0.3 instead of 0.025 cut the
# mean estimate-to-odometry error from 0.255 m to 0.080 m. The exact value
# barely matters above ~0.1 rad; claiming 2 degrees is what breaks it.
#: The Sphero locator, once the synthetic observation noise is switched off
#: (see make_real_env). It reports in centimetres, so ~0.01 m is the honest
#: resolution. It was 0.05 while `Robot` was still adding 5 cm of fabricated
#: Gaussian noise on top of it; leaving it there afterwards told the filter a
#: near-exact measurement was 5 cm noisy, and the estimate lagged the locator
#: by 6.8 cm mean / 16 cm max on a real run - which the controller then steered
#: on, driving a visibly wrong path.
REAL_SIGMA_POS = 0.01       # m, Sphero locator
REAL_SIGMA_HEADING = 0.30   # rad (17 deg) - honest about the drift
REAL_SIGMA_SPEED = 0.05     # m/s, wheel encoders

#: Floor on the REPORTED position sigma on hardware, metres.
#:
#: The filter's P measures uncertainty against the LOCATOR, and with the
#: locator trusted heavily that collapses to under a centimetre. But the demo
#: is scored against a physical marker and a tape measure, and the locator
#: itself drifts from the world - integrated wheel odometry with no external
#: reference, exactly the situation the W04 lecture's closing question is
#: about. That drift is invisible to the filter, so P cannot contain it.
#:
#: Lab 1 measured a final error of 0.038 m against a physical target with a
#: well-tuned controller, so locator-versus-world sits in the 3-4 cm range.
#: Reporting sigma below that is not confidence, it is a filter claiming to
#: know something it has no sensor for - and the demo scores exactly that
#: claim.
#:
#: Applied at REPORTING time only, never fed back into the gain, so the
#: estimate stays as tight as the measurements allow.
REAL_P_FLOOR_SIGMA = 0.035


def position_measurement(sigma_xy=SIM_SIGMA_POS):
    """Locator position - the only sensor that observes x and y directly."""
    return MeasurementModel("position", (0, 1), (sigma_xy ** 2, sigma_xy ** 2))


def heading_measurement(sigma_rad=SIM_SIGMA_VEL):
    """Attitude yaw. Absolute but drifting; the innovation must be wrapped."""
    return MeasurementModel("heading", (2,), (sigma_rad ** 2,), angle_indices=(0,))


def speed_measurement(sigma_ms=SIM_SIGMA_VEL):
    """Encoder speed projected onto the heading."""
    return MeasurementModel("speed", (3,), (sigma_ms ** 2,))


def full_state_measurement(sigma_xy=SIM_SIGMA_POS, sigma_rad=SIM_SIGMA_VEL,
                           sigma_ms=SIM_SIGMA_VEL):
    """Everything the environment's obs vector carries."""
    return MeasurementModel(
        "full", (0, 1, 2, 3),
        (sigma_xy ** 2, sigma_xy ** 2, sigma_rad ** 2, sigma_ms ** 2),
        angle_indices=(2,))


def collision_measurement(sigma_ms=0.005):
    """Zero-speed pseudo-measurement, applied on the step a collision is seen.

    obs[4] is the collision flag. A ball against a wall is not moving, and that
    is information the dynamics model has no way to know - it will happily
    integrate the commanded speed straight through the wall. Folding in "speed
    is zero" with a tight R pulls the estimate back, and it costs one extra
    update. The brief names collision as an available measurement; almost
    nobody uses it.
    """
    return MeasurementModel("collision", (3,), (sigma_ms ** 2,))


def odometry_measurement(sigma_rad=SIM_SIGMA_VEL, sigma_ms=SIM_SIGMA_VEL):
    """Heading and speed - the sensors the brief actually allows.

    The Lab 2 brief lists the available measurements as "robot speed, robot
    heading, robot collision". **Position is not on that list**, and the
    W04 lecture's own reference implementation goes further still:

        "Our sphero has a number of internal sensors. I am just going to use
         the internal odometry as speed measurement."     - W04, slide 48

    That matters, because it is the honest situation. The Sphero's "locator"
    x, y is not an external reference - it is the ball's own wheel odometry,
    integrated. Feeding it back as a position measurement is close to circular:
    the filter is being corrected by its own dead reckoning.

    The consequence is the point of the whole lecture, and its closing review
    question - "without an external reference, can a robot localise itself in
    an unknown environment?". No. With heading and speed only, x and y are
    pure dead reckoning and their variance grows without bound. The filter
    cannot stop that; it can only be HONEST about it, which is exactly what
    the marker scores.
    """
    return MeasurementModel("odometry", (2, 3),
                            (sigma_rad ** 2, sigma_ms ** 2), angle_indices=(0,))


def position_speed_measurement(sigma_xy=REAL_SIGMA_POS, sigma_ms=REAL_SIGMA_SPEED):
    """Position and speed. Heading deliberately NOT measured.

    The hardware default, and the reason is not tuning - it is that on this
    robot the heading channel is not a measurement at all.

    Measured over a real 200-step run: the reported heading settled to exactly
    `heading_cmd - 171 degrees` with a standard deviation of 0.1 degrees. No
    physical sensor tracks a command that precisely. It is the command being
    echoed back with an offset, which the dashboard's own help text warns
    about: `api.get_heading()` returns what you asked for, not what happened.

    Feeding that to the filter is worse than feeding it nothing. A missing
    measurement leaves the covariance honestly large; a fabricated one that
    correlates perfectly with the input looks informative, so the filter trusts
    it - and because Q couples heading to position, it drags x and y with it.

    The same run shows the ball travels within 1.4 degrees (median) of the
    COMMANDED heading, so the command frame is sound and the process model
    already predicts heading correctly without help.
    """
    return MeasurementModel("pos_speed", (0, 1, 3),
                            (sigma_xy ** 2, sigma_xy ** 2, sigma_ms ** 2))


MEASUREMENT_BUILDERS = {
    "position": position_measurement,
    "heading": heading_measurement,
    "speed": speed_measurement,
    "odometry": odometry_measurement,
    "pos_speed": position_speed_measurement,
    "full": full_state_measurement,
}


# --------------------------------------------------------------------------- #
# Process noise, derived rather than guessed
# --------------------------------------------------------------------------- #

#: The simulator's process noise is applied to the ACTION, not to the state:
#: SpheroEnv integrates ground truth with the clean action and odometry with
#: `action + N(0, [process_noise_std_speed, process_noise_std_heading])`. The
#: observation is then odometry plus measurement noise. So the error the marker
#: scores - truth minus estimate - is driven by the divergence between those
#: two integrations, and it is a random walk that grows with the run.
SIM_PROCESS_STD_SPEED = 0.005      # on the speed command
SIM_PROCESS_STD_HEADING = 0.01     # rad, on the heading command


def input_jacobian(state, action, dt=None, eps=1e-5):
    """G = df/du by central differences, for the 2-element action."""
    state = np.asarray(state, dtype=float)[:N_STATES]
    action = np.asarray(action, dtype=float)[:2]
    G = np.zeros((N_STATES, 2))
    for j in range(2):
        step = eps * max(1.0, abs(action[j]))
        up, down = action.copy(), action.copy()
        up[j] += step
        down[j] -= step
        G[:, j] = wrap_state_diff(
            np.asarray(dynamics(state, up, dt), dtype=float)
            - np.asarray(dynamics(state, down, dt), dtype=float)) / (2.0 * step)
    return G


#: Floor added to Q on the position states every step, m^2.
#:
#: Not a numerical guard - it is the term that makes the filter pass. The
#: environment's observation is generated from ODOMETRY, and odometry is
#: integrated with a noisy action while ground truth is integrated with a clean
#: one. So no sensor on this robot observes truth: the truth-minus-odometry gap
#: is a random walk that nothing can correct, and it keeps growing.
#:
#: The input-noise term below handles the noise entering on THIS step, but it
#: is state dependent and collapses to zero when the ball stops - while the
#: bias already accumulated does not go anywhere. Park at the last waypoint for
#: seventy steps and P shrinks to a 4 mm sigma around a 14 mm error, which is
#: exactly how the chi-square pass rate falls to 0.645 on a filter whose own
#: innovations look perfectly consistent.
#:
#: A constant term makes P grow linearly with the number of steps, which is
#: what a random walk actually does. It is the filter admitting it cannot know
#: its absolute position better than the drift it has accumulated.
Q_POSITION_FLOOR = 1e-6


def Q_from_input_noise(state, action, dt=None,
                       std_speed=SIM_PROCESS_STD_SPEED,
                       std_heading=SIM_PROCESS_STD_HEADING,
                       floor=Q_POSITION_FLOOR):
    """Q = G * diag(sigma_u^2) * G' + floor, the input noise through the model.

    The first term is the principled construction, and it beats sweeping a
    scalar: the environment states exactly what noise it injects and where, so
    the honest Q is that covariance mapped into state space by the input
    Jacobian. It is state dependent in the right way - G is near zero inside
    the deadband, so a stationary robot stops accumulating *new* motion
    uncertainty, which a fixed diagonal Q cannot express.

    The floor covers what that term structurally cannot: drift already banked.
    """
    G = input_jacobian(state, action, dt)
    Q = G @ np.diag([std_speed ** 2, std_heading ** 2]) @ G.T
    return Q + np.diag([floor, floor, floor * 10.0, floor * 10.0])


# --------------------------------------------------------------------------- #
# The filter
# --------------------------------------------------------------------------- #


class EKF:
    """EKF over [x, y, heading, speed] driven by the Lab 1 lag model.

    Signature-compatible with the lab skeleton: `predict(action)` and
    `update(measurement)` still work with no extra arguments, and `state_est`
    and `P` are still the public attributes lab2.py reads.
    """

    def __init__(self, dt=0.1, q_scale=1.0, r_scale=1.0,
                 measurement="full", model=None, Q=None, P0=None,
                 derived_Q=True, hardware=False):
        self.dt = float(dt)
        set_dt(self.dt)
        self.model = dynamics if model is None else model

        #: Build Q per step from the input noise (see Q_from_input_noise), or
        #: fall back to the fixed diagonal below. Derived is better: it is
        #: state dependent, so a stopped ball stops accumulating uncertainty.
        self.derived_Q = bool(derived_Q) and model is None
        self.q_scale = float(q_scale)

        # Process noise. Q represents MODEL ERROR, not just physical process
        # noise - unmodelled heading drift, the deadband edge, BLE jitter. It
        # is the only place the filter can admit "my motion model is wrong",
        # and on this robot the model is the dominant error, so this is the
        # knob that matters. Scaled by dt because a longer step accumulates
        # proportionally more of it.
        base_Q = (np.diag([1e-4, 1e-4, 1e-3, 1e-3]) if Q is None
                  else np.asarray(Q, dtype=float))
        self.Q = base_Q * float(q_scale) * (self.dt / 0.1)

        # Initial covariance: honest about not knowing the start heading well.
        self.P0 = (np.diag([1e-3, 1e-3, 1e-2, 1e-2]) if P0 is None
                   else np.asarray(P0, dtype=float))
        self.P = self.P0.copy()

        self.state_est = np.zeros(N_STATES)
        self.n = N_STATES
        self.jacobian_eps = 1e-5
        self.r_scale = float(r_scale)

        # Which sensor noise to believe. The simulator's numbers and the real
        # robot's are genuinely different instruments - see REAL_SIGMA_*.
        self.hardware = bool(hardware)
        builder = MEASUREMENT_BUILDERS.get(measurement, full_state_measurement)
        if self.hardware:
            kw = {
                "full": dict(sigma_xy=REAL_SIGMA_POS, sigma_rad=REAL_SIGMA_HEADING,
                             sigma_ms=REAL_SIGMA_SPEED),
                "odometry": dict(sigma_rad=REAL_SIGMA_HEADING,
                                 sigma_ms=REAL_SIGMA_SPEED),
                "pos_speed": dict(sigma_xy=REAL_SIGMA_POS,
                                  sigma_ms=REAL_SIGMA_SPEED),
                "position": dict(sigma_xy=REAL_SIGMA_POS),
                "heading": dict(sigma_rad=REAL_SIGMA_HEADING),
                "speed": dict(sigma_ms=REAL_SIGMA_SPEED),
            }.get(measurement, {})
            self.measurement = builder(**kw).scaled(self.r_scale)
        else:
            self.measurement = builder().scaled(self.r_scale)
        self.collision_model = collision_measurement().scaled(self.r_scale)

        self.history = []

    # -- alias so `ekf.state` and `ekf.state_est` are the same thing -------- #

    @property
    def state(self):
        return self.state_est

    @state.setter
    def state(self, value):
        self.state_est = np.asarray(value, dtype=float).ravel()[:self.n]

    # -- setup -------------------------------------------------------------- #

    def reset(self, state=None, P=None):
        self.state_est = (np.zeros(self.n) if state is None else
                          np.asarray(state, dtype=float).ravel()[:self.n].copy())
        self.P = self.P0.copy() if P is None else np.asarray(P, dtype=float).copy()
        self.history = []
        return self.state_est, self.P

    # -- prediction --------------------------------------------------------- #

    def _f(self, state, action):
        return np.asarray(self.model(state, action), dtype=float).ravel()[:self.n]

    def jacobian(self, state, action):
        """F = df/dx by central differences."""
        F = np.zeros((self.n, self.n))
        for j in range(self.n):
            step = self.jacobian_eps * max(1.0, abs(state[j]))
            up, down = state.copy(), state.copy()
            up[j] += step
            down[j] -= step
            F[:, j] = wrap_state_diff(self._f(up, action) - self._f(down, action)) \
                / (2.0 * step)
        return F

    def predict(self, action):
        """Propagate the mean through f and the covariance through F P F' + Q."""
        F = self.jacobian(self.state_est, action)
        Q = (Q_from_input_noise(self.state_est, action, self.dt) * self.q_scale
             if self.derived_Q else self.Q)
        self.state_est = self._f(self.state_est, action)
        self.state_est[ANGLE_INDEX] = wrap_angle(self.state_est[ANGLE_INDEX])
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)          # keep it symmetric
        return self.state_est, self.P

    # -- update ------------------------------------------------------------- #

    def update(self, measurement, model=None):
        """Fold in a measurement and record its NIS.

        `measurement` may be the environment's 5-element observation
        [x, y, heading, speed, collision] or a bare 4-element state. The
        collision flag is sliced off before use - passing the raw 5-vector into
        a 4-state update is the single easiest way to break this lab.
        """
        z_full = np.asarray(measurement, dtype=float).ravel()
        collided = bool(z_full[4]) if z_full.size > 4 else False
        m = self.measurement if model is None else model
        z = z_full[list(m.indices)] if z_full.size >= self.n else z_full

        self._apply(z, m)

        # A ball in contact with a wall is stationary, whatever the model says.
        if collided:
            self._apply(np.array([0.0]), self.collision_model)

        return self.state_est, self.P

    def _apply(self, z, m):
        H = m.H(self.n)
        R = m.R()
        y = m.innovation(z, self.state_est)
        S = H @ self.P @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = self.P @ H.T @ S_inv
        self.state_est = self.state_est + K @ y
        self.state_est[ANGLE_INDEX] = wrap_angle(self.state_est[ANGLE_INDEX])

        # Joseph form. Stays symmetric positive-definite under rounding, which
        # the simple (I - KH)P form does not over a 200 step run.
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.history.append({"measurement": m.name, "dof": m.dof,
                             "nis": float(y @ S_inv @ y), "innovation": y.copy()})

    # -- reporting ---------------------------------------------------------- #

    def position_covariance(self):
        """Reported position covariance, with the hardware honesty floor.

        See REAL_P_FLOOR_SIGMA. The floor is applied here rather than to
        `self.P`, so it never enters the Kalman gain - the filter keeps
        tracking as tightly as the locator allows, and only the number it
        REPORTS admits what it cannot observe.
        """
        P = self.P[:2, :2].copy()
        if self.hardware:
            floor = REAL_P_FLOOR_SIGMA ** 2
            P[0, 0] = max(P[0, 0], floor)
            P[1, 1] = max(P[1, 1], floor)
        return P

    def sigma_xy(self):
        """Per-axis standard deviation, metres - what the demo is scored on."""
        P = self.position_covariance()
        return math.sqrt(max(P[0, 0], 0.0)), math.sqrt(max(P[1, 1], 0.0))

    def nis_summary(self):
        """Is the filter consistent? This is what Lab 2 is scored on."""
        if not self.history:
            return {}
        by_dof = {}
        for row in self.history:
            by_dof.setdefault(row["dof"], []).append(row["nis"])
        out = {}
        for dof, values in by_dof.items():
            arr = np.asarray(values)
            hi, lo = CHI2_95.get(dof, 5.991), CHI2_05.get(dof, 0.103)
            out["dof%d" % dof] = {
                "n": int(arr.size),
                "mean_nis": float(arr.mean()),
                "expected_mean": float(dof),
                "inside_95_gate": float(np.mean(arr <= hi)),
                "below_5_gate": float(np.mean(arr < lo)),
                "verdict": ("optimistic - raise Q or R" if arr.mean() > dof * 1.5
                            else "pessimistic - lower Q or R" if arr.mean() < dof * 0.5
                            else "consistent"),
            }
        return out


# --------------------------------------------------------------------------- #
# Consistency metrics - exactly what the Lab 2 automarker computes
# --------------------------------------------------------------------------- #


def mahalanobis_series(truth_xy, est_xy, cov_series):
    """Per-step Mahalanobis distance of the position error.

        d_k = sqrt( (t_k - e_k)^T P_k^-1 (t_k - e_k) )

    A *scaled* error: large only if the estimate is wrong by more than the
    filter claimed it might be.
    """
    truth = np.asarray(truth_xy, dtype=float).reshape(-1, 2)
    est = np.asarray(est_xy, dtype=float).reshape(-1, 2)
    covs = np.asarray(cov_series, dtype=float).reshape(-1, 2, 2)
    n = min(len(truth), len(est), len(covs))
    out = np.zeros(n)
    for k in range(n):
        d = truth[k] - est[k]
        try:
            P_inv = np.linalg.inv(covs[k])
        except np.linalg.LinAlgError:
            P_inv = np.linalg.pinv(covs[k])
        out[k] = math.sqrt(max(float(d @ P_inv @ d), 0.0))
    return out


def consistency_report(truth_xy, est_xy, cov_series,
                       mahalanobis_threshold=4.0, chi2_pass_threshold=0.90):
    """Reproduce the Lab 2 verdict locally, before submitting.

    The chi-square gate is TWO-SIDED, and getting that wrong cost half the
    marks on the first M2 submission.

    A 2-DoF NIS must land inside [CHI2_05, CHI2_95] = [0.103, 5.991]. Testing
    only the upper bound asks "is the filter over-confident?" and says nothing
    about the opposite failure, which is the one this project actually had: a
    filter reporting sigma = 3.5 cm while tracking to 0.7 mm scores d^2 = 0.008,
    sails under the upper bound, and is wrong by a factor of 50.

    The marker returned 0.020 for a run this function scored 1.000. Recomputing
    the submitted CSV with the two-sided gate reproduces 0.020 exactly, and 193
    of its 200 rows fall below the LOWER bound. The unit's own brief says so in
    as many words: "NIS is BELOW the bounds (Fail Low): The filter is
    Pessimistic (sluggish)."

    `chi2_pass_rate` is now the two-sided rate - the number the marker uses.
    `chi2_upper_only` is kept alongside it so the two can be compared, because
    a large gap between them is the signature of a pessimistic filter.
    """
    d = mahalanobis_series(truth_xy, est_xy, cov_series)
    if d.size == 0:
        return {"error": "no samples"}
    d2 = d ** 2
    inside = float(np.mean((d2 >= CHI2_05[2]) & (d2 <= CHI2_95[2])))
    upper_only = float(np.mean(d2 <= CHI2_95[2]))
    below = float(np.mean(d2 < CHI2_05[2]))
    above = float(np.mean(d2 > CHI2_95[2]))
    mean_d = float(d.mean())
    checks = {
        "mean_mahalanobis": {"value": mean_d, "threshold": mahalanobis_threshold,
                             "pass": bool(mean_d <= mahalanobis_threshold)},
        "chi2_pass_rate": {"value": inside, "threshold": chi2_pass_threshold,
                           "pass": bool(inside >= chi2_pass_threshold)},
    }
    err = np.linalg.norm(np.asarray(truth_xy, dtype=float)[:len(d)]
                         - np.asarray(est_xy, dtype=float)[:len(d)], axis=1)
    return {
        "checks": checks,
        "pass": all(c["pass"] for c in checks.values()),
        "n": int(d.size),
        "mahalanobis": d,
        "chi2_upper_only": upper_only,
        "chi2_below_lower": below,
        "chi2_above_upper": above,
        "mean_nis": float(d2.mean()),
        "rms_position_error_m": float(np.sqrt(np.mean(err ** 2))),
        "mean_position_error_m": float(err.mean()),
        "advice": _tuning_advice(mean_d, inside, below, above),
    }


def lecture_jacobian(state, action, dt=None, params=None):
    """The closed form from the W04 lecture, slide 46, in THIS convention.

    The lecture writes, for x += v cos(th) dt and y += v sin(th) dt,

        A = [[1, 0, -v sin(th) dt,  cos(th) dt],
             [0, 1,  v cos(th) dt,  sin(th) dt],
             [0, 0,  1 - alpha_th,  0         ],
             [0, 0,  0,             1 - alpha_v]]

    This environment uses the other convention - heading 0 points along +y -
    so the trig terms swap and the minus sign moves to the y row.

    This does NOT describe the identified model, for three separate reasons,
    and `self_test()` measures each:

    1. **Turn-rate limit.** Once `alpha_h * heading_error` exceeds
       `max_turn_rate * dt` the step is clipped to a constant, so
       d(heading_new)/d(heading) is 1, not `1 - alpha_h`.
    2. **New versus old state.** The lecture propagates position using the
       OLD speed and heading. This model - like `SpheroEnv._base_dynamics`,
       which is what the marker runs - uses the newly updated ones, so every
       position derivative picks up an extra chain-rule factor.
    3. **alpha_h saturates.** At dt = 0.1 and tau_heading = 0.05,
       `dt / tau = 2`, clipped to 1. So `1 - alpha_h = 0`: heading snaps to
       the command in one step and position has *no* sensitivity to the
       previous heading at all.

    Kept for comparison, never used by the filter.
    """
    dt = DT if dt is None else float(dt)
    p = LAG_PARAMS if params is None else params
    _x, _y, heading, speed = (float(v) for v in np.asarray(state, dtype=float)[:4])

    alpha_h = float(np.clip(dt / max(p["tau_heading"], 1e-6), 0.0, 1.0))
    alpha_v = float(np.clip(dt / max(p["tau_speed"], 1e-6), 0.0, 1.0))

    F = np.eye(4)
    F[0, 2] = speed * math.cos(heading) * dt
    F[0, 3] = math.sin(heading) * dt
    F[1, 2] = -speed * math.sin(heading) * dt
    F[1, 3] = math.cos(heading) * dt
    F[2, 2] = 1.0 - alpha_h
    F[3, 3] = 1.0 - alpha_v
    return F


def analytic_jacobian(state, action, dt=None, params=None):
    """Exact closed-form F for the model `dynamics()` actually implements.

    Same derivation as the lecture, but carrying the pieces the slide leaves
    out: the turn-rate clip, the velocity clip, and the fact that position is
    integrated with the UPDATED heading and speed rather than the previous
    ones - which puts a `d(heading_new)/d(heading)` and a
    `d(speed_new)/d(speed)` factor on every position derivative.

    Used to VERIFY the numerical Jacobian, not to replace it. Numerical stays
    the filter's default because the deadband and the alignment gate are hard
    switches whose derivative does not exist at the switch, and central
    differences degrade gracefully there while a closed form has to pick a
    side.
    """
    dt = DT if dt is None else float(dt)
    p = LAG_PARAMS if params is None else params
    _x, _y, heading, speed = (float(v) for v in np.asarray(state, dtype=float)[:4])
    speed_cmd, heading_cmd = float(action[0]), float(action[1])

    desired_heading = wrap_angle(heading_cmd + p["heading_bias"])
    heading_error = wrap_angle(desired_heading - heading)
    alpha_h = float(np.clip(dt / max(p["tau_heading"], 1e-6), 0.0, 1.0))
    limit = p["max_turn_rate"] * dt
    raw_step = alpha_h * heading_error

    if abs(raw_step) > limit:
        # step is clipped to a constant, so heading_new tracks heading 1:1
        dh_dh = 1.0
        heading_new = wrap_angle(heading + math.copysign(limit, raw_step))
    else:
        dh_dh = 1.0 - alpha_h
        heading_new = wrap_angle(heading + raw_step)

    magnitude = abs(speed_cmd)
    if magnitude <= p["deadband"]:
        target = 0.0
    else:
        target = p["speed_gain"] * math.copysign(magnitude - p["deadband"], speed_cmd)
    residual = abs(wrap_angle(desired_heading - heading_new))
    if residual > p["heading_tol"]:
        target *= p["turn_speed_factor"]
    target = float(np.clip(target, -p["vel_limit"], p["vel_limit"]))

    alpha_v = float(np.clip(dt / max(p["tau_speed"], 1e-6), 0.0, 1.0))
    speed_new = speed + alpha_v * (target - speed)
    if abs(speed_new) >= p["vel_limit"]:
        dv_dv = 0.0          # clipped: no sensitivity to the previous speed
        speed_new = float(np.clip(speed_new, -p["vel_limit"], p["vel_limit"]))
    else:
        dv_dv = 1.0 - alpha_v

    F = np.eye(4)
    F[0, 2] = speed_new * math.cos(heading_new) * dt * dh_dh
    F[0, 3] = math.sin(heading_new) * dt * dv_dv
    F[1, 2] = -speed_new * math.sin(heading_new) * dt * dh_dh
    F[1, 3] = math.cos(heading_new) * dt * dv_dv
    F[2, 2] = dh_dh
    F[3, 3] = dv_dv
    return F


def self_test(dt=0.1, n=400, verbose=True):
    """Verify the numerical Jacobian, and measure the gap to the lecture form.

    Two comparisons:

    * against `analytic_jacobian` - the exact derivative of the model that
      runs. These must agree to central-difference precision. Disagreement
      means the filter's F is wrong, which silently corrupts every covariance.
    * against `lecture_jacobian` - slide 46. Expected to differ; the size of
      the difference is the cost of the three simplifications listed there.
    """
    rng = np.random.default_rng(0)
    set_dt(dt)
    ekf = EKF(dt=dt)
    exact, slide = [], []
    for _ in range(int(n)):
        state = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1),
                          rng.uniform(-np.pi, np.pi), rng.uniform(0.0, 0.16)])
        # keep the command clear of the deadband, where no derivative exists
        action = np.array([rng.uniform(0.02, 0.15), rng.uniform(-np.pi, np.pi)])
        Fn = ekf.jacobian(state, action)
        exact.append(np.abs(Fn - analytic_jacobian(state, action, dt)).max())
        slide.append(np.abs(Fn - lecture_jacobian(state, action, dt)).max())

    out = {"n": int(n), "dt": float(dt),
           "vs_analytic_median": float(np.median(exact)),
           "vs_analytic_worst": float(np.max(exact)),
           "vs_lecture_median": float(np.median(slide)),
           "vs_lecture_worst": float(np.max(slide))}
    if verbose:
        print(f"Jacobian self-test, {n} random states at dt = {dt}")
        print(f"  numerical vs exact closed form : median "
              f"{out['vs_analytic_median']:.2e}  worst {out['vs_analytic_worst']:.2e}")
        print(f"  numerical vs lecture slide 46  : median "
              f"{out['vs_lecture_median']:.2e}  worst {out['vs_lecture_worst']:.2e}")
        ok = out["vs_analytic_worst"] < 1e-4
        print(f"  -> numerical Jacobian {'VERIFIED' if ok else 'MISMATCH'}")
    return out


def _tuning_advice(mean_d, inside, below=0.0, above=0.0):
    """Which way the filter is wrong, from which side of the gate it falls."""
    if inside >= 0.90 and mean_d <= 4.0:
        return ("Consistent. NIS sits inside the chi-square band.")
    if below > above:
        return (f"PESSIMISTIC: {below:.0%} of steps fall BELOW the lower "
                f"chi-square bound ({CHI2_05[2]}). The covariance is much "
                "larger than the error justifies, so the filter is claiming "
                "uncertainty it does not have. Lower Q and R - and check "
                "whether a floor is being applied to the reported P, which "
                "caps how tight the covariance can get however you tune.")
    return ("OVER-CONFIDENT: the true error is larger than the covariance "
            "claims. Raise Q first - the model is the usual culprit on this "
            "robot - then R.")
