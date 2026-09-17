"""Dynamics model zoo for the Sphero Bolt+.

Every model maps

    state  = [x, y, heading, speed]        (world frame, metres / radians / m/s)
    action = [speed_cmd, heading_cmd]      (the lab's action interface)

to the next state, matching the signature `SpheroEnv` expects for its
`dynamics=` argument.

Heading convention (inherited from `SpheroEnv._base_dynamics`):

    heading = 0     -> +y
    heading = +pi/2 -> +x
    dx = v * sin(heading) * dt
    dy = v * cos(heading) * dt

The models form a ladder of increasing fidelity, so you can show what each
piece of physics buys you:

    M0 baseline   the crude model shipped with the repo (reference only)
    M1 unicycle   instant heading, static speed gain              (kinematic)
    M2 lag        + first-order speed/heading lag, deadband       (kinematic+)
    M3 greybox    + course-angle lag, drag, accel limit, slip     (dynamic)

Only the parameters change between M1..M3; the structure is nested, so a
fitted M3 can always reproduce M2 by driving the extra parameters to their
degenerate values.

Aux state
---------
M3 carries internal state that is not part of the 4-vector (the course angle
psi, i.e. the direction the ball is actually travelling, which lags the
commanded heading because of the ball's inertia).  `SpheroEnv.step()` calls
the dynamics twice per control step - once for the ground-truth branch and
once for the odometry branch - so aux state is stored per branch and the
branch is selected by a call-parity counter that `reset()` re-zeroes.  Call
order inside `SpheroEnv.step()` is fixed (true first, then odom), so this is
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi


def wrap_angle(angle):
    """Wrap to [-pi, pi)."""
    return (angle + np.pi) % TWO_PI - np.pi


# --------------------------------------------------------------------------- #
# Parameter specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter: default, bounds, and UI metadata."""

    name: str
    default: float
    low: float
    high: float
    unit: str = ""
    description: str = ""
    fit: bool = True          # include in system identification by default
    step: Optional[float] = None   # UI slider granularity

    def ui_step(self) -> float:
        if self.step is not None:
            return self.step
        span = self.high - self.low
        return float(f"{span / 200.0:.1e}")


@dataclass
class ModelConfig:
    """Concrete parameter values for one model, plus the timestep."""

    name: str
    dt: float = 0.1
    params: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "dt": self.dt, "params": dict(self.params)}


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #


class DynamicsModel:
    """Callable dynamics model with optional per-branch auxiliary state."""

    #: human-readable label shown in the dashboard
    label: str = ""
    #: longer explanation shown in the dashboard
    blurb: str = ""
    #: ordered parameter specs
    specs: Tuple[ParamSpec, ...] = ()
    #: number of extra (non [x,y,heading,speed]) internal states
    n_aux: int = 0

    def __init__(self, dt: float = 0.1, params: Optional[Dict[str, float]] = None):
        self.dt = float(dt)
        self.params: Dict[str, float] = {s.name: s.default for s in self.specs}
        if params:
            self.set_params(params)
        self._branches: List[Optional[np.ndarray]] = [None, None]
        self._call_parity = 0

    # -- parameters -------------------------------------------------------- #

    @classmethod
    def spec_map(cls) -> Dict[str, ParamSpec]:
        return {s.name: s for s in cls.specs}

    @classmethod
    def default_params(cls) -> Dict[str, float]:
        return {s.name: s.default for s in cls.specs}

    @classmethod
    def fit_names(cls) -> List[str]:
        return [s.name for s in cls.specs if s.fit]

    @classmethod
    def bounds(cls, names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        m = cls.spec_map()
        low = np.array([m[n].low for n in names], dtype=float)
        high = np.array([m[n].high for n in names], dtype=float)
        return low, high

    def set_params(self, params: Dict[str, float]) -> None:
        known = self.spec_map()
        for key, value in params.items():
            if key in known:
                self.params[key] = float(value)

    def get_vector(self, names: Sequence[str]) -> np.ndarray:
        return np.array([self.params[n] for n in names], dtype=float)

    def set_vector(self, names: Sequence[str], values: Sequence[float]) -> None:
        for name, value in zip(names, values):
            self.params[name] = float(value)

    def config(self) -> ModelConfig:
        return ModelConfig(name=self.name(), dt=self.dt, params=dict(self.params))

    @classmethod
    def name(cls) -> str:
        for key, value in MODEL_REGISTRY.items():
            if value is cls:
                return key
        return cls.__name__

    # -- aux state --------------------------------------------------------- #

    def reset(self, state: Optional[Sequence[float]] = None) -> None:
        """Clear auxiliary state. Call once per episode, before the first step."""
        aux = self.init_aux(state) if self.n_aux else None
        self._branches = [None if aux is None else aux.copy() for _ in range(2)]
        self._call_parity = 0

    def init_aux(self, state: Optional[Sequence[float]]) -> np.ndarray:
        """Initial auxiliary state given the initial 4-vector."""
        return np.zeros(self.n_aux, dtype=float)

    def _take_branch(self, state) -> Tuple[int, Optional[np.ndarray]]:
        if not self.n_aux:
            return 0, None
        idx = self._call_parity % 2
        self._call_parity += 1
        if self._branches[idx] is None:
            self._branches[idx] = self.init_aux(state)
        return idx, self._branches[idx]

    # -- stepping ---------------------------------------------------------- #

    def __call__(self, state, action):
        state = np.asarray(state, dtype=float).reshape(-1)
        action = np.asarray(action, dtype=float).reshape(-1)
        idx, aux = self._take_branch(state)
        next_state, next_aux = self.step(state, action, aux)
        if self.n_aux:
            self._branches[idx] = next_aux
        return np.asarray(next_state, dtype=np.float32)

    def step(self, state, action, aux):
        """Return (next_state, next_aux). Subclasses implement this."""
        raise NotImplementedError

    # -- rollout helper (used by system ID; bypasses branch bookkeeping) ---- #

    def rollout(self, state0, actions, n_steps: Optional[int] = None) -> np.ndarray:
        """Free-run the model from `state0` through `actions`.

        Returns an array of shape (len(actions) + 1, 4) including `state0`.
        """
        state = np.asarray(state0, dtype=float).reshape(-1)[:4].copy()
        aux = self.init_aux(state) if self.n_aux else None
        actions = np.asarray(actions, dtype=float).reshape(-1, 2)
        if n_steps is not None:
            actions = actions[:n_steps]
        out = np.empty((len(actions) + 1, 4), dtype=float)
        out[0] = state
        for k, action in enumerate(actions):
            state, aux = self.step(state, action, aux)
            state = np.asarray(state, dtype=float).reshape(-1)[:4]
            out[k + 1] = state
        return out

    def predict_one_step(self, states, actions) -> np.ndarray:
        """Teacher-forced one-step prediction from each measured state.

        `states` is (N, 4), `actions` is (N, 2); returns (N, 4) predictions of
        the state at k+1.  Auxiliary state is re-initialised at every k, so
        this is only exact for memoryless models - use `rollout` (free-run)
        as the primary objective for M3.
        """
        states = np.asarray(states, dtype=float).reshape(-1, 4)
        actions = np.asarray(actions, dtype=float).reshape(-1, 2)
        out = np.empty_like(states)
        for k in range(len(states)):
            aux = self.init_aux(states[k]) if self.n_aux else None
            nxt, _ = self.step(states[k], actions[k], aux)
            out[k] = np.asarray(nxt, dtype=float).reshape(-1)[:4]
        return out


# --------------------------------------------------------------------------- #
# M0 - the model shipped with the repo
# --------------------------------------------------------------------------- #


class BaselineModel(DynamicsModel):
    label = "M0 - repo baseline"
    blurb = (
        "The crude model shipped with the skeleton code. Heading snaps to the "
        "command, speed integrates the command instead of tracking it, and dt "
        "is hard-coded to 0.1 s. Kept only as the reference to beat."
    )
    specs = (
        ParamSpec("accel_gain", 0.1, 0.0, 1.0, "m/s per cmd",
                  "Speed increment per step per unit speed_cmd", fit=False),
        ParamSpec("speed_cap", 1.0, 0.0, 3.0, "m/s",
                  "Hard clip on the integrated speed", fit=False),
    )

    def step(self, state, action, aux):
        x, y, heading, speed = state[:4]
        speed_cmd, heading_cmd = action[0], action[1]
        dt = self.dt
        heading_new = wrap_angle(heading_cmd)
        speed_new = np.clip(speed + speed_cmd * self.params["accel_gain"],
                            0.0, self.params["speed_cap"])
        x_new = x + speed_new * np.sin(heading_new) * dt
        y_new = y + speed_new * np.cos(heading_new) * dt
        return np.array([x_new, y_new, heading_new, speed_new], dtype=float), aux


# --------------------------------------------------------------------------- #
# M1 - static-gain unicycle
# --------------------------------------------------------------------------- #


class UnicycleModel(DynamicsModel):
    label = "M1 - calibrated unicycle"
    blurb = (
        "Pure kinematics (Lecture 2 differential-drive model with heading as "
        "the input instead of a turn rate). Heading is achieved instantly and "
        "speed is a static gain on the command. One free parameter for the "
        "actuator scale - this alone removes the bulk of the sim-to-real error "
        "because the raw speed command was mis-scaled by ~2.5x."
    )
    specs = (
        ParamSpec("speed_gain", 1.0, 0.05, 6.0, "(m/s)/cmd",
                  "Steady-state speed per unit speed_cmd"),
        ParamSpec("vel_limit", 0.60, 0.05, 3.0, "m/s",
                  "Physical top speed of the ball"),
        ParamSpec("heading_bias", 0.0, -0.6, 0.6, "rad",
                  "Constant offset between commanded and achieved heading"),
    )

    def step(self, state, action, aux):
        x, y, _, _ = state[:4]
        speed_cmd, heading_cmd = action[0], action[1]
        dt = self.dt
        p = self.params

        heading_new = wrap_angle(heading_cmd + p["heading_bias"])
        speed_new = float(np.clip(p["speed_gain"] * speed_cmd,
                                  -p["vel_limit"], p["vel_limit"]))

        x_new = x + speed_new * np.sin(heading_new) * dt
        y_new = y + speed_new * np.cos(heading_new) * dt
        return np.array([x_new, y_new, heading_new, speed_new], dtype=float), aux


# --------------------------------------------------------------------------- #
# M2 - first-order lag + deadband
# --------------------------------------------------------------------------- #


class LagModel(DynamicsModel):
    label = "M2 - first-order lag + deadband"
    blurb = (
        "Adds the three effects that dominate a rolling ball at 10 Hz: the "
        "motor takes time to spin the shell up (first-order speed lag), the "
        "heading servo has a finite turn rate, and small commands produce no "
        "motion at all (deadband, because the raw 0-255 command is quantised "
        "and has to overcome rolling friction)."
    )
    specs = (
        ParamSpec("speed_gain", 1.0, 0.05, 6.0, "(m/s)/cmd",
                  "Steady-state speed per unit speed_cmd"),
        ParamSpec("vel_limit", 0.60, 0.05, 3.0, "m/s", "Physical top speed"),
        ParamSpec("tau_speed", 0.30, 0.02, 3.0, "s",
                  "Time constant of the speed response"),
        ParamSpec("deadband", 0.02, 0.0, 0.30, "cmd",
                  "Speed command below which the ball does not move"),
        ParamSpec("max_turn_rate", 12.0, 0.2, 60.0, "rad/s",
                  "Maximum achievable heading rate"),
        ParamSpec("tau_heading", 0.05, 0.01, 1.5, "s",
                  "Time constant of the heading response"),
        ParamSpec("heading_tol", 0.35, 0.0, 3.2, "rad",
                  "Heading error above which forward drive is cut"),
        ParamSpec("turn_speed_factor", 1.0, 0.0, 1.0, "-",
                  "Fraction of commanded speed retained while misaligned"),
        ParamSpec("heading_bias", 0.0, -0.6, 0.6, "rad",
                  "Constant heading offset"),
    )

    def step(self, state, action, aux):
        x, y, heading, speed = state[:4]
        speed_cmd, heading_cmd = action[0], action[1]
        dt = self.dt
        p = self.params

        desired_heading = wrap_angle(heading_cmd + p["heading_bias"])

        # Heading: first-order response, rate limited.
        heading_error = wrap_angle(desired_heading - heading)
        alpha_h = np.clip(dt / max(p["tau_heading"], 1e-6), 0.0, 1.0)
        step_h = np.clip(alpha_h * heading_error,
                         -p["max_turn_rate"] * dt, p["max_turn_rate"] * dt)
        heading_new = wrap_angle(heading + step_h)

        # Speed: deadband on the command, alignment gate, first-order lag.
        magnitude = abs(speed_cmd)
        if magnitude <= p["deadband"]:
            target = 0.0
        else:
            effective = np.sign(speed_cmd) * (magnitude - p["deadband"])
            target = p["speed_gain"] * effective

        residual = abs(wrap_angle(desired_heading - heading_new))
        if residual > p["heading_tol"]:
            target *= p["turn_speed_factor"]

        target = float(np.clip(target, -p["vel_limit"], p["vel_limit"]))
        alpha_v = np.clip(dt / max(p["tau_speed"], 1e-6), 0.0, 1.0)
        speed_new = speed + alpha_v * (target - speed)
        speed_new = float(np.clip(speed_new, -p["vel_limit"], p["vel_limit"]))

        x_new = x + speed_new * np.sin(heading_new) * dt
        y_new = y + speed_new * np.cos(heading_new) * dt
        return np.array([x_new, y_new, heading_new, speed_new], dtype=float), aux


# --------------------------------------------------------------------------- #
# M3 - grey-box ball dynamics
# --------------------------------------------------------------------------- #


class GreyBoxModel(DynamicsModel):
    """Grey-box model of the Bolt+ (differential drive inside a rolling shell).

    Extra physics over M2:

    * **Course angle psi.** The heading the controller commands is where the
      internal drive unit points; the ball's momentum means the *velocity
      vector* only turns towards it with time constant `tau_course`. This is
      what produces the wide curved arcs the real robot draws when the heading
      command changes quickly, and it is the single biggest structural error in
      the shipped model.
    * **Acceleration limit + quadratic drag.** Torque is limited by the shell
      slipping on the floor, so `dv/dt` saturates; rolling/aero losses give a
      `-c_drag * v|v|` term, which bends the steady-state speed curve over.
    * **Actuator quantisation.** The command is converted to an integer 0-255
      raw value on the wire, so the achievable speeds are a staircase.
    * **Slip factor.** Distance travelled per unit of commanded speed is not 1
      when the shell slips; `slip` scales the translation without touching the
      speed state (this is what corrupts the encoder odometry).
    """

    label = "M3 - grey-box ball dynamics"
    blurb = (
        "Adds the physics that is specific to a differential drive inside a "
        "rolling ball: the velocity vector (course angle) lags the commanded "
        "heading, acceleration is torque-limited, drag bends the steady-state "
        "speed curve, the wire command is quantised to 0-255, and the shell "
        "slips against the floor."
    )
    n_aux = 1  # [course_angle]
    specs = (
        ParamSpec("speed_gain", 1.0, 0.05, 6.0, "(m/s)/cmd",
                  "Steady-state speed per unit speed_cmd (before drag)"),
        ParamSpec("vel_limit", 0.60, 0.05, 3.0, "m/s", "Physical top speed"),
        ParamSpec("tau_speed", 0.30, 0.02, 3.0, "s", "Speed time constant"),
        ParamSpec("accel_limit", 4.0, 0.05, 30.0, "m/s^2",
                  "Torque/traction limit on |dv/dt|"),
        ParamSpec("drag", 0.25, 0.0, 6.0, "1/m",
                  "Quadratic drag coefficient (-drag * v|v|)"),
        ParamSpec("deadband", 0.02, 0.0, 0.30, "cmd",
                  "Speed command below which the ball does not move"),
        ParamSpec("max_turn_rate", 12.0, 0.2, 60.0, "rad/s",
                  "Maximum heading rate of the internal drive"),
        ParamSpec("tau_heading", 0.05, 0.01, 1.5, "s", "Heading time constant"),
        ParamSpec("tau_course", 0.25, 0.01, 3.0, "s",
                  "Lag of the velocity vector behind the heading (ball inertia)"),
        # `slip` and `speed_gain` both scale the distance travelled, so they are
        # confounded unless the loss also weights a trustworthy measured speed
        # channel. Left out of the fit by default; enable it explicitly (and
        # raise w_speed) once you trust the encoder speed.
        ParamSpec("slip", 1.0, 0.2, 2.0, "-",
                  "Translation scale: shell slip against the floor "
                  "(confounded with speed_gain - fit only with a good speed channel)",
                  fit=False),
        ParamSpec("heading_tol", 1.20, 0.0, 3.2, "rad",
                  "Heading error above which forward drive is cut"),
        ParamSpec("turn_speed_factor", 0.6, 0.0, 1.0, "-",
                  "Fraction of commanded speed retained while misaligned"),
        ParamSpec("heading_bias", 0.0, -0.6, 0.6, "rad", "Constant heading offset"),
        ParamSpec("cmd_quantisation", 255.0, 0.0, 255.0, "levels",
                  "Raw command levels (0 disables quantisation)", fit=False),
    )

    def init_aux(self, state):
        heading = float(state[2]) if state is not None and len(state) > 2 else 0.0
        return np.array([heading], dtype=float)

    def _quantise(self, speed_cmd: float) -> float:
        levels = self.params["cmd_quantisation"]
        if levels < 1.0:
            return speed_cmd
        scale = self.params["vel_limit"] / max(self.params["speed_gain"], 1e-6)
        if scale <= 0:
            return speed_cmd
        raw = np.round(np.clip(speed_cmd / scale, -1.0, 1.0) * levels)
        return float(raw / levels * scale)

    def step(self, state, action, aux):
        x, y, heading, speed = state[:4]
        speed_cmd, heading_cmd = action[0], action[1]
        dt = self.dt
        p = self.params

        if aux is None:
            aux = self.init_aux(state)
        course = float(aux[0])

        desired_heading = wrap_angle(heading_cmd + p["heading_bias"])

        # --- heading of the internal drive unit -------------------------- #
        heading_error = wrap_angle(desired_heading - heading)
        alpha_h = np.clip(dt / max(p["tau_heading"], 1e-6), 0.0, 1.0)
        step_h = np.clip(alpha_h * heading_error,
                         -p["max_turn_rate"] * dt, p["max_turn_rate"] * dt)
        heading_new = wrap_angle(heading + step_h)

        # --- speed: deadband, alignment gate, lag, drag, accel limit ------ #
        cmd = self._quantise(speed_cmd)
        magnitude = abs(cmd)
        if magnitude <= p["deadband"]:
            target = 0.0
        else:
            target = p["speed_gain"] * np.sign(cmd) * (magnitude - p["deadband"])

        residual = abs(wrap_angle(desired_heading - heading_new))
        if residual > p["heading_tol"]:
            target *= p["turn_speed_factor"]
        target = float(np.clip(target, -p["vel_limit"], p["vel_limit"]))

        accel = (target - speed) / max(p["tau_speed"], 1e-6) - p["drag"] * speed * abs(speed)
        accel = float(np.clip(accel, -p["accel_limit"], p["accel_limit"]))
        speed_new = float(np.clip(speed + accel * dt,
                                  -p["vel_limit"], p["vel_limit"]))

        # --- course angle: velocity vector lags the heading --------------- #
        alpha_c = np.clip(dt / max(p["tau_course"], 1e-6), 0.0, 1.0)
        course_new = wrap_angle(course + alpha_c * wrap_angle(heading_new - course))

        travel = p["slip"] * speed_new * dt
        x_new = x + travel * np.sin(course_new)
        y_new = y + travel * np.cos(course_new)

        next_state = np.array([x_new, y_new, heading_new, speed_new], dtype=float)
        return next_state, np.array([course_new], dtype=float)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


MODEL_REGISTRY: Dict[str, type] = {
    "baseline": BaselineModel,
    "unicycle": UnicycleModel,
    "lag": LagModel,
    "greybox": GreyBoxModel,
}

MODEL_ORDER: Tuple[str, ...] = ("baseline", "unicycle", "lag", "greybox")


def list_models() -> List[Dict[str, Any]]:
    """Describe every model for the dashboard."""
    out = []
    for key in MODEL_ORDER:
        cls = MODEL_REGISTRY[key]
        out.append(
            {
                "key": key,
                "label": cls.label,
                "blurb": cls.blurb,
                "n_aux": cls.n_aux,
                "params": [
                    {
                        "name": s.name,
                        "default": s.default,
                        "low": s.low,
                        "high": s.high,
                        "unit": s.unit,
                        "description": s.description,
                        "fit": s.fit,
                        "step": s.ui_step(),
                    }
                    for s in cls.specs
                ],
            }
        )
    return out


def build_model(name: str, dt: float = 0.1,
                params: Optional[Dict[str, float]] = None) -> DynamicsModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](dt=dt, params=params)
