import time
import threading
from typing import Optional, Tuple, Dict, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sphero_env.envs.visualiser import Visualiser
from sphero_unsw.sphero_edu import SpheroEduAPI


class Robot(gym.Env):
    """
    Real Sphero robot wrapper that mimics the SpheroEnv interface.
    
    Allows interchangeable use between simulation (SpheroEnv) and real hardware (Robot).
    
    The Robot class:
    - Wraps SpheroEduAPI for hardware control
    - Tracks odometry state [x_o, y_o, vx_o, vy_o]
    - Maintains a mirror of the observation interface
    - Provides reset() and step() methods compatible with SpheroEnv
    
    Note: Real robot doesn't have "true" ground truth, so we track odometry as the best estimate.
    """

    metadata = {"render_modes": [], "render_fps": 60}

    # ---- Actuator calibration (Lab 1) ---- #
    # The raw command sent over BLE is 0-255 and maps onto the ball's *physical*
    # speed range, which has nothing to do with `vel_limit` (a soft cap the lab
    # applies to speed_cmd). Scaling the raw command by vel_limit therefore
    # makes the real robot travel by the ratio physical_max_speed/vel_limit
    # faster than the simulator for the same command - measured at ~2.5x with
    # the shipped defaults. Calibrate `physical_max_speed` from a straight-line
    # staircase run (see labs/lab1/MODELLING.md) and set it here or via
    # labs/lab1/config/lab1_config.json.
    physical_max_speed: float = 3.0     # m/s at raw command 255
    raw_min_move: int = 0               # floor applied to non-zero raw commands
    #: Hard ceiling on the raw 0-255 command. This is the one safety limit that
    #: does NOT depend on calibration being correct: whatever `speed_cmd`,
    #: `speed_gain` or `physical_max_speed` say, the wire command can never
    #: exceed this, so the ball has a guaranteed top speed. Set it low while
    #: testing in a small space (40 is roughly a fast walk).
    max_raw_command: int = 255
    use_encoder_speed: bool = True      # measure speed from encoders, not the cmd echo
    do_reset_aim: bool = True           # zero the heading reference on reset()
    #: Degrees added to every heading command before it is sent.
    #:
    #: `reset_aim()` makes "heading 0" mean whichever way the internal drive
    #: unit happens to be facing, which after a shake is arbitrary and after a
    #: run is wherever it last drove. Aiming by hand does not survive either.
    #: So instead of aiming the robot, measure how far its heading 0 is from
    #: the locator frame's +y once, and correct every command by that amount.
    #: `auto_aim()` measures it. With an offset set, `do_reset_aim` must be
    #: False - otherwise the next reset moves the reference the offset is
    #: measured against.
    heading_offset_deg: float = 0.0
    #: Re-measure the heading offset at the start of every run. The aim
    #: reference does not survive a disconnect, and every run opens its own
    #: connection, so measuring it once and reusing it later does not work.
    #: Costs about two seconds and 30 cm of travel per run.
    auto_aim_on_reset: bool = True
    auto_aim_speed: float = 0.2
    #: How far each probe should travel, in metres. Specified as a distance
    #: rather than a step count because the loop period varies by a factor of
    #: three between a healthy link and a struggling one - a fixed step count
    #: silently turns into a metre of travel at a slow loop.
    auto_aim_distance_m: float = 0.20
    #: How long to allow for the ball to come to rest. Measured on this
    #: hardware at 2-3 s: it rolls on and rocks back after the motors stop, so
    #: anything shorter starts the run mid-settle.
    settle_still_timeout_s: float = 9.0
    #: Consecutive quiet samples required before declaring it still. At a
    #: 0.39 s loop, 4 samples is about 1.5 s of confirmed stillness.
    settle_still_samples: int = 4
    #: +1 or -1, multiplying the locator's x axis.
    #:
    #: The locator frame and the heading convention are not guaranteed to have
    #: the same handedness: on this hardware a commanded +90 (which should be
    #: +x) moves the robot in -x. That is a *reflection*, and no heading offset
    #: can correct it - which is why aiming appeared to make things worse. It
    #: also breaks closed-loop control outright, because the controller steers
    #: on measured position: mirrored feedback makes the x error grow instead
    #: of shrink. `auto_aim()` detects it with a second probe.
    locator_x_sign: float = 1.0

    # ---- Heading measurement ---- #
    # `api.get_heading()` returns `self.__heading`, i.e. the heading you last
    # COMMANDED. It is not a measurement, and logging it as one makes the
    # heading channel a copy of the command with no information in it (it
    # tracks the command to within the 1 degree integer quantisation of
    # set_heading, which is exactly what we measured on the week-1 logs).
    #
    # The real measurements are:
    #   api.get_orientation()['yaw']  attitude, degrees, absolute but noisy
    #   api.get_gyroscope()['yaw']    rate, deg/s, smooth but drifts
    #
    # `heading_source` picks between them:
    #   "fused"       complementary filter, gyro integrated + yaw correction
    #   "orientation" raw attitude yaw
    #   "command"     the old behaviour, for comparison only
    heading_source: str = "fused"
    heading_filter_alpha: float = 0.90   # weight on the integrated gyro term
    #: +1 or -1. The attitude yaw sign convention has to be checked against the
    #: command convention on real hardware (spin the robot 90 degrees and see
    #: which way the measurement moves). See MODELLING.md.
    heading_sign: float = 1.0

    # ---- Settling ---- #
    settle_gyro_threshold: float = 8.0   # deg/s; below this the robot is at rest
    settle_min_samples: int = 5          # quiet samples required before starting
    settle_timeout_s: float = 6.0        # give up waiting and proceed anyway

    # ---- Geometry ---- #
    #: Effective track width in metres, used to split body motion into left and
    #: right wheel speeds (Lecture 2). Calibrate it from a spin test: command a
    #: pure rotation and compare gyro yaw rate against wheel speed difference.
    track_width: float = 0.07

    def __init__(
        self,
        api: SpheroEduAPI,
        # Time & dynamics
        dt: float = 0.05,
        max_steps: int = 5000,
        vel_limit: float = 0.5,
        settle_steps: int = 15,

        # World dimensions (meters) - for compatibility
        world_width: float = 2.0,
        world_height: float = 2.0,

        # Goal configuration (meters in world frame)
        goal_pos: Tuple[float, float] = (0.0, 0.0),
        goal_tolerance: float = 0.2,

        # Observation noise (std dev)
        obs_noise_std_pos: float = 0.05,
        obs_noise_std_vel: float = 0.05,

        # Logging
        csv_path: Optional[str] = None,

        # Rendering
        render_mode: Optional[str] = None,
        window_size: Tuple[int, int] = (600, 600),
    ):
        super().__init__()

        self.api = api

        # Logging + visualization settings
        self.last_command = None  # (heading_cmd_rad, speed_cmd_norm, timestamp)

        # Thread safety
        self._lock = threading.Lock()
        self.dt = dt
        self.max_steps = max_steps
        self.vel_limit = vel_limit
        self.settle_steps = int(max(0, settle_steps))

        # World geometry
        self.world_width = world_width
        self.world_height = world_height
        self.x_min = -world_width / 2.0
        self.x_max = world_width / 2.0
        self.y_min = -world_height / 2.0
        self.y_max = world_height / 2.0

        # Noise parameters
        self.obs_noise_std_pos = obs_noise_std_pos
        self.obs_noise_std_vel = obs_noise_std_vel

        # Heading estimator state, all set properly by reset().
        self._gyro_yaw_bias_deg: float = 0.0   # resting gyro bias, deg/s
        self._yaw_zero: float = 0.0            # attitude yaw at reset, radians
        self._heading_est: float = 0.0         # filtered heading, radians
        self._last_core_time: Optional[float] = None
        #: None = not yet calibrated, 0.0 = unusable, else seconds per unit
        self._core_time_scale: Optional[float] = None
        self._core_time_probe: Optional[Tuple[float, float]] = None
        self._core_time_attempts: int = 0

        # Internal states (tracking real robot via odometry)
        self.state_odom: Optional[np.ndarray] = None   # [x_o, y_o, heading_o, speed_o]
        self.state_true: Optional[np.ndarray] = None   # same as odom for real robot
        self.last_obs: Optional[np.ndarray] = None
        self.step_count = 0
        self.latest_est_mean: Optional[np.ndarray] = None
        self.latest_est_cov: Optional[np.ndarray] = None

        # Software collision sensing state. This feeds obs[4] and info["collision"].
        self.last_collision_flag: float = 0.0
        self._last_accel_mag: Optional[float] = None
        self._last_speed_cm_s: Optional[float] = None
        self._last_gyro_mag: Optional[float] = None
        self._last_collision_time: float = 0.0
        self._last_collision_debug: Dict[str, float] = {}

        # Collision thresholds. Tune these after printing collision_debug values
        # while driving into a wall/object a few times.
        self.collision_dead_time: float = 0.25
        self.collision_accel_threshold: float = 1.5
        self.collision_jerk_threshold: float = 0.8
        self.collision_speed_drop_threshold: float = 10.0
        self.stationary_collision_jerk_threshold: float = 1.2
        self.collision_gyro_spike_threshold: float = 100.0
        self.collision_gyro_mag_threshold: float = 200.0

        # Goal
        self.goal_pos = np.array(goal_pos, dtype=np.float32)
        self.goal_tolerance = goal_tolerance
        self.current_setpoint = self.goal_pos.copy()
        self._use_custom_setpoint = False

        # Gym spaces
        obs_low = np.array(
            [self.x_min, self.y_min, -np.pi, -self.vel_limit, 0.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [self.x_max, self.y_max, np.pi, self.vel_limit, 1.0],
            dtype=np.float32,
        )
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-self.vel_limit, -np.pi], dtype=np.float32),
            high=np.array([self.vel_limit, np.pi], dtype=np.float32),
            dtype=np.float32,
        )

        self.vis = Visualiser(
            world_width=self.world_width,
            world_height=self.world_height,
            goal_pos=self.goal_pos,
            render_mode=render_mode,
            window_size=window_size,
            occupancy_grid=None,
            grid_resolution=0.1,
            csv_path=csv_path,
        )

    def start_logging(self):
        self.vis.start_logging()

    def stop_logging(self):
        self.vis.stop_logging()

    def set_log_path(self, csv_path: Optional[str]):
        """Set/override CSV log path for the internal visualiser."""
        self.vis.csv_path = csv_path

    def set_current_setpoint(self, setpoint_xy: Optional[np.ndarray]):
        """Set the current position setpoint used for logging."""
        if setpoint_xy is None:
            self._use_custom_setpoint = False
            self.current_setpoint = self.goal_pos.copy()
            return
        setpoint_arr = np.asarray(setpoint_xy, dtype=np.float32).reshape(-1)
        if setpoint_arr.shape[0] < 2:
            raise ValueError("setpoint_xy must contain at least 2 values [x, y].")
        self.current_setpoint = setpoint_arr[:2].copy()
        self._use_custom_setpoint = True

    def update_estimate(
        self,
        est_mean: Optional[np.ndarray] = None,
        est_cov: Optional[np.ndarray] = None,
    ):
        """Update estimator belief used for rendering and step-wise logging."""
        if est_mean is None or est_cov is None:
            return
        self.vis.set_belief(est_mean, est_cov)
        self.latest_est_mean = np.asarray(est_mean, dtype=np.float32).copy()
        self.latest_est_cov = np.asarray(est_cov, dtype=np.float32).copy()

    def update_visualization(
        self,
        action: Optional[np.ndarray] = None,
        est_mean: Optional[np.ndarray] = None,
        est_cov: Optional[np.ndarray] = None,
    ):
        """Backward-compatible wrapper; logging now happens in step()."""
        _ = action
        self.update_estimate(est_mean=est_mean, est_cov=est_cov)

    def get_live_traj(self):
        """Get current trajectory data.

        Returns:
            traj_x, traj_y, None.
        """
        return self.vis.get_traj()

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _heading_deg_to_rad(heading_deg: float) -> float:
        """Convert hardware heading in degrees to wrapped radians."""
        return Robot._wrap_angle(np.deg2rad(float(heading_deg)))

    def _measured_speed(self, heading: float) -> float:
        """Signed forward speed in m/s.

        `api.get_speed()` echoes the last *commanded* raw value, so using it as
        the measured speed makes the speed column useless for identification.
        `api.get_velocity()` is the motor-encoder velocity in cm/s, expressed in
        the same fixed frame as `get_location()` (x right, y forward, matching
        heading 0 -> +y). Project it onto the heading unit vector.
        """
        if not self.use_encoder_speed:
            return (self.api.get_speed() / 255.0) * self.physical_max_speed
        velocity = self.api.get_velocity()
        if not isinstance(velocity, dict):
            return 0.0
        vx = self.locator_x_sign * float(velocity.get("x", 0.0)) / 100.0  # cm/s -> m/s
        vy = float(velocity.get("y", 0.0)) / 100.0
        return float(vx * np.sin(heading) + vy * np.cos(heading))

    # ---- Raw sensor access ---- #

    def _sensor_dict(self) -> Dict[str, Any]:
        """The API's internal sensor cache, for streams with no public getter.

        Bolt+ streams `quaternion` and `core_time` but `SpheroEduAPI` exposes
        no accessor for them. `core_time` in particular is the robot's own
        clock, which is a far better timestamp than the host wall clock.
        """
        for attr in ("_SpheroEduAPI__sensor_data", "__sensor_data", "sensor_data"):
            data = getattr(self.api, attr, None)
            if isinstance(data, dict):
                return data
        return {}

    def _core_time_raw(self) -> Optional[float]:
        """The device clock in whatever units it reports, or None."""
        data = self._sensor_dict().get("core_time")
        if isinstance(data, dict):
            for key in ("core_time", "time", "value", "t"):
                if key in data:
                    return float(data[key])
        elif isinstance(data, (int, float)):
            return float(data)
        return None

    def core_time(self) -> Optional[float]:
        """The robot's own clock in seconds, or None if unusable.

        The units are not documented and guessing them from magnitude is
        unsafe (a device that has just booted reports a small number whether
        it counts milliseconds or seconds). So the scale is calibrated once,
        against the host clock: if the device delta is about a thousand times
        the wall-clock delta it is milliseconds, if it is about equal it is
        seconds, and anything else means the stream cannot be trusted for
        timing and we fall back to the wall clock.
        """
        raw = self._core_time_raw()
        if raw is None:
            return None
        now = time.time()

        if self._core_time_scale is None:
            if self._core_time_probe is None:
                self._core_time_probe = (raw, now)
                return None
            prev_raw, prev_wall = self._core_time_probe
            d_raw = raw - prev_raw
            d_wall = now - prev_wall
            if d_wall < 1e-3 or d_raw <= 0:
                return None            # not enough separation yet, try again
            ratio = d_raw / d_wall
            if 0.5 <= ratio <= 2.0:
                self._core_time_scale = 1.0
            elif 500.0 <= ratio <= 2000.0:
                self._core_time_scale = 1e-3
            else:
                # One odd ratio is not proof: the stream can lag behind a burst
                # of commands, or a probe can straddle a stall. Retry a few
                # times before giving up on the clock entirely.
                self._core_time_attempts += 1
                if self._core_time_attempts >= 5:
                    self._core_time_scale = 0.0   # unusable; stop trying
                    print(f"[Robot] core_time advances {ratio:.1f}x wall clock; "
                          f"ignoring it and timing steps with the host clock.")
            self._core_time_probe = None

        if not self._core_time_scale:
            return None
        return raw * self._core_time_scale

    def _orientation_yaw_rad(self) -> Optional[float]:
        """Measured attitude yaw in radians, or None if unavailable."""
        orientation = self.api.get_orientation()
        if not isinstance(orientation, dict) or "yaw" not in orientation:
            return None
        return self._wrap_angle(self.heading_sign * np.deg2rad(float(orientation["yaw"])))

    def _gyro_yaw_rate(self) -> float:
        """Measured yaw rate in rad/s, with the resting bias removed."""
        gyro = self.api.get_gyroscope()
        if not isinstance(gyro, dict):
            return 0.0
        raw = float(gyro.get("yaw", gyro.get("z", 0.0)))
        return float(self.heading_sign * np.deg2rad(raw - self._gyro_yaw_bias_deg))

    def _measured_heading(self, dt: float) -> float:
        """Heading in radians, relative to the pose held at reset().

        Complementary filter:

            theta = a * (theta_prev + omega*dt)  +  (1 - a) * yaw_measured

        The gyro term is smooth and has no lag but drifts; the attitude term is
        absolute but noisy. `a` near 1 trusts the gyro over a short horizon and
        lets the attitude pull it back slowly. Angles are blended through
        `wrap(yaw - predicted)` so the filter never sees a 2*pi jump.
        """
        yaw = self._orientation_yaw_rad()
        rate = self._gyro_yaw_rate()

        if self.heading_source == "command":
            return self._heading_deg_to_rad(self.api.get_heading())

        if yaw is None:
            # No attitude stream: integrate the gyro and accept the drift.
            self._heading_est = self._wrap_angle(self._heading_est + rate * dt)
            return self._heading_est

        relative_yaw = self._wrap_angle(yaw - self._yaw_zero)
        if self.heading_source == "orientation":
            self._heading_est = relative_yaw
            return relative_yaw

        predicted = self._wrap_angle(self._heading_est + rate * dt)
        alpha = float(np.clip(self.heading_filter_alpha, 0.0, 1.0))
        correction = self._wrap_angle(relative_yaw - predicted)
        self._heading_est = self._wrap_angle(predicted + (1.0 - alpha) * correction)
        return self._heading_est

    def wheel_speeds(self, forward_speed: float, yaw_rate: float):
        """Split body motion into left/right wheel speeds (Lecture 2).

            v = (v_R + v_L) / 2        omega = (v_R - v_L) / L
            v_R = v + omega*L/2        v_L   = v - omega*L/2

        Requires `track_width` to be calibrated; see the class attributes.
        """
        half = 0.5 * float(self.track_width) * float(yaw_rate)
        return float(forward_speed - half), float(forward_speed + half)

    def sensor_snapshot(self, dt: float = 0.0) -> Dict[str, Any]:
        """Every raw channel worth logging, in one call.

        Saving all of this is the point: you cannot re-derive a channel you did
        not record, and which one turns out to be trustworthy is exactly what
        calibration is supposed to discover.
        """
        velocity = self.api.get_velocity() or {}
        gyro = self.api.get_gyroscope() or {}
        orientation = self.api.get_orientation() or {}
        acceleration = self.api.get_acceleration() or {}

        vx = self.locator_x_sign * float(velocity.get("x", 0.0)) / 100.0
        vy = float(velocity.get("y", 0.0)) / 100.0
        yaw_rate = self._gyro_yaw_rate()
        v_left, v_right = self.wheel_speeds(vy, yaw_rate)

        return {
            # measured
            "meas_yaw_rad": self._orientation_yaw_rad(),
            "meas_yaw_rate_rad_s": yaw_rate,
            "meas_vx_m_s": vx,
            "meas_vy_m_s": vy,
            "meas_wheel_left_m_s": v_left,
            "meas_wheel_right_m_s": v_right,
            "meas_distance_m": (float(self.api.get_distance() or 0.0) / 100.0),
            "core_time_s": self.core_time(),
            # command echoes, kept explicitly labelled so they are never
            # mistaken for measurements again
            "cmd_heading_rad": self._heading_deg_to_rad(self.api.get_heading()),
            "cmd_speed_raw": float(self.api.get_speed() or 0.0),
            # raw, for anything we have not thought of yet
            "gyro_pitch": float(gyro.get("pitch", 0.0)),
            "gyro_roll": float(gyro.get("roll", 0.0)),
            "gyro_yaw": float(gyro.get("yaw", 0.0)),
            "att_pitch": float(orientation.get("pitch", 0.0)),
            "att_roll": float(orientation.get("roll", 0.0)),
            "att_yaw": float(orientation.get("yaw", 0.0)),
            "accel_x": float(acceleration.get("x", 0.0)),
            "accel_y": float(acceleration.get("y", 0.0)),
            "accel_z": float(acceleration.get("z", 0.0)),
        }

    def _locator_xy(self) -> Tuple[float, float]:
        """Locator position in metres, with the x axis handedness corrected."""
        location = self.api.get_location()
        if not isinstance(location, dict):
            return 0.0, 0.0
        return (self.locator_x_sign * float(location.get("x", 0.0)) / 100.0,
                float(location.get("y", 0.0)) / 100.0)

    def _probe(self, heading_deg: int, speed: float, steps: int,
               settle: float) -> Tuple[float, float, float]:
        """Drive a short fixed *distance* at a raw heading.

        Stops as soon as the target distance is covered, so the probe is the
        same size whether the loop is running at 10 Hz or 2.5 Hz.
        """
        x0, y0 = self._locator_xy()
        target = float(self.auto_aim_distance_m)
        # Cap the time as well, in case the robot cannot move at all.
        max_steps = max(int(np.ceil(target / max(speed * self.dt, 1e-3))) + 4, 3)
        self.set_heading_and_speed(int(heading_deg), self.speed_cmd_to_raw(speed))
        for _ in range(max_steps):
            time.sleep(self.dt)
            cx, cy = self._locator_xy()
            if float(np.hypot(cx - x0, cy - y0)) >= target:
                break
        # Let the ball actually come to rest before reading the end point,
        # otherwise the probe measures where it was still rolling to.
        self.wait_until_still(timeout=self.settle_still_timeout_s,
                              quiet_needed=self.settle_still_samples)
        x1, y1 = self._locator_xy()
        dx, dy = x1 - x0, y1 - y0
        return dx, dy, float(np.hypot(dx, dy))

    def wait_until_still(self, timeout: float = 9.0, quiet_needed: int = 4,
                         tolerance_m: float = 0.004) -> Dict[str, Any]:
        """Block until the robot has actually stopped moving.

        A ball does not stop when the motors do: it rolls on, rocks back, and
        settles. Starting a run before that has finished means the first
        samples are the tail of the previous motion, not the experiment - which
        silently corrupts a `hold` run (the noise floor becomes the wobble) and
        gives every other run a non-zero initial velocity the model does not
        know about.

        Position is used rather than the gyro, because the gyro stream is not
        always running.
        """
        self.api.set_speed(0)
        deadline = time.time() + float(timeout)
        previous = self._locator_xy()
        quiet = 0
        samples = 0
        last_speed = 0.0
        while time.time() < deadline:
            time.sleep(self.dt)
            samples += 1
            current = self._locator_xy()
            moved = float(np.hypot(current[0] - previous[0], current[1] - previous[1]))
            previous = current

            # Position alone is not enough: the locator can update more slowly
            # than we poll it, so a stale reading looks exactly like stillness.
            # The encoders are sampled independently, so requiring both to be
            # quiet rejects that false positive.
            velocity = self.api.get_velocity()
            if isinstance(velocity, dict):
                last_speed = float(np.hypot(float(velocity.get("x", 0.0)),
                                            float(velocity.get("y", 0.0)))) / 100.0
            else:
                last_speed = 0.0

            if moved <= tolerance_m and last_speed <= 0.02:
                quiet += 1
            else:
                quiet = 0
            if quiet >= quiet_needed:
                return {"still": True, "samples": samples,
                        "settle_time_s": round(samples * self.dt, 2)}
        return {"still": False, "samples": samples,
                "settle_time_s": round(samples * self.dt, 2),
                "last_speed_m_s": last_speed}

    def auto_aim(self, speed: float = 0.2, steps: int = 8,
                 settle: float = 0.6) -> Dict[str, Any]:
        """Measure how far the robot's heading 0 is from the locator's +y.

        Drives a short probe at commanded heading 0 and looks at which way it
        actually went in locator coordinates. The difference is the offset that
        every later command needs, and it replaces aiming the robot by hand -
        which does not survive a shake, a reconnect, or the next `reset_aim()`.

        Call this once after waking the robot; it returns the offset in degrees
        and applies it to this instance.
        """
        previous_offset = self.heading_offset_deg
        previous_sign = self.locator_x_sign
        self.heading_offset_deg = 0.0
        self.locator_x_sign = 1.0

        # Probe 1: straight ahead. Gives the rotation between the two frames.
        dx0, dy0, d0 = self._probe(0, speed, steps, settle)
        if d0 < 0.03:
            self.heading_offset_deg = previous_offset
            self.locator_x_sign = previous_sign
            return {"ok": False, "travelled_m": d0,
                    "reason": "did not move far enough to measure a direction"}
        course0 = float(np.degrees(np.arctan2(dx0, dy0)))

        # Probe 2: 90 degrees right. Tells us whether the frames have the same
        # handedness. A rotation puts probe 2 at course0 + 90; a reflection
        # puts it at course0 - 90, and no offset can fix that.
        dx1, dy1, d1 = self._probe(90, speed, steps, settle)
        mirrored = False
        turn_deg = None
        if d1 >= 0.03:
            course1 = float(np.degrees(np.arctan2(dx1, dy1)))
            turn_deg = float((course1 - course0 + 180.0) % 360.0 - 180.0)
            mirrored = turn_deg < 0.0

        if mirrored:
            self.locator_x_sign = -1.0
            # Re-derive probe 1's course in the corrected frame.
            course0 = float(np.degrees(np.arctan2(-dx0, dy0)))

        # Commanding 0 produced `course0`, so subtract it from every command.
        self.heading_offset_deg = float(-course0)
        return {"ok": True, "travelled_m": d0,
                "course_deg": course0,
                "turn_deg": turn_deg,
                "mirrored": mirrored,
                "locator_x_sign": self.locator_x_sign,
                "heading_offset_deg": self.heading_offset_deg}

    def speed_cmd_to_raw(self, speed_cmd: float) -> int:
        """Map a speed command in m/s onto the hardware's 0-255 scale.

        `max_raw_command` is applied last and clamps everything, so it holds
        even if the calibration is wrong in the dangerous direction (too small
        a `physical_max_speed` makes the raw command too large, which is what
        sends the ball into a wall).
        """
        magnitude = abs(float(speed_cmd))
        if magnitude <= 0.0:
            return 0

        # Below a threshold raw value the ball does not move at all: the motors
        # cannot break static friction, and the pendulum just leans. Measured on
        # this robot, raw 15 produced anything between 0.001 and 0.124 m/s
        # depending on the surface and the battery - the command was sitting on
        # the stiction threshold, so motion was a coin flip.
        #
        # So the speed range is mapped onto the *usable* part of the raw scale
        # rather than onto 0-255. Physically the ball's speed is roughly linear
        # in raw above the threshold, v = k * (raw - raw_min_move), so inverting
        # that is an affine map, not a floor: a floor would make every small
        # speed command produce the same motion and destroy the model's
        # linearity, whereas this keeps commanded speed proportional to actual
        # speed all the way down to zero.
        raw_min = int(np.clip(self.raw_min_move, 0, 254))
        span = 255 - raw_min
        fraction = min(magnitude / max(self.physical_max_speed, 1e-9), 1.0)
        raw = int(round(raw_min + fraction * span))
        ceiling = int(np.clip(self.max_raw_command, 0, 255))
        return int(np.clip(raw, 0, ceiling))

    # ---- Core API (matching SpheroEnv interface) ---- #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Reset the robot to initial state.
        
        For real hardware, this initializes odometry tracking to origin.
        """
        super().reset(seed=seed)

        # Reset to origin with zero velocity
        x0 = 0.0
        y0 = 0.0
        heading0 = 0.0
        speed0 = 0.0

        self.last_collision_flag = 0.0
        self._last_accel_mag = None
        self._last_speed_cm_s = None
        self._last_gyro_mag = None
        self._last_collision_time = 0.0
        self._last_collision_debug = {}

        # Initialize state
        self.state_odom = np.array([x0, y0, heading0, speed0], dtype=np.float32)
        self.state_true = self.state_odom.copy()
        self.step_count = 0
        self.current_setpoint = self.goal_pos.copy()
        self._use_custom_setpoint = False
        self.latest_est_mean = None
        self.latest_est_cov = None

        self.vis.reset()
        self.vis.set_goal(self.goal_pos)

        # Stop robot
        self.api.set_speed(0)

        # Zero the heading reference to the robot's current physical orientation.
        # Without this, heading=0 is whatever direction the robot happened to be
        # facing when it paired, so the world frame is rotated by an arbitrary
        # unknown offset and sim and real can never line up. Point the robot the
        # way you want heading=0 to mean (i.e. along +y) before calling reset().
        if self.do_reset_aim or self.auto_aim_on_reset:
            try:
                self.api.reset_aim()
            except Exception as exc:  # hardware/API variation - not fatal
                print(f"[Robot] reset_aim() unavailable ({exc}); heading zero is uncalibrated")

        # --- Settle, then calibrate the heading reference ------------------ #
        # The attitude estimate needs time to stop moving after the robot is
        # put down, and the gyro has a resting bias that would otherwise be
        # integrated straight into the heading. So: wait until the gyro is
        # actually quiet, average what it reads while quiet (that is the bias),
        # and take the median attitude yaw over the same window as the zero.
        self._gyro_yaw_bias_deg = 0.0
        self._yaw_zero = 0.0
        self._heading_est = 0.0

        deadline = time.time() + float(self.settle_timeout_s)
        quiet_rates = []
        quiet_yaws = []
        samples = 0
        settled = False

        while time.time() < deadline:
            time.sleep(self.dt)
            samples += 1
            gyro = self.api.get_gyroscope() or {}
            rate_deg = float(gyro.get("yaw", gyro.get("z", 0.0)))
            orientation = self.api.get_orientation() or {}
            yaw_deg = orientation.get("yaw")

            if abs(rate_deg) <= self.settle_gyro_threshold:
                quiet_rates.append(rate_deg)
                if yaw_deg is not None:
                    quiet_yaws.append(float(yaw_deg))
            else:
                # Still moving: throw the window away and start again.
                quiet_rates.clear()
                quiet_yaws.clear()

            if (len(quiet_rates) >= self.settle_min_samples
                    and samples >= self.settle_steps):
                settled = True
                break

        if quiet_rates:
            self._gyro_yaw_bias_deg = float(np.median(quiet_rates))
        if quiet_yaws:
            # Median, not the last sample: one noisy reading should not define
            # the world frame for the whole run.
            self._yaw_zero = self._wrap_angle(
                self.heading_sign * np.deg2rad(float(np.median(quiet_yaws))))
        if not settled:
            print(f"[Robot] heading did not settle within {self.settle_timeout_s}s "
                  f"(gyro still above {self.settle_gyro_threshold} deg/s). "
                  f"Proceeding, but the initial heading may be off.")

        x0, y0 = self._locator_xy()

        heading = self._measured_heading(self.dt)
        speed = self._measured_speed(heading)
        self.state_odom = np.array([x0, y0, heading, speed], dtype=np.float32)
        self.state_true = self.state_odom.copy()

        self._last_core_time = self.core_time()

        # Auto-aim inside this connection, not a previous one.
        #
        # `reset_aim()` sets heading 0 to whatever the drive unit currently
        # faces, and that reference does not survive a disconnect - so an
        # offset measured in an earlier session is meaningless by the time the
        # next run opens its own connection. Measuring it here, right after the
        # aim is taken and before the run starts, is the only placement that is
        # actually valid.
        aim_info = None
        if self.auto_aim_on_reset:
            try:
                aim_info = self.auto_aim(speed=self.auto_aim_speed)
                # The probes left the ball rolling. Wait for it to settle, or
                # the run starts with a velocity nothing in the model knows
                # about - and a `hold` run records the wobble as noise floor.
                still = self.wait_until_still(
                    timeout=self.settle_still_timeout_s,
                    quiet_needed=self.settle_still_samples)
                if aim_info:
                    aim_info["settled_after_aim"] = still
                if still["still"]:
                    print(f"[Robot] settled {still['settle_time_s']}s after auto-aim")
                else:
                    print(f"[Robot] STILL MOVING {still['settle_time_s']}s after "
                          f"auto-aim ({still.get('last_speed_m_s', 0):.3f} m/s) - "
                          f"the run starts with a non-zero velocity")
                # Driving the probe moved us; re-zero the origin so the run
                # still starts from (0, 0).
                x0, y0 = self._locator_xy()
                self.state_odom = np.array([x0, y0, heading, 0.0], dtype=np.float32)
                self.state_true = self.state_odom.copy()
                print(f"[Robot] auto-aim: offset {self.heading_offset_deg:+.1f} deg")
            except Exception as exc:  # noqa: BLE001
                print(f"[Robot] auto-aim failed ({exc}); using the stored offset")

        obs = self._get_observation()
        self.last_obs = obs.copy()

        info = {
            "state_true": self.state_true.copy(),
            "state_odom": self.state_odom.copy(),
            "auto_aim": aim_info,
            "heading_offset_deg": self.heading_offset_deg,
            "settled": settled,
            "gyro_yaw_bias_deg": self._gyro_yaw_bias_deg,
            "yaw_zero_rad": self._yaw_zero,
            "settle_samples": samples,
        }

        return obs, info

    def _sense_collision(self) -> bool:
        """
        Detect collisions from live Sphero sensor readings.

        Uses acceleration jerk, velocity drop, and gyroscope spikes. Updates
        self.last_collision_flag before _get_observation() is called so obs[4]
        reflects the current step's collision state.
        """
        acceleration = self.api.get_acceleration()
        velocity = self.api.get_velocity()
        gyroscope = self.api.get_gyroscope()

        if acceleration is None or velocity is None or gyroscope is None:
            self.last_collision_flag = 0.0
            self._last_collision_debug = {}
            return False

        ax = float(acceleration.get("x", 0.0))
        ay = float(acceleration.get("y", 0.0))
        az = float(acceleration.get("z", 0.0))
        accel_mag = float(np.sqrt(ax * ax + ay * ay + az * az))

        vx = float(velocity.get("x", 0.0))
        vy = float(velocity.get("y", 0.0))
        speed_cm_s = float(np.sqrt(vx * vx + vy * vy))

        gx = float(gyroscope.get("x", gyroscope.get("pitch", 0.0)))
        gy = float(gyroscope.get("y", gyroscope.get("roll", 0.0)))
        gz = float(gyroscope.get("z", gyroscope.get("yaw", 0.0)))
        gyro_mag = float(np.sqrt(gx * gx + gy * gy + gz * gz))

        if self._last_accel_mag is None:
            self._last_accel_mag = accel_mag
            self._last_speed_cm_s = speed_cm_s
            self._last_gyro_mag = gyro_mag
            self.last_collision_flag = 0.0
            self._last_collision_debug = {
                "accel_mag": accel_mag,
                "accel_jerk": 0.0,
                "speed_cm_s": speed_cm_s,
                "speed_drop_cm_s": 0.0,
                "gyro_mag": gyro_mag,
                "gyro_spike": 0.0,
            }
            return False

        accel_jerk = abs(accel_mag - float(self._last_accel_mag))
        speed_drop = float(self._last_speed_cm_s) - speed_cm_s
        gyro_spike = abs(gyro_mag - float(self._last_gyro_mag))
        now = time.time()

        moving_collision = (
            accel_jerk > self.collision_jerk_threshold
            and accel_mag > self.collision_accel_threshold
            and speed_drop > self.collision_speed_drop_threshold
        )
        stationary_impact = (
            speed_cm_s < 5.0
            and accel_jerk > self.stationary_collision_jerk_threshold
        )
        rotation_collision = (
            gyro_spike > self.collision_gyro_spike_threshold
            and gyro_mag > self.collision_gyro_mag_threshold
            and accel_mag > 1.2
        )

        collision = bool(
            (moving_collision or stationary_impact or rotation_collision)
            and (now - self._last_collision_time > self.collision_dead_time)
        )

        self._last_accel_mag = accel_mag
        self._last_speed_cm_s = speed_cm_s
        self._last_gyro_mag = gyro_mag

        self._last_collision_debug = {
            "accel_mag": accel_mag,
            "accel_jerk": accel_jerk,
            "speed_cm_s": speed_cm_s,
            "speed_drop_cm_s": speed_drop,
            "gyro_mag": gyro_mag,
            "gyro_spike": gyro_spike,
        }

        if collision:
            self._last_collision_time = now
            self.last_collision_flag = 1.0
            return True

        self.last_collision_flag = 0.0
        return False

    def step(self, action: np.ndarray):
        """Execute one step with the real robot.

        Args:
            action: Action array [speed, heading].

        Returns:
            Observation, reward, terminated, truncated, info.
        """
        assert self.state_odom is not None, "Call reset() before step()."

        # By default, logged setpoint tracks goal_pos. Wrappers/controllers can
        # override this per-step via set_current_setpoint().
        if not self._use_custom_setpoint:
            self.current_setpoint = np.asarray(self.goal_pos, dtype=np.float32).copy()

        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        speed_cmd, heading_cmd = float(action[0]), float(action[1])
        heading_cmd = self._wrap_angle(heading_cmd)

        # Convert to robot command. Raw power is scaled against the ball's real
        # top speed (physical_max_speed), not vel_limit - see the class docstring.
        # A negative speed is sent as a positive speed on the opposite heading.
        # `heading_offset_deg` rotates every command into the robot's own
        # heading frame, so a command of 0 really does drive along locator +y
        # whatever direction the drive unit was left facing.
        if speed_cmd < 0:
            heading_deg = int((np.degrees(self._wrap_angle(heading_cmd + np.pi))
                               + self.heading_offset_deg) % 360)
        else:
            heading_deg = int((np.degrees(heading_cmd)
                               + self.heading_offset_deg) % 360)
        speed_raw = self.speed_cmd_to_raw(speed_cmd)

        # Send command
        self.set_heading_and_speed(heading_deg, speed_raw)
        self.last_command = (heading_cmd, speed_cmd, time.time())

        # Prefer the robot's own clock for the filter timestep; fall back to the
        # nominal dt if `core_time` is not being streamed.
        core_now = self.core_time()
        if core_now is not None and self._last_core_time is not None:
            step_dt = float(np.clip(core_now - self._last_core_time, 1e-3, 1.0))
        else:
            step_dt = self.dt
        if core_now is not None:
            self._last_core_time = core_now

        heading = self._measured_heading(step_dt)
        speed = self._measured_speed(heading)
        # `locator_x_sign` corrects the handedness mismatch between the locator
        # frame and the heading convention - see the class attribute.
        x_new, y_new = self._locator_xy()

        self.state_odom = np.array([x_new, y_new, heading, speed], dtype=np.float32)
        self.state_true = self.state_odom.copy()

        collision_detected = self._sense_collision()

        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_observation()
        self.last_obs = obs.copy()

        info: Dict[str, Any] = {
            "state_true": self.state_true.copy(),
            "state_odom": self.state_odom.copy(),
            "collision": collision_detected,
            "collision_flag": self.last_collision_flag,
            "collision_debug": self._last_collision_debug.copy(),
            "acceleration": self.api.get_acceleration(),
            "gyroscope": self.api.get_gyroscope(),
            "orientation": self.api.get_orientation(),
            "velocity": self.api.get_velocity(),
            "speed_cmd": speed_cmd,
            "heading_cmd": heading_cmd,
            "speed_raw": speed_raw,
            "step_dt": step_dt,
            "setpoint_xy": self.current_setpoint.copy(),
        }
        # Every raw channel, so nothing has to be re-derived later.
        info["sensors"] = self.sensor_snapshot(step_dt)

        # Log each control step once; estimator updates are handled separately.
        self.vis.record(
            gt_state=self.state_true,
            odom_state=self.state_odom,
            action=np.array([speed_cmd, heading_cmd], dtype=np.float32),
            est_state=self.latest_est_mean,
            est_cov=self.latest_est_cov,
            setpoint=self.current_setpoint,
        )

        return obs, None, terminated, truncated, info
    
    # ---- Interface compatibility methods ---- #

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation from odometry state plus measurement noise.
        """
        assert self.state_odom is not None
        rng = self.np_random

        x_o, y_o, heading_o, speed_o = self.state_odom

        x_meas = x_o + rng.normal(0.0, self.obs_noise_std_pos)
        y_meas = y_o + rng.normal(0.0, self.obs_noise_std_pos)
        heading_meas = self._wrap_angle(heading_o + rng.normal(0.0, self.obs_noise_std_vel))
        speed_meas = speed_o + rng.normal(0.0, self.obs_noise_std_vel)

        obs = np.array(
            [x_meas, y_meas, heading_meas, speed_meas, self.last_collision_flag],
            dtype=np.float32,
        )
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def get_true_state(self) -> np.ndarray:
        """
        Return a copy of the state [x, y, heading, speed].
        For real robot, this is the odometry estimate.
        """
        assert self.state_true is not None
        return self.state_true.copy()

    def get_odom_state(self) -> np.ndarray:
        """
        Return a copy of the odometry state [x, y, heading, speed].
        """
        assert self.state_odom is not None
        return self.state_odom.copy()

    def render(self):
        """
        Delegate rendering to the shared visualiser.
        """
        self.vis.render(self.state_true, self.state_odom)

    def close(self):
        """
        Clean up: stop the robot and close connection if needed.
        """
        self.api.set_speed(0)
        self.vis.close()

    # ---- Robot-specific control methods ---- #

    def set_heading_and_speed(self, heading_deg: float, speed: int):
        """
        Send heading and speed command to real robot.
        
        Args:
            heading_deg: Heading in degrees (0=forward, 90=right, 180=backward, 270=left)
            speed: Speed command in [0, 255]
        """
        with self._lock:
            self.api.set_heading(int(heading_deg))
            self.api.set_speed(int(np.clip(speed, 0, 255)))

    def emergency_stop(self):
        """
        Emergency stop: set speed to 0.
        """
        with self._lock:
            self.api.set_speed(0)

    def set_speed(self, speed: int):
        """
        Set speed directly.
        
        Args:
            speed: Speed command in [0, 255]
        """
        with self._lock:
            self.api.set_speed(int(np.clip(speed, 0, 255)))

    def set_heading(self, heading_deg: float):
        """
        Set heading directly.
        
        Args:
            heading_deg: Heading in degrees
        """
        with self._lock:
            self.api.set_heading(int(heading_deg))
