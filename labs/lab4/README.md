# Lab 4 learned navigation

Lab 4 changes one block in the existing robot stack: the policy chooses the
control action. It does not discard the work from Labs 1-3.

```text
Sphero sensors -> Lab 2 EKF state -> learned Policy -> safety layer -> Sphero
                              A* + PID expert -----------^ (fallback only)
Lab 1 dynamics -----------------------> simulator and EKF prediction
```

## Method

This implementation uses **behaviour cloning**. Passing Lab 3 runs already
contain an expert demonstration from the shortest A* route and the tuned PID.
The trainer learns the mapping from the state available on the robot to the
command the expert applied.

The default dataset is all passing, collision-free rows from real Lab 3 runs.
Simulator and real runs have different command distributions, so mixing them
without a domain flag teaches an ambiguous speed target. The real data is the
right default for the assessed physical robot. Validation holds out complete
runs, rather than shuffling adjacent timesteps from the same run onto both
sides of the split. Intermediate zero-speed pivot rows are excluded because
the Lab 3 follower's hidden corner-pivot latch is not visible to this
memoryless policy; the deployed safety layer still performs those pivots.

## Independent and dependent variables

Raw independent state from the Lab 2 EKF:

```text
x_k = [x, y, heading, speed]
```

The policy deterministically expands it to:

```text
[x/s, y/s, (goal_x-x)/s, (goal_y-y)/s,
 sin(heading), cos(heading), speed/0.16, collision]
```

where `s = 0.625 m` is the half-width of the official world. Absolute position
is kept because the maze walls are fixed in the world frame; goal-relative
features alone cannot tell the policy where the corners are.

Dependent targets:

```text
y_k = [speed_cmd, sin(heading_cmd), cos(heading_cmd)]
```

Deployed output:

```text
u_k = [speed_cmd, atan2(predicted_sin, predicted_cos)]
```

The heading command is an **absolute angle**, not a turn rate. Heading `0` is
world `+y`, and `+pi/2` is world `+x`, matching the Sphero environment.

## Equations

The course's discrete motion-model structure is retained:

```text
x_(k+1) = f_d(x_k, u_k, epsilon_k)
```

Its planar position update uses the Sphero heading convention:

```text
x_(k+1) = x_k + v_(k+1) sin(theta_(k+1)) dt
y_(k+1) = y_k + v_(k+1) cos(theta_(k+1)) dt
```

The behaviour-cloning objective is:

```text
L = MSE(speed_hat/max_speed_cmd, speed/max_speed_cmd)
  + lambda * mean((sin(theta_hat)-sin(theta))^2
                + (cos(theta_hat)-cos(theta))^2)
```

Sine/cosine removes the artificial discontinuity at `-pi/+pi`.

The articulated-robot material uses the same important idea of a mapping
between configuration/state and task/output spaces. Its manipulator-specific
mass matrices, joint torques and Jacobians are not inserted into this mobile
robot controller because the Sphero has no articulated links or joint-torque
interface.

## Training and running

From the repository root:

```powershell
python -m pip install -r requirements.txt
python labs/lab4/training.py --sources real --epochs 300
python labs/lab4/lab4.py --sim --no-render --speed-scale 2.0
```

Use the dashboard's **Lab 4 (Learn)** tab for the same workflow with dataset
counts, live loss curves, held-out diagnostics and simulator evaluation.

Generated files:

- `labs/lab4/weight.pth`: submission-compatible model state
- `labs/lab4/training_metadata.json`: split, hyperparameters and diagnostics
- `labs/lab4/training_history.json`: train/validation loss by epoch
- `labs/lab4/exports/<student>_lab4.csv`: four-column trajectory export

## Safety and assessment

The learned action is bounded and rolled forward through the measured Lab 1
dynamics. A predicted collision, route departure, impossible state or stall
causes A* + PID to take over for that step. Every takeover and reason is logged.

Two of those rules were set by measurement rather than by taste:

- **Corners belong to the geometric controller.** The trainer drops the expert's
  zero-speed pivot rows, because a memoryless policy sees one state carrying two
  labels, "stop and turn" and "drive on". The policy therefore only ever learned
  the smooth part of the demonstration and rounded every corner: 0.072 m of
  error at the tightest one against 0.016 m mean on the straights. Handing back
  the three route corners cuts the worst error to 0.019 m. The policy still
  drives every straight, which is most of the run.
- **The route limit has to trip before the wall, not after it.** The simulator
  collides once the ball centre leaves its 0.125 m cell, at 6.25 cm, so the
  original 0.10 m limit only fired once the ball had already hit something. In
  simulation that is unrecoverable: `SpheroEnv` resolves the true and odometry
  positions separately, so a jammed ball keeps reporting motion and the
  estimator never learns it is stuck.

`--speed-scale` multiplies the learned speed command, clipped to the
environment's `vel_limit`. It exists because every usable demonstration cruised
at 0.040 m/s, so the policy cannot ask for more than it was shown. In simulation
a scale of 2.0 saturates `vel_limit` and takes the run from 224 steps to 189
with no loss of accuracy. **On hardware, start at 1.0 and raise it gradually**:
the demonstrations were recorded on a slower cruise and nothing above it has
been tested on the physical floor.

The assessment evidence supplied on Ed says the two demonstration criteria are
Manhattan distance to the goal and elapsed time from the first physical motion.
The whole ball must remain inside the 25 cm goal plate for full goal credit,
which means its centre must be within 8.85 cm of the plate centre on each axis.
M4 has different timing requirements from M3, but no numeric M4 threshold was
included in the supplied notice; keep the dashboard deadline configurable and
confirm the Week 10 value with the teaching team.
