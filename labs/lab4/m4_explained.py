"""M4 in one file: learn how the robot moves, then use that to choose commands.

This is a short, readable version of what the real code does
(dynamics_learning.py and learned_mpc.py). Run it to see each step:

    python labs/lab4/m4_explained.py

It only reads files; it never changes the saved model or any logs.

    STEP 1  data      logged robot runs  ->  training pairs (X, Y)
    STEP 2  model     a small neural network: 6 inputs -> 4 outputs
    STEP 3  loss      how wrong the prediction is (normalised MSE)
    STEP 4  training  adjust the weights to make the loss small
    STEP 5  control   try 30 commands through the model, send the best
"""

import csv
import math
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "lab1" / "runs"          # every logged robot run lives here


# ---------------------------------------------------------------------------
# STEP 1  DATA: turn a logged run into training pairs
# ---------------------------------------------------------------------------
# Each row of a run's data.csv has the ball's position (odom_x, odom_y), the
# command we sent (speed_cmd, heading_cmd) and the time step (dt).
#
# For every time step t we ask: "the ball was moving like THIS, we commanded
# THAT - what did it do next?"
#
#   X (6 inputs)  = [v_fwd, v_lat, speed_cmd, prev_fwd, prev_lat, dt]
#   Y (4 targets) = [d_fwd, d_lat, v_fwd_next, v_lat_next]
#
# "fwd/lat" means along / across the commanded heading. Measuring everything
# relative to the command makes the model the same everywhere in the maze.

def to_command_frame(vector, heading):
    """Split a world (x, y) vector into (along the command, across it).
    Heading 0 points to world +y, heading pi/2 to world +x."""
    forward = np.array([math.sin(heading), math.cos(heading)])
    across = np.array([math.cos(heading), -math.sin(heading)])
    return float(vector @ forward), float(vector @ across)


def pairs_from_run(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    data = []
    for r in rows:
        try:
            row = [float(r["odom_x"]), float(r["odom_y"]),
                   float(r["speed_cmd"]), float(r["heading_cmd"]), float(r["dt"])]
        except (KeyError, ValueError, TypeError):
            continue                                       # skip incomplete rows
        if all(math.isfinite(v) for v in row):             # and rows with nan/inf
            data.append(row)
    if len(data) < 3:
        return [], []
    data = np.array(data)
    X, Y = [], []
    for t in range(1, len(data) - 1):                      # needs t-1 and t+1
        pos_before, pos_now, pos_next = data[t - 1, :2], data[t, :2], data[t + 1, :2]
        speed_cmd, heading = data[t, 2], data[t, 3]
        prev_speed, prev_heading = data[t - 1, 2], data[t - 1, 3]
        dt_in, dt_out = data[t, 4], data[t + 1, 4]
        step_in, step_out = pos_now - pos_before, pos_next - pos_now
        # throw away bad samples: Bluetooth stalls and locator jumps
        if not (0.05 <= dt_in <= 1.2 and 0.05 <= dt_out <= 1.2):
            continue
        if np.linalg.norm(step_in) > 0.15 or np.linalg.norm(step_out) > 0.15:
            continue
        v_fwd, v_lat = to_command_frame(step_in / dt_in, heading)          # how it was moving
        d_fwd, d_lat = to_command_frame(step_out, heading)                 # how far it went
        v_fwd2, v_lat2 = to_command_frame(step_out / dt_out, heading)      # how fast after
        turn = prev_heading - heading
        prev_fwd, prev_lat = prev_speed * math.cos(turn), prev_speed * math.sin(turn)
        X.append([float(v) for v in (v_fwd, v_lat, speed_cmd, prev_fwd, prev_lat, dt_out)])
        Y.append([float(v) for v in (d_fwd, d_lat, v_fwd2, v_lat2)])
    return X, Y


# ---------------------------------------------------------------------------
# STEP 2  MODEL: a small neural network
# ---------------------------------------------------------------------------
# 6 inputs -> 64 -> 64 -> 4 outputs, tanh in between (smooth).
# Inputs are standardised (subtract mean, divide by std) before the network,
# and outputs are un-standardised after, so the network only sees numbers
# around 0 and 1 whatever their units are.

class DynamicsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(6, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 4),
        )
        self.register_buffer("x_mean", torch.zeros(6))
        self.register_buffer("x_std", torch.ones(6))
        self.register_buffer("y_mean", torch.zeros(4))
        self.register_buffer("y_std", torch.ones(4))

    def forward(self, x):
        return self.net((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean


# ---------------------------------------------------------------------------
# STEP 3  LOSS: how wrong is the prediction?
# ---------------------------------------------------------------------------
#   loss = mean over samples and the 4 outputs of ((predicted - real) / y_std)^2
#
# Dividing by y_std (each output's spread) matters: displacements are a few
# millimetres and velocities a tenth of a metre per second, so without it the
# loss would only care about velocity. This way all four outputs count equally.

def loss_fn(model, x, y):
    return (((model(x) - y) / model.y_std) ** 2).mean()


# ---------------------------------------------------------------------------
# STEP 4  TRAINING
# ---------------------------------------------------------------------------
# Whole runs are held out for validation (not random rows), so the score is
# honest. Adam adjusts the weights to lower the loss; we keep the weights
# from the epoch that did best on the held-out runs.

def train(X_train, Y_train, X_val, Y_val, epochs=40):
    torch.manual_seed(0)
    model = DynamicsModel()
    model.x_mean.copy_(X_train.mean(0)); model.x_std.copy_(X_train.std(0) + 1e-6)
    model.y_mean.copy_(Y_train.mean(0)); model.y_std.copy_(Y_train.std(0) + 1e-6)
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    best, best_weights = float("inf"), None
    for epoch in range(epochs):
        order = torch.randperm(len(X_train))
        for start in range(0, len(order), 512):          # batches of 512
            batch = order[start:start + 512]
            optimiser.zero_grad()
            loss = loss_fn(model, X_train[batch], Y_train[batch])
            loss.backward()                               # gradients
            optimiser.step()                              # update weights
        with torch.no_grad():
            val = float(loss_fn(model, X_val, Y_val))
        if val < best:
            best, best_weights = val, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"    epoch {epoch:3d}   held-out loss {val:.3f}")
    model.load_state_dict(best_weights)
    return model, best


# ---------------------------------------------------------------------------
# STEP 5  CONTROL: use the model to choose a command (MPC)
# ---------------------------------------------------------------------------
# Every control step: make 30 candidate commands (6 speeds x 5 headings),
# predict where each one takes the ball over the next ~1.2 s with the model,
# score it, send the cheapest. (The real controller also checks walls and
# where the ball would stop if the drive were cut; this shows the core idea.)

def predict(model, pos, vel, speed_cmd, heading, prev_speed, prev_heading, dt):
    """One step through the model, back in world coordinates."""
    v_fwd, v_lat = to_command_frame(vel, heading)
    turn = prev_heading - heading
    x = torch.tensor([[v_fwd, v_lat, speed_cmd, prev_speed * math.cos(turn),
                       prev_speed * math.sin(turn), dt]], dtype=torch.float32)
    with torch.no_grad():
        d_fwd, d_lat, v_fwd2, v_lat2 = model(x)[0].tolist()
    forward = np.array([math.sin(heading), math.cos(heading)])
    across = np.array([math.cos(heading), -math.sin(heading)])
    return pos + d_fwd * forward + d_lat * across, v_fwd2 * forward + v_lat2 * across


def choose_command(model, pos, vel, target, cruise=0.081, dt=0.2, steps=6):
    aim = math.atan2(target[0] - pos[0], target[1] - pos[1])        # heading to target
    best = None
    for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        for offset in (-0.45, -0.22, 0.0, 0.22, 0.45):
            speed, heading = fraction * cruise, aim + offset
            p, v, prev_s, prev_h = pos.copy(), vel.copy(), speed, heading
            lateral_cost = 0.0
            for _ in range(steps):                                  # 6 x 0.2 s = 1.2 s
                p, v = predict(model, p, v, speed, heading, prev_s, prev_h, dt)
                # distance off the straight line to the target
                to_t = target - pos
                lateral = abs(to_t[0] * (p - pos)[1] - to_t[1] * (p - pos)[0]) / np.linalg.norm(to_t)
                lateral_cost += lateral ** 2
            progress = np.linalg.norm(target - pos) - np.linalg.norm(target - p)
            overshoot = max(0.0, float((p - target) @ ((target - pos) / np.linalg.norm(target - pos))))
            cost = -30 * progress + 600 * lateral_cost + 8000 * overshoot ** 2 + 3000 * overshoot
            if best is None or cost < best[0]:
                best = (cost, speed, heading, p)
    return best


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("STEP 1  data")
    runs = sorted(d for d in RUNS.iterdir() if "__real__" in d.name and (d / "data.csv").exists())
    pairs = [pairs_from_run(d / "data.csv") for d in runs]
    pairs = [(x, y) for x, y in pairs if len(x) >= 4]
    print(f"    {len(pairs)} real runs, {sum(len(x) for x, _ in pairs)} training pairs")
    x0, y0 = pairs[-1][0][len(pairs[-1][0]) // 2], pairs[-1][1][len(pairs[-1][1]) // 2]
    print("    one pair:  X =", [round(v, 3) for v in x0])
    print("               Y =", [round(v, 4) for v in y0])

    print("STEP 2-4  model, loss, training (40-epoch demo; the real model trained 400)")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(pairs))
    n_val = max(1, len(pairs) // 5)                             # 20 % of RUNS held out
    val_runs, train_runs = order[:n_val], order[n_val:]
    def stack(ids):
        X = torch.tensor([r for i in ids for r in pairs[i][0]], dtype=torch.float32)
        Y = torch.tensor([r for i in ids for r in pairs[i][1]], dtype=torch.float32)
        return X, Y
    X_train, Y_train = stack(train_runs); X_val, Y_val = stack(val_runs)
    model, best = train(X_train, Y_train, X_val, Y_val)
    print(f"    best held-out loss {best:.3f}")

    print("STEP 5  control, with the REAL trained model the robot uses")
    real_model = DynamicsModel()
    real_model.load_state_dict(torch.load(HERE / "dynamics_model_real.pth", map_location="cpu"))
    real_model.eval()
    pos, vel = np.array([-0.5, -0.30]), np.array([0.0, 0.10])   # going up the first straight
    target = np.array([-0.5, 0.0])                              # corner 1
    cost, speed, heading, end = choose_command(real_model, pos, vel, target)
    print(f"    ball at {pos.tolist()} moving {vel.tolist()} m/s, corner 1 at {target.tolist()}")
    print(f"    chosen: speed_cmd {speed:.3f}, heading {math.degrees(heading):+.0f} deg "
          f"-> predicted position after 1.2 s {end.round(3).tolist()}")
    pos2 = np.array([-0.5, -0.08])                              # 8 cm before the corner
    cost, speed, heading, end = choose_command(real_model, pos2, vel, target)
    print(f"    8 cm before the corner the same model chooses speed_cmd {speed:.3f} "
          f"(it predicts the coast and brakes)")
