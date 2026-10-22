"""Focused contract tests for the Lab 4 learned-navigation layer."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import lab4
from Policy import FEATURE_NAMES, Policy, torch
from training import Rollout, split_by_rollout


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch is not installed")


def _rollout(index: int) -> Rollout:
    return Rollout(
        run_id=f"run-{index}",
        source="real",
        observations=np.zeros((8, 4), dtype=np.float32),
        actions=np.zeros((8, 2), dtype=np.float32),
    )


def test_validation_split_holds_out_whole_rollouts() -> None:
    training, validation = split_by_rollout([_rollout(i) for i in range(10)], seed=7)
    training_ids = {rollout.run_id for rollout in training}
    validation_ids = {rollout.run_id for rollout in validation}
    assert training_ids
    assert validation_ids
    assert training_ids.isdisjoint(validation_ids)


def test_policy_marker_contract_and_bounds() -> None:
    policy = Policy()
    observations = torch.tensor([
        [-0.5, -0.5, 0.0, 0.0],
        [0.5, 0.5, 3.14, 0.1],
    ], dtype=torch.float32)
    features = policy.features(observations)
    actions = policy(observations)
    assert features.shape == (2, len(FEATURE_NAMES))
    assert actions.shape == (2, 2)
    assert bool(torch.all(torch.isfinite(actions)))
    assert bool(torch.all(actions[:, 0] >= 0.0))
    assert bool(torch.all(actions[:, 0] <= policy.max_speed_cmd + 1e-7))
    assert bool(torch.all(actions[:, 1] >= -np.pi))
    assert bool(torch.all(actions[:, 1] <= np.pi))


def test_trained_checkpoint_loads_with_submission_policy() -> None:
    checkpoint = HERE / "weight.pth"
    assert checkpoint.exists()
    policy = lab4.load_policy(checkpoint)
    action = policy(torch.tensor([-0.5, -0.5, 0.0, 0.0], dtype=torch.float32))
    assert tuple(action.shape) == (2,)
    assert bool(torch.all(torch.isfinite(action)))


def test_documented_model_contract_matches_runtime() -> None:
    contract = lab4.model_contract()
    assert contract["raw_state"] == ["x", "y", "heading", "speed"]
    assert contract["action"] == ["speed_cmd", "absolute heading_cmd"]
    assert contract["features"] == list(FEATURE_NAMES)


# ------------------------------------------------- MPC fault detector, rule by rule
#
# The simulator never drives the ball into trouble, so a clean run only shows
# the detector stays silent. These inject each fault directly - and check the
# combined rules do NOT fire on any one signal alone.

ROUTE = [np.array([-0.5, -0.5]), np.array([-0.5, 0.0]), np.array([-0.25, 0.0])]
EXPERT = np.array([0.03, 0.0])
DT = 0.18


def _detector(near_wall=lambda p: False) -> "lab4.MPCSafety":
    return lab4.MPCSafety(route=ROUTE, recovery_speed=0.025, near_wall=near_wall)


def _step(layer, xy, action=(0.04, 0.0), wall_predicted=False, jolt=False):
    return layer.apply([xy[0], xy[1], 0.0, 0.1], list(action), EXPERT,
                       wall_predicted=wall_predicted, dt=DT, collision_flag=jolt)


def test_detector_is_silent_when_the_ball_behaves() -> None:
    # Same order as the controller: apply this step, then record where the model
    # expects the ball on the NEXT step.
    layer = _detector()
    for k in range(12):
        out, reason = _step(layer, (-0.5, -0.5 + 0.03 * k))
        assert reason is None and np.allclose(out, [0.04, 0.0])
        layer.expect((-0.5, -0.5 + 0.03 * (k + 1)))      # model predicts it exactly
    assert layer.takeovers == 0


def test_non_finite_command_goes_to_the_expert() -> None:
    out, reason = _step(_detector(), (-0.5, -0.4), action=(float("nan"), 0.0))
    assert reason == "non-finite command" and np.allclose(out, EXPERT)


def test_imu_jolt_alone_is_not_a_collision() -> None:
    _out, reason = _step(_detector(near_wall=lambda p: False), (-0.5, -0.4), jolt=True)
    assert reason is None


def test_imu_jolt_and_a_wall_nearby_is_a_collision() -> None:
    out, reason = _step(_detector(near_wall=lambda p: True), (-0.5, -0.4), jolt=True)
    # On the route already, so it backs away from whatever it hit.
    assert reason == "collision" and out[0] < 0


def test_one_noisy_step_off_track_is_ignored_but_a_sustained_one_is_not() -> None:
    layer = _detector()
    _o, first = _step(layer, (-0.425, -0.3))           # 7.5 cm off, one step
    assert first is None
    reasons = [_step(layer, (-0.425, -0.3))[1] for _ in range(3)]
    assert "off corridor" in reasons                   # >= 0.4 s off


def test_far_off_track_acts_at_once() -> None:
    _o, reason = _step(_detector(), (-0.41, -0.3))      # 9 cm off: wall imminent
    assert reason == "off corridor"


def test_scraping_along_a_wall_while_still_moving_is_caught() -> None:
    # Moving every step, but only half as far as the model predicted - the
    # IMU says nothing and it is not stuck, yet it is rubbing on something.
    layer = _detector()
    y, reasons = -0.5, []
    for _ in range(6):
        _o, reason = _step(layer, (-0.5, y))
        reasons.append(reason)
        layer.expect((-0.5, y + 0.06))                  # model: 6 cm next step
        y += 0.02                                       # reality: 2 cm
    assert "scrape" in reasons
    assert reasons[0] is None and reasons[1] is None    # not on the first shortfall


def test_stuck_for_a_second_triggers_recovery() -> None:
    layer = _detector()
    reasons = [_step(layer, (-0.5, -0.3))[1] for _ in range(10)]
    assert "stall recovery" in reasons
    # and the recovery is held, not a single step
    i = reasons.index("stall recovery")
    assert reasons[i + 1:i + layer.recovery_steps] == ["recovering"] * (layer.recovery_steps - 1)


def test_wall_stops_escalate_instead_of_deadlocking() -> None:
    # Two robot runs froze because every candidate kept predicting a wall and
    # the layer kept stopping. It must give up stopping and recover.
    layer = _detector()
    # Remember the direction that really drove the ball. The zero-speed wall
    # predictions which follow may point somewhere else and must not replace it.
    _step(layer, (-0.5, -0.3), action=(0.04, 1.2))
    outputs = [_step(layer, (-0.5, -0.3), action=(0.0, -2.4), wall_predicted=True)
               for _ in range(7)]
    reasons = [reason for _out, reason in outputs]
    assert reasons[:layer.wall_stop_limit] == ["predicted wall contact"] * layer.wall_stop_limit
    recovery = reasons.index("wall contact recovery")
    assert outputs[recovery][0][0] < 0.0
    assert outputs[recovery][0][1] == pytest.approx(1.2)
    held = [out for out, reason in outputs[recovery + 1:] if reason == "recovering"]
    assert len(held) == layer.recovery_steps - 1
    assert all(np.allclose(out, outputs[recovery][0]) for out in held)
