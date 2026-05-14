from __future__ import annotations

from irrgate.config import Config
from irrgate.data.loader import Trajectory
from irrgate.evaluation.metrics import compute_false_positive_rate, compute_recall
from irrgate.evaluation.runner import TrajectoryResult, evaluate_trajectory
from irrgate.gate import GateDecision


def make_trajectory(steps: list[dict], side_effect: str = "No") -> Trajectory:
    return Trajectory(
        benchmark="webarena",
        task_id="test",
        model="gpt-4",
        goal="Test",
        steps=steps,
        side_effect_label=side_effect,
    )


def make_step(action: str, axtree: str = "", url: str = "https://example.com") -> dict:
    return {
        "action": action,
        "axtree": axtree,
        "url": url,
        "reasoning": "",
        "bounding_boxes": [],
    }


def test_evaluate_trajectory_collects_all_step_decisions():
    trajectory = make_trajectory([
        make_step("noop()"),
        make_step("goto('https://example.com')"),
    ])
    config = Config()

    result = evaluate_trajectory(trajectory, config)

    assert isinstance(result, TrajectoryResult)
    assert len(result.step_decisions) == 2
    assert all(isinstance(d, GateDecision) for d in result.step_decisions)
    assert result.trajectory == trajectory


def test_evaluate_trajectory_identifies_first_blocking_step():
    trajectory = make_trajectory([
        make_step("noop()"),
        make_step("fill('100', 'text')", "[100] role='textbox' name='Input'"),
        make_step("click('200')", "[200] role='button' name='Submit'"),
    ])
    config = Config()

    result = evaluate_trajectory(trajectory, config)

    # The submit click should be blocked due to missing consent
    assert result.first_blocking_step == 2
    assert not result.reached_completion


def test_evaluate_trajectory_reached_completion_when_no_blocks():
    trajectory = make_trajectory([
        make_step("noop()"),
        make_step("goto('https://example.com')"),
    ])
    config = Config()

    result = evaluate_trajectory(trajectory, config)

    assert result.first_blocking_step is None
    assert result.reached_completion


def test_compute_recall_all_blocked():
    # Create trajectories with L3 actions at step 2
    steps_with_l3 = [
        make_step("noop()"),
        make_step("noop()"),
        make_step("click('123')", "[123] role='button' name='Submit'"),  # L3 submit
    ]
    results = [
        TrajectoryResult(
            trajectory=make_trajectory(steps_with_l3, "Yes"),
            step_decisions=[],
            first_blocking_step=0,  # blocked before side-effect
            reached_completion=False,
        ),
        TrajectoryResult(
            trajectory=make_trajectory(steps_with_l3, "Yes"),
            step_decisions=[],
            first_blocking_step=2,  # blocked at side-effect step
            reached_completion=False,
        ),
    ]

    recall = compute_recall(results)
    assert recall == 1.0


def test_compute_recall_none_blocked():
    # Create trajectories with L3 actions at step 2
    steps_with_l3 = [
        make_step("noop()"),
        make_step("noop()"),
        make_step("click('123')", "[123] role='button' name='Submit'"),  # L3 submit
    ]
    results = [
        TrajectoryResult(
            trajectory=make_trajectory(steps_with_l3, "Yes"),
            step_decisions=[],
            first_blocking_step=None,  # not blocked
            reached_completion=True,
        ),
        TrajectoryResult(
            trajectory=make_trajectory(steps_with_l3, "Yes"),
            step_decisions=[],
            first_blocking_step=None,  # not blocked
            reached_completion=True,
        ),
    ]

    recall = compute_recall(results)
    assert recall == 0.0


def test_compute_recall_mixed():
    # Create trajectories with L3 actions at step 2
    steps_with_l3 = [
        make_step("noop()"),
        make_step("noop()"),
        make_step("click('123')", "[123] role='button' name='Submit'"),  # L3 submit
    ]
    results = [
        TrajectoryResult(
            trajectory=make_trajectory(steps_with_l3, "Yes"),
            step_decisions=[],
            first_blocking_step=0,  # blocked before side-effect
            reached_completion=False,
        ),
        TrajectoryResult(
            trajectory=make_trajectory(steps_with_l3, "Yes"),
            step_decisions=[],
            first_blocking_step=None,  # not blocked
            reached_completion=True,
        ),
    ]

    recall = compute_recall(results)
    assert recall == 0.5


def test_compute_recall_empty_list():
    recall = compute_recall([])
    assert recall == 0.0


def test_compute_false_positive_rate():
    results = [
        TrajectoryResult(
            trajectory=make_trajectory([], "No"),
            step_decisions=[],
            first_blocking_step=None,
            reached_completion=True,
        ),
        TrajectoryResult(
            trajectory=make_trajectory([], "No"),
            step_decisions=[],
            first_blocking_step=1,
            reached_completion=False,
        ),
    ]

    fpr = compute_false_positive_rate(results)
    assert fpr == 0.5