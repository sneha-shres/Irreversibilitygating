from __future__ import annotations

from dataclasses import dataclass

from irrgate.config import Config
from irrgate.data.loader import Trajectory
from irrgate.gate import GateDecision, gate_step
from irrgate.routing import Regime
from irrgate.taxonomy import Level


def make_step(action: str, axtree: str = "", url: str = "https://example.com") -> dict[str, str]:
    return {
        "action": action,
        "axtree": axtree,
        "url": url,
        "reasoning": "",
        "bounding_boxes": [],
    }


def make_trajectory(steps: list[dict[str, str]], side_effect_label: str = "No") -> Trajectory:
    return Trajectory(
        benchmark="webarena",
        task_id="test",
        model="gpt-4",
        goal="Test plan",
        steps=steps,
        side_effect_label=side_effect_label,
    )


def test_pure_read_only_trajectory_approves_all_steps():
    trajectory = make_trajectory([
        make_step("noop()", ""),
        make_step("goto('https://example.com')", ""),
    ])
    config = Config()

    for idx in range(len(trajectory.steps)):
        decision = gate_step(trajectory, idx, config)
        assert decision.decision == "approve"
        assert decision.regime == Regime.BYPASS
        assert decision.blocking_reason is None


def test_draft_then_submit_blocks_submit_for_missing_consent():
    trajectory = make_trajectory([
        make_step("fill('100', 'hello')", "[100] role='textbox' name='Message'"),
        make_step("click('200')", "[200] role='button' name='Submit'"),
    ])
    config = Config()

    first = gate_step(trajectory, 0, config)
    second = gate_step(trajectory, 1, config)

    assert first.decision == "approve"
    assert second.decision == "block"
    assert second.rubric_verdicts is not None
    assert second.rubric_verdicts["R3"] is False
    assert second.regime == Regime.GATED


def test_submit_after_consent_approves_submit():
    trajectory = make_trajectory([
        make_step("send_msg_to_user('Please confirm')", "[200] role='button' name='Submit'"),
        make_step("click('200')", "[200] role='button' name='Submit'"),
    ])
    config = Config()

    second = gate_step(trajectory, 1, config)
    assert second.decision == "approve"
    assert second.rubric_verdicts is not None
    assert second.rubric_verdicts["R3"] is True


def test_multi_L3_trajectory_routes_to_gated_regime():
    trajectory = make_trajectory([
        make_step("click('100')", "[100] role='button' name='Delete'"),
        make_step("click('200')", "[200] role='button' name='Submit'"),
    ])
    config = Config()

    second = gate_step(trajectory, 1, config)
    assert second.regime == Regime.GATED
    assert second.profile.f == 1
    assert second.profile.d_I == 1.0


def test_L3_after_exploration_can_route_to_low_or_gated():
    trajectory = make_trajectory([
        make_step("fill('100', 'Email')", "[100] role='textbox' name='Email'"),
        make_step("fill('101', 'Address')", "[101] role='textbox' name='Address'"),
        make_step("fill('102', 'Phone')", "[102] role='textbox' name='Phone'"),
        make_step("fill('103', 'City')", "[103] role='textbox' name='City'"),
        make_step("send_msg_to_user('Please confirm')", "[200] role='button' name='Submit'"),
        make_step("click('200')", "[200] role='button' name='Submit'"),
    ])
    config = Config()

    second = gate_step(trajectory, 5, config)
    assert second.regime in {Regime.LOW, Regime.GATED}
    assert second.profile.pi < 1.0
