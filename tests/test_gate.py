from __future__ import annotations

from irrgate.actions import Action
from irrgate.classifier import classify
from irrgate.config import Config
from irrgate.data.loader import Trajectory
from irrgate.gate import GateDecision, gate_decision, make_gate_decision
from irrgate.profile import RiskProfile
from irrgate.taxonomy import Level


def gate_step(trajectory: Trajectory, step_index: int, config: Config) -> GateDecision:
    if step_index < 0 or step_index >= len(trajectory.steps):
        raise IndexError("step_index is out of range for the trajectory")
    actions: list[Action] = []
    levels: list[Level] = []
    for idx, step in enumerate(trajectory.steps[: step_index + 1]):
        action = Action.from_step(step, step_index=idx)
        level = classify(action)
        actions.append(action)
        levels.append(level)
    return make_gate_decision(actions, levels, config)


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


# ── gate_decision unit tests ────────────────────────────────────────────────

def test_gate_decision_approves_when_f_zero():
    profile = RiskProfile(f=0, d_I=0.0, pi=0.0)
    assert gate_decision(profile, tau_d=0.15, tau_pi=0.30) == "approve"


def test_gate_decision_approves_when_both_below_thresholds():
    profile = RiskProfile(f=1, d_I=0.1, pi=0.2)
    assert gate_decision(profile, tau_d=0.15, tau_pi=0.30) == "approve"


def test_gate_decision_blocks_when_d_I_meets_threshold():
    profile = RiskProfile(f=1, d_I=0.15, pi=0.0)
    assert gate_decision(profile, tau_d=0.15, tau_pi=0.30) == "block"


def test_gate_decision_blocks_when_pi_meets_threshold():
    profile = RiskProfile(f=1, d_I=0.0, pi=0.30)
    assert gate_decision(profile, tau_d=0.15, tau_pi=0.30) == "block"


def test_gate_decision_blocks_when_both_exceed_thresholds():
    profile = RiskProfile(f=1, d_I=0.5, pi=0.5)
    assert gate_decision(profile, tau_d=0.15, tau_pi=0.30) == "block"


def test_gate_decision_conjunction_requires_both():
    profile_d_only = RiskProfile(f=1, d_I=0.5, pi=0.1)
    profile_pi_only = RiskProfile(f=1, d_I=0.1, pi=0.5)
    profile_both = RiskProfile(f=1, d_I=0.5, pi=0.5)
    assert gate_decision(profile_d_only, tau_d=0.15, tau_pi=0.30, use_conjunction=True) == "approve"
    assert gate_decision(profile_pi_only, tau_d=0.15, tau_pi=0.30, use_conjunction=True) == "approve"
    assert gate_decision(profile_both, tau_d=0.15, tau_pi=0.30, use_conjunction=True) == "block"


def test_gate_decision_f_zero_always_approves_regardless_of_thresholds():
    profile = RiskProfile(f=0, d_I=0.99, pi=0.99)
    assert gate_decision(profile, tau_d=0.0, tau_pi=0.0) == "approve"


# ── GateDecision dataclass ───────────────────────────────────────────────────

def test_gate_decision_has_no_regime_or_rubric_fields():
    d = GateDecision(step_index=0, decision="approve", profile=RiskProfile(f=0, d_I=0.0, pi=0.0))
    assert not hasattr(d, "regime")
    assert not hasattr(d, "rubric_verdicts")
    assert not hasattr(d, "blocking_reason")


# ── gate_step integration tests ─────────────────────────────────────────────

def test_pure_read_only_trajectory_approves_all_steps():
    trajectory = make_trajectory([
        make_step("noop()", ""),
        make_step("goto('https://example.com')", ""),
    ])
    config = Config()
    for idx in range(len(trajectory.steps)):
        decision = gate_step(trajectory, idx, config)
        assert decision.decision == "approve"
        assert decision.profile.f == 0


def test_multi_L3_trajectory_blocks():
    trajectory = make_trajectory([
        make_step("click('100')", "[100] role='button' name='Delete'"),
        make_step("click('200')", "[200] role='button' name='Submit'"),
    ])
    config = Config()
    second = gate_step(trajectory, 1, config)
    assert second.decision == "block"
    assert second.profile.f == 1
    assert second.profile.d_I == 1.0


def test_single_L3_in_many_steps_blocks_with_tight_tau():
    trajectory = make_trajectory([
        make_step("fill('100', 'Email')", "[100] role='textbox' name='Email'"),
        make_step("fill('101', 'Address')", "[101] role='textbox' name='Address'"),
        make_step("fill('102', 'Phone')", "[102] role='textbox' name='Phone'"),
        make_step("fill('103', 'City')", "[103] role='textbox' name='City'"),
        make_step("noop()"),
        make_step("click('200')", "[200] role='button' name='Submit'"),
    ])
    config = Config(tau_d=0.02, tau_pi=0.02)
    decision = gate_step(trajectory, 5, config)
    assert decision.decision == "block"


def test_gate_approves_all_when_thresholds_disabled():
    trajectory = make_trajectory([
        make_step("click('100')", "[100] role='button' name='Delete'"),
        make_step("click('200')", "[200] role='button' name='Submit'"),
    ])
    config = Config(tau_d=99.0, tau_pi=99.0)
    for idx in range(len(trajectory.steps)):
        decision = gate_step(trajectory, idx, config)
        assert decision.decision == "approve"
