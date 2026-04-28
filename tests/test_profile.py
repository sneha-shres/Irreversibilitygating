from __future__ import annotations

from irrgate.actions import Action
from irrgate.profile import RiskProfile, compute_risk_profile
from irrgate.taxonomy import Level


def make_action(target_bid: str | None, fill_text: str | None, axtree: str, page_url: str = "") -> Action:
    return Action(
        action_type="click" if target_bid else "noop",
        raw="",
        target_bid=target_bid,
        fill_text=fill_text,
        target_url=None,
        target_element_text=None,
        page_url=page_url,
        reasoning="",
        step_index=0,
    )


def test_pure_read_only_plan():
    actions = [make_action(None, None, "") for _ in range(3)]
    profile = compute_risk_profile([Level.L0, Level.L0, Level.L0], actions, ["", "", ""])

    assert profile == RiskProfile(f=0, d_I=0.0, pi=0.0)


def test_single_L3_at_start_has_high_pi():
    actions = [make_action("1976", None, "")]
    profile = compute_risk_profile([Level.L3], actions, [""])

    assert profile.f == 1
    assert profile.d_I == 1.0
    assert profile.pi == 1.0


def test_single_L3_at_end_after_exploration_has_lower_pi():
    actions = [
        make_action("100", None, "[100] role='textbox' name='Email'"),
        make_action("1976", None, "[1976] role='button' name='Submit'"),
    ]
    profile = compute_risk_profile([Level.L1, Level.L3], actions, ["[100] role='textbox' name='Email'", "[1976] role='button' name='Submit'"])

    assert profile.f == 1
    assert profile.d_I == 0.5
    assert 0.0 < profile.pi < 1.0


def test_L3_with_fully_grounded_args_has_zero_pi():
    actions = [
        make_action("100", None, "[100] role='textbox' placeholder='Email'"),
        make_action("100", None, "[100] role='button' name='Submit'"),
    ]
    profile = compute_risk_profile([Level.L1, Level.L3], actions, ["[100] role='textbox' placeholder='Email'", "[100] role='button' name='Submit'"])

    assert profile.f == 1
    assert profile.pi == 0.0


def test_mixed_L2_L3_alpha_reflects_weights():
    actions = [
        make_action("200", None, "[200] role='button' name='Like'"),
        make_action("300", None, "[300] role='button' name='Delete'"),
        make_action("400", None, "[400] role='button' name='Edit'"),
    ]
    profile = compute_risk_profile([Level.L2, Level.L3, Level.L2], actions, ["[200] role='button' name='Like'", "[300] role='button' name='Delete'", "[400] role='button' name='Edit'"])

    assert profile.f == 1
    assert abs(profile.d_I - (0.5 + 1.0 + 0.5) / 3) < 1e-8
    assert 0.0 <= profile.pi <= 1.0
