from __future__ import annotations

import pytest

from irrgate.actions import Action
from irrgate.profile import RiskProfile, compute_risk_profile
from irrgate.taxonomy import Level


def make_action(
    target_bid: str | None = None,
    fill_text: str | None = None,
    page_url: str = "",
) -> Action:
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
    actions = [make_action(page_url=f"https://example.com/{i}") for i in range(3)]
    profile = compute_risk_profile([Level.L0, Level.L0, Level.L0], actions)

    assert profile == RiskProfile(f=0, d_I=0.0, irr_pos=0)  # L0 contributes 0 severity


def test_l3_at_start_on_single_page():
    # L3 at step 0 on page "p1" → irr_pos = 1 (the page where L3 fires is counted)
    actions = [make_action("1976", None, page_url="https://example.com/checkout/")]
    profile = compute_risk_profile([Level.L3], actions)

    assert profile.f == 1
    assert profile.d_I == pytest.approx(1.0)   # 1 L3 × 1.0
    assert profile.irr_pos == 1


def test_l3_after_multi_page_exploration():
    # L0 on p1, L0 on p2, L3 on p3 → irr_pos = 3
    actions = [
        make_action(page_url="https://example.com/a"),
        make_action(page_url="https://example.com/b"),
        make_action("1976", None, page_url="https://example.com/c"),
    ]
    profile = compute_risk_profile([Level.L0, Level.L0, Level.L3], actions)

    assert profile.f == 1
    assert profile.irr_pos == 3


def test_l3_on_same_page_as_prior_steps():
    # L0 and L3 both on "p1" → only 1 distinct page
    actions = [
        make_action(page_url="https://example.com/order"),
        make_action("100", None, page_url="https://example.com/order"),
    ]
    profile = compute_risk_profile([Level.L0, Level.L3], actions)

    assert profile.f == 1
    assert profile.irr_pos == 1


def test_irr_pos_zero_when_no_se():
    # All L0/L1 → f=0 → irr_pos is always 0
    actions = [make_action(page_url=f"https://example.com/{i}") for i in range(5)]
    profile = compute_risk_profile([Level.L0, Level.L1, Level.L0, Level.L1, Level.L0], actions)

    assert profile.f == 0
    assert profile.irr_pos == 0


def test_irr_pos_expands_up_to_last_se():
    # Pages after the first L2/L3 but before the last L2/L3 ARE included
    actions = [
        make_action(page_url="https://example.com/p1"),
        make_action("100", None, page_url="https://example.com/p2"),  # first L2
        make_action(page_url="https://example.com/p3"),               # between SEs
        make_action("99", None, page_url="https://example.com/p4"),   # last L3 — anchor
    ]
    profile = compute_risk_profile([Level.L0, Level.L2, Level.L0, Level.L3], actions)

    assert profile.f == 1
    assert profile.irr_pos == 4   # p1, p2, p3, p4 — all up to last L2/L3


def test_mixed_L2_L3_alpha_reflects_weights():
    actions = [
        make_action("200", None, page_url="https://example.com/p1"),
        make_action("300", None, page_url="https://example.com/p2"),
        make_action("400", None, page_url="https://example.com/p3"),
    ]
    profile = compute_risk_profile([Level.L2, Level.L3, Level.L2], actions)

    assert profile.f == 1
    assert abs(profile.d_I - (0.5 + 1.0 + 0.5)) < 1e-8   # absolute sum = 2.0
    assert profile.irr_pos == 3   # p1, p2, p3 — up to last L2 at step 2
