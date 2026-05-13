from __future__ import annotations

from unittest.mock import patch

from irrgate.actions import Action
from irrgate.rubric import (
    evaluate_rubric,
    r1_target_bid_present,
    r2_args_traceable,
    r3_consent_precedes_L3,
    rubric_llm_check,
)
from irrgate.routing import Regime
from irrgate.taxonomy import Level


def make_action(action: str, axtree: str = "", url: str = "") -> Action:
    return Action(
        action_type=action.split("(", 1)[0],
        raw=action,
        target_bid=None,
        fill_text=None,
        target_url=None,
        target_element_text=None,
        page_url=url,
        reasoning="",
        step_index=0,
    )


def test_r1_true_when_bid_seen_prior():
    action = Action(
        action_type="click",
        raw="click('1976')",
        target_bid="1976",
        fill_text=None,
        target_url=None,
        target_element_text=None,
        page_url="",
        reasoning="",
        step_index=1,
    )
    assert r1_target_bid_present(action, ["[1976] role='button' name='Submit'"])


def test_r1_false_when_bid_not_seen_prior():
    action = Action(
        action_type="click",
        raw="click('1976')",
        target_bid="1976",
        fill_text=None,
        target_url=None,
        target_element_text=None,
        page_url="",
        reasoning="",
        step_index=1,
    )
    assert not r1_target_bid_present(action, ["[1977] role='button' name='Submit'"])


def test_r2_true_when_fill_text_traceable():
    action = Action(
        action_type="fill",
        raw="fill('1497', 'Email')",
        target_bid="1497",
        fill_text="Email",
        target_url=None,
        target_element_text=None,
        page_url="",
        reasoning="",
        step_index=1,
    )
    assert r2_args_traceable(action, ["[1497] role='textbox' name='Email'"])


def test_r2_false_when_fill_text_not_traceable():
    action = Action(
        action_type="fill",
        raw="fill('1497', 'Email')",
        target_bid="1497",
        fill_text="Email",
        target_url=None,
        target_element_text=None,
        page_url="",
        reasoning="",
        step_index=1,
    )
    assert not r2_args_traceable(action, ["[1497] role='textbox' name='Name'"])


def test_r3_true_when_consent_precedes_L3():
    actions = [
        Action(
            action_type="send_msg_to_user",
            raw="send_msg_to_user('Please confirm')",
            target_bid=None,
            fill_text="Please confirm",
            target_url=None,
            target_element_text=None,
            page_url="",
            reasoning="",
            step_index=0,
        ),
        Action(
            action_type="click",
            raw="click('1976')",
            target_bid="1976",
            fill_text=None,
            target_url=None,
            target_element_text=None,
            page_url="",
            reasoning="",
            step_index=1,
        ),
    ]
    assert r3_consent_precedes_L3(actions, [Level.L0, Level.L3])


def test_r3_false_without_consent():
    actions = [
        Action(
            action_type="click",
            raw="click('1976')",
            target_bid="1976",
            fill_text=None,
            target_url=None,
            target_element_text=None,
            page_url="",
            reasoning="",
            step_index=0,
        )
    ]
    assert not r3_consent_precedes_L3(actions, [Level.L3])


def test_rubric_llm_check_returns_r4_r5():
    actions = [make_action("click('1976')")]
    with patch("irrgate.rubric.rubric_llm_check_gemini", return_value={"R4": True, "R5": True}):
        results = rubric_llm_check(actions, 0, [""])
    assert results["R4"] is True
    assert results["R5"] is True


def test_evaluate_rubric_low_includes_r3():
    actions = [
        Action(
            action_type="send_msg_to_user",
            raw="send_msg_to_user('Confirm')",
            target_bid=None,
            fill_text="Confirm",
            target_url=None,
            target_element_text=None,
            page_url="",
            reasoning="",
            step_index=0,
        ),
        Action(
            action_type="click",
            raw="click('1976')",
            target_bid="1976",
            fill_text=None,
            target_url=None,
            target_element_text=None,
            page_url="",
            reasoning="",
            step_index=1,
        ),
    ]
    verdict = evaluate_rubric(
        actions,
        [Level.L0, Level.L3],
        ["[1976] role='button' name='Submit'", "[1976] role='button' name='Submit'"],
        Regime.LOW,
    )
    assert set(verdict.keys()) == {"R1", "R2", "R3"}
    assert verdict["R3"] is True


def test_evaluate_rubric_low_r3_fails_without_consent():
    actions = [
        Action(
            action_type="click",
            raw="click('1976')",
            target_bid="1976",
            fill_text=None,
            target_url=None,
            target_element_text=None,
            page_url="",
            reasoning="",
            step_index=0,
        ),
    ]
    verdict = evaluate_rubric(
        actions,
        [Level.L3],
        ["[1976] role='button' name='Submit'"],
        Regime.LOW,
    )
    assert verdict["R3"] is False
    assert "R4" not in verdict
    assert "R5" not in verdict


def test_evaluate_rubric_gated_requires_r1_r2_r3_r4_r5():
    actions = [
        Action(
            action_type="send_msg_to_user",
            raw="send_msg_to_user('Confirm')",
            target_bid=None,
            fill_text="Confirm",
            target_url=None,
            target_element_text=None,
            page_url="",
            reasoning="",
            step_index=0,
        ),
        Action(
            action_type="click",
            raw="click('1976')",
            target_bid="1976",
            fill_text=None,
            target_url=None,
            target_element_text=None,
            page_url="",
            reasoning="",
            step_index=1,
        ),
    ]
    with patch("irrgate.rubric.rubric_llm_check_gemini", return_value={"R4": True, "R5": True}):
        verdict = evaluate_rubric(
            actions,
            [Level.L0, Level.L3],
            ["[1976] role='button' name='Submit'", "[1976] role='button' name='Submit'"],
            Regime.GATED,
        )
    assert verdict["R1"]
    assert verdict["R2"]
    assert verdict["R3"]
    assert verdict["R4"]
    assert verdict["R5"]


def test_evaluate_rubric_gated_r3_fails_without_consent():
    actions = [
        Action(
            action_type="click",
            raw="click('1976')",
            target_bid="1976",
            fill_text=None,
            target_url=None,
            target_element_text=None,
            page_url="",
            reasoning="",
            step_index=0,
        ),
    ]
    with patch("irrgate.rubric.rubric_llm_check_gemini", return_value={"R4": True, "R5": True}):
        verdict = evaluate_rubric(
            actions,
            [Level.L3],
            ["[1976] role='button' name='Submit'"],
            Regime.GATED,
        )
    assert not verdict["R3"]
