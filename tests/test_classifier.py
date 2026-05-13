from __future__ import annotations

from unittest.mock import patch

from irrgate.actions import Action
from irrgate.classifier import classify, classify_stage1, classify_stage2
from irrgate.taxonomy import Level


def make_step(action: str, axtree: str = "", url: str = "https://example.com") -> dict:
    return {
        "action": action,
        "axtree": axtree,
        "url": url,
        "reasoning": "",
        "bounding_boxes": [],
    }


def test_stage1_goto_is_L0():
    action = Action.from_step(make_step("goto('https://example.com')"), 0)
    assert classify_stage1(action) == Level.L0


def test_stage1_send_msg_is_L0():
    action = Action.from_step(make_step("send_msg_to_user('Hi there')"), 0)
    assert classify_stage1(action) == Level.L0


def test_stage1_submit_element_is_L3():
    action = Action.from_step(
        make_step("click('1976')", "[1976] role='button' name='Submit'"),
        0,
    )
    assert classify_stage1(action) == Level.L3


def test_stage1_delete_element_is_L3():
    action = Action.from_step(
        make_step("click('1976')", "[1976] role='button' name='Delete'"),
        0,
    )
    assert classify_stage1(action) == Level.L3


def test_stage1_checkout_url_is_L3():
    action = Action.from_step(make_step("click('1976')", "", "https://example.com/checkout/cart"), 0)
    assert classify_stage1(action) == Level.L3


def test_stage1_like_element_is_L2():
    action = Action.from_step(
        make_step("click('1976')", "[1976] role='button' name='Like'"),
        0,
    )
    assert classify_stage1(action) == Level.L2


def test_stage1_fill_without_submit_is_L1():
    action = Action.from_step(
        make_step("fill('1497', 'hello')", "[1497] role='textbox' name='Message'"),
        0,
    )
    assert classify_stage1(action) == Level.L1


def test_stage1_click_cancel_is_L1():
    action = Action.from_step(
        make_step("click('1976')", "[1976] role='button' name='Cancel'"),
        0,
    )
    assert classify_stage1(action) == Level.L1


def test_stage1_ambiguous_click_returns_none():
    action = Action.from_step(
        make_step("click('1976')", "[1976] role='button' name='More info'"),
        0,
    )
    assert classify_stage1(action) is None


def test_classify_ambiguous_click_uses_stage2():
    action = Action.from_step(
        make_step("click('1976')", "[1976] role='button' name='More info'"),
        0,
    )
    with patch("irrgate.classifier._classify_stage2_gemini_cached", return_value=(Level.L1, None)):
        assert classify(action, prior_axtrees=[]) == Level.L1


def test_stage2_calls_gemini():
    action = Action.from_step(make_step("click('1976')"), 0)
    with patch("irrgate.classifier._classify_stage2_gemini_cached", return_value=(Level.L2, None)) as mock_gemini:
        result = classify_stage2(action, prior_axtrees=[])
        assert result == Level.L2
        mock_gemini.assert_called_once()


def test_stage1_malformed_action_with_empty_text_returns_none():
    action = Action.from_step(make_step("bad_action()", "", "https://example.com"), 0)
    assert classify_stage1(action) is None


def test_stage1_fill_with_empty_placeholder_escalates_to_stage2():
    # BrowserGym labels unlabeled grid cells as '(empty)'; stage1 must not
    # hard-code L1 so that stage-2 can inspect fill_text / reasoning.
    action = Action.from_step(
        make_step("fill('a582', '2029-04-15 08:25:08')", "[a582] gridcell '(empty)', visible"),
        5,
    )
    assert classify_stage1(action) is None


def test_fill_with_empty_placeholder_defers_to_stage2():
    action = Action.from_step(
        make_step("fill('a582', '2029-04-15 08:25:08')", "[a582] gridcell '(empty)', visible"),
        5,
    )
    with patch("irrgate.classifier._classify_stage2_gemini_cached", return_value=(Level.L1, None)):
        assert classify(action) == Level.L1
