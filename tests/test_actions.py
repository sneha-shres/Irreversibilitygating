from __future__ import annotations

import pytest

from irrgate.actions import Action, extract_element_text_from_axtree, parse_action_string


def test_parse_click_action():
    action_type, target_bid, fill_text, target_url = parse_action_string("click('1976')")

    assert action_type == "click"
    assert target_bid == "1976"
    assert fill_text is None
    assert target_url is None


def test_parse_fill_action():
    action_type, target_bid, fill_text, target_url = parse_action_string("fill('1497', 'message body')")

    assert action_type == "fill"
    assert target_bid == "1497"
    assert fill_text == "message body"
    assert target_url is None


def test_parse_goto_action():
    action_type, target_bid, fill_text, target_url = parse_action_string("goto('https://example.com')")

    assert action_type == "goto"
    assert target_url == "https://example.com"
    assert target_bid is None
    assert fill_text is None


def test_parse_send_msg_to_user_action():
    action_type, target_bid, fill_text, target_url = parse_action_string("send_msg_to_user('Hello there')")

    assert action_type == "send_msg_to_user"
    assert fill_text == "Hello there"
    assert target_bid is None
    assert target_url is None


def test_parse_select_option_action():
    action_type, target_bid, fill_text, target_url = parse_action_string("select_option('123', 'Option A')")

    assert action_type == "select_option"
    assert target_bid == "123"
    assert fill_text == "Option A"
    assert target_url is None


def test_parse_terminal_and_misc_actions():
    for raw in ["report_infeasible('reason')", "noop()", "scroll('down')", "tab_focus('input')"]:
        action_type, target_bid, fill_text, target_url = parse_action_string(raw)
        assert action_type in {"report_infeasible", "noop", "scroll", "tab_focus"}
        assert target_bid is None
        assert fill_text is None
        assert target_url is None


def test_extract_element_text_from_axtree_with_label():
    axtree = "[1976] role='button' name='Submit'\n[1977] role='textbox' name='Email'"
    assert extract_element_text_from_axtree(axtree, "1976") == "Submit"


def test_extract_element_text_from_axtree_missing_bid():
    axtree = "[1977] role='textbox' name='Email'"
    assert extract_element_text_from_axtree(axtree, "1976") is None


def test_extract_element_text_from_axtree_no_label():
    axtree = "[1976] role='button' aria-hidden='true'"
    assert extract_element_text_from_axtree(axtree, "1976") is None


def test_extract_element_text_with_multiple_matches():
    axtree = (
        "[1976] role='button' name='Submit'\n"
        "[1976] role='button' name='Confirm'"
    )
    assert extract_element_text_from_axtree(axtree, "1976") == "Submit"


def test_extract_element_text_handles_special_characters():
    axtree = "[42] role='button' name='Upload & Save (draft)'"
    assert extract_element_text_from_axtree(axtree, "42") == "Upload & Save (draft)"


def test_action_from_step_populates_target_element_text():
    step = {
        "action": "click('1976')",
        "axtree": "[1976] role='button' name='Submit'",
        "url": "https://example.com/contact",
        "reasoning": "Submit the form.",
        "bounding_boxes": [],
    }
    action = Action.from_step(step, step_index=0)

    assert action.action_type == "click"
    assert action.target_bid == "1976"
    assert action.target_element_text == "Submit"
    assert action.page_url == "https://example.com/contact"
    assert action.reasoning == "Submit the form."
    assert action.step_index == 0


def test_parse_action_with_escaped_quotes():
    action_type, target_bid, fill_text, target_url = parse_action_string("fill('1497', 'It\\'s complicated')")
    assert action_type == "fill"
    assert target_bid == "1497"
    assert fill_text == "It's complicated"


def test_parse_malformed_action_falls_back():
    action_type, target_bid, fill_text, target_url = parse_action_string("invalid-action")
    assert action_type == "invalid-action"
    assert target_bid is None
    assert fill_text is None
    assert target_url is None
