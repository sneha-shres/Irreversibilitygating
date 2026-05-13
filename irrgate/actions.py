from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

_ACTION_ARG_PATTERN = re.compile(r"^([a-z_]+)\((.*)\)$", re.IGNORECASE)
_SINGLE_QUOTED_PATTERN = re.compile(r"^'((?:[^'\\]|\\.)*)'$")
_DOUBLE_QUOTED_PATTERN = re.compile(r'^"((?:[^"\\]|\\.)*)"$')
_LABEL_FIELD_PATTERN = re.compile(
    r"(?:name|label|text|placeholder|content)\s*=\s*'((?:[^'\\]|\\.)*)'",
    re.IGNORECASE,
)
# BrowserGym AX-tree format: `[bid] role 'name', attr=val, ...`. The accessible
# name is the first single-quoted string after the role token; capture it.
_ROLE_NAME_PATTERN = re.compile(
    r"\[[^\]]+\]\s+\S+\s+'((?:[^'\\]|\\.)*)'",
)


@dataclass
class Action:
    action_type: str
    raw: str
    target_bid: Optional[str]
    fill_text: Optional[str]
    target_url: Optional[str]
    target_element_text: Optional[str]
    page_url: str
    reasoning: str
    step_index: int

    @classmethod
    def from_step(cls, step: dict[str, Any], step_index: int) -> "Action":
        raw = str(step.get("action", ""))
        action_type, target_bid, fill_text, target_url = parse_action_string(raw)
        axtree = step.get("axtree")
        target_element_text = (
            extract_element_text_from_axtree(str(axtree), target_bid)
            if target_bid and axtree is not None
            else None
        )

        return cls(
            action_type=action_type,
            raw=raw,
            target_bid=target_bid,
            fill_text=fill_text,
            target_url=target_url,
            target_element_text=target_element_text,
            page_url=str(step.get("url", "")),
            reasoning=str(step.get("reasoning", "")),
            step_index=step_index,
        )


def parse_action_string(raw_action: str) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    action_type = raw_action.strip()
    target_bid: Optional[str] = None
    fill_text: Optional[str] = None
    target_url: Optional[str] = None

    match = _ACTION_ARG_PATTERN.match(action_type)
    if not match:
        return action_type, None, None, None

    action_type = match.group(1).lower()
    args_text = match.group(2).strip()
    args = [arg.strip() for arg in split_args(args_text)] if args_text else []
    args = [unquote_string(arg) for arg in args if arg != ""]

    if action_type == "click" and len(args) >= 1:
        target_bid = args[0]
    elif action_type == "fill" and len(args) >= 1:
        target_bid = args[0]
        if len(args) >= 2:
            fill_text = args[1]
    elif action_type == "goto" and len(args) >= 1:
        target_url = args[0]
    elif action_type == "send_msg_to_user" and len(args) >= 1:
        fill_text = args[0]
    elif action_type == "select_option" and len(args) >= 1:
        target_bid = args[0]
        if len(args) >= 2:
            fill_text = args[1]
    elif action_type in {"report_infeasible", "noop", "scroll", "tab_focus"}:
        pass

    return action_type, target_bid, fill_text, target_url


def split_args(args_text: str) -> list[str]:
    args: list[str] = []
    current = []
    depth = 0
    in_quote = False
    escape = False

    for char in args_text:
        if escape:
            current.append(char)
            escape = False
            continue

        if char == "\\":
            current.append(char)
            escape = True
            continue

        if char in ("'", '"'):
            if in_quote is False:
                in_quote = char
            elif char == in_quote:
                in_quote = False
            current.append(char)
            continue

        if char == "," and in_quote is False and depth == 0:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
            continue

        if char in "([{" and in_quote is False:
            depth += 1
        elif char in ")]}" and in_quote is False and depth > 0:
            depth -= 1

        current.append(char)

    if current:
        arg = "".join(current).strip()
        if arg:
            args.append(arg)

    return args


def unquote_string(value: str) -> str:
    m = _SINGLE_QUOTED_PATTERN.fullmatch(value)
    if m:
        return m.group(1).replace("\\'", "'").replace('\\\\', "\\")
    m = _DOUBLE_QUOTED_PATTERN.fullmatch(value)
    if m:
        return m.group(1).replace('\\"', '"').replace('\\\\', "\\")
    return value


def extract_element_text_from_axtree(axtree: str, bid: str) -> Optional[str]:
    if not axtree or not bid:
        return None

    escaped_bid = re.escape(bid)
    bid_pattern = re.compile(rf"\[{escaped_bid}\]")

    for line in axtree.splitlines():
        if bid_pattern.search(line):
            label_match = _LABEL_FIELD_PATTERN.search(line)
            if label_match:
                return label_match.group(1)
            role_match = _ROLE_NAME_PATTERN.search(line)
            if role_match:
                return role_match.group(1)

    return None
