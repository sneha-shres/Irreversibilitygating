from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from irrgate.config import ALPHA
from irrgate.taxonomy import Level, severity_weight

_LABEL_FIELD_PATTERN = re.compile(
    r"(?:name|label|text|placeholder|content)\s*=\s*'((?:[^'\\]|\\.)*)'",
    re.IGNORECASE,
)


@dataclass
class RiskProfile:
    f: int
    d_I: float
    pi: float


def _extract_labels_from_axtree(axtree: str) -> set[str]:
    if not axtree:
        return set()

    labels: set[str] = set()
    for match in _LABEL_FIELD_PATTERN.finditer(axtree):
        labels.add(match.group(1).strip().lower())
    return labels


def _target_bid_seen_in_prior_axtrees(bid: str, prior_axtrees: Iterable[str]) -> bool:
    if not bid:
        return False
    pattern = re.compile(rf"\[{re.escape(bid)}\]")
    return any(pattern.search(axtree or "") for axtree in prior_axtrees)


def _fill_text_seen_in_prior_axtrees(fill_text: str, prior_axtrees: Iterable[str]) -> bool:
    if not fill_text:
        return False
    needle = fill_text.strip().lower()
    if not needle:
        return False
    for axtree in prior_axtrees:
        labels = _extract_labels_from_axtree(axtree or "")
        if any(needle in label for label in labels):
            return True
    return False


def target_bid_traceable_to_prior_axtrees(bid: str, prior_axtrees: Iterable[str]) -> bool:
    return _target_bid_seen_in_prior_axtrees(bid, prior_axtrees)


def fill_text_traceable_to_prior_axtrees(fill_text: str, prior_axtrees: Iterable[str]) -> bool:
    return _fill_text_seen_in_prior_axtrees(fill_text, prior_axtrees)


def compute_risk_profile(levels: list[Level], actions: list[object], axtrees: list[str]) -> RiskProfile:
    if len(actions) != len(axtrees):
        raise ValueError("actions and axtrees must have the same length")

    severity_values = [severity_weight(level, ALPHA) for level in levels]
    f = int(any(value > 0 for value in severity_values))
    d_I = sum(severity_values) / len(severity_values) if severity_values else 0.0

    distinct_bids = [action.target_bid for action in actions if getattr(action, "target_bid", None)]
    distinct_bids_in_full_plan = len(set(distinct_bids))

    weighted_residual = 0.0
    total_weight = sum(severity_values)

    prior_axtrees: list[str] = []
    seen_bids: set[str] = set()

    for index, (action, severity, axtree) in enumerate(zip(actions, severity_values, axtrees)):
        target_bid = getattr(action, "target_bid", None)
        fill_text = getattr(action, "fill_text", None)

        if distinct_bids_in_full_plan > 0 and distinct_bids_in_full_plan != 0:
            bid_term = len(seen_bids) / distinct_bids_in_full_plan
        else:
            bid_term = 1.0

        args_total = int(target_bid is not None) + int(fill_text is not None)
        if args_total == 0:
            trace_term = 1.0
        else:
            trace_count = 0
            if target_bid is not None and _target_bid_seen_in_prior_axtrees(target_bid, prior_axtrees):
                trace_count += 1
            if fill_text is not None and _fill_text_seen_in_prior_axtrees(fill_text, prior_axtrees):
                trace_count += 1
            trace_term = trace_count / args_total

        u_i = 0.5 * bid_term + 0.5 * trace_term
        weighted_residual += severity * (1.0 - u_i)

        if target_bid is not None:
            seen_bids.add(target_bid)
        prior_axtrees.append(axtree)

    pi = weighted_residual / total_weight if total_weight > 0 else 0.0
    return RiskProfile(f=f, d_I=d_I, pi=pi)
