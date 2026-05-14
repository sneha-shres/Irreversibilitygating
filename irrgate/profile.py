from __future__ import annotations

from dataclasses import dataclass

from irrgate.actions import Action
from irrgate.config import ALPHA
from irrgate.taxonomy import Level, severity_weight


@dataclass
class RiskProfile:
    f: int
    d_I: float
    pi: float


def compute_risk_profile(levels: list[Level], actions: list[Action]) -> RiskProfile:
    severity_values = [severity_weight(level, ALPHA) for level in levels]
    f = int(any(value > 0 for value in severity_values))
    d_I = sum(severity_values) / len(severity_values) if severity_values else 0.0

    distinct_bids = [action.target_bid for action in actions if action.target_bid]
    distinct_bids_in_full_plan = len(set(distinct_bids))

    weighted_residual = 0.0
    total_weight = sum(severity_values)

    seen_bids: set[str] = set()

    for action, severity in zip(actions, severity_values):
        if distinct_bids_in_full_plan > 0:
            bid_term = len(seen_bids) / distinct_bids_in_full_plan
        else:
            bid_term = 1.0

        weighted_residual += severity * (1.0 - bid_term)

        if action.target_bid is not None:
            seen_bids.add(action.target_bid)

    pi = weighted_residual / total_weight if total_weight > 0 else 0.0
    return RiskProfile(f=f, d_I=d_I, pi=pi)
