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
    n_l2_actions: int = 0         # count of L2 steps in the trajectory prefix
    n_distinct_l2_bids: int = 0   # distinct non-None BIDs among L2 steps
    n_l2_actions_pre_se: int = 0  # L2 steps at or before the last L3 step seen (all L2 if no L3 yet)


def compute_risk_profile(levels: list[Level], actions: list[Action]) -> RiskProfile:
    severity_values = [severity_weight(level, ALPHA) for level in levels]
    f = int(any(value > 0 for value in severity_values))
    d_I = sum(severity_values) / len(severity_values) if severity_values else 0.0

    distinct_bids = [action.target_bid for action in actions if action.target_bid]
    distinct_bids_in_full_plan = len(set(distinct_bids))

    weighted_residual = 0.0
    total_weight = sum(severity_values)

    seen_bids: set[str] = set()
    n_l2_actions = 0
    l2_bids: set[str] = set()
    last_l3_idx: int | None = None

    for i, (action, severity, level) in enumerate(zip(actions, severity_values, levels)):
        if distinct_bids_in_full_plan > 0:
            bid_term = len(seen_bids) / distinct_bids_in_full_plan
        else:
            bid_term = 1.0

        weighted_residual += severity * (1.0 - bid_term)

        if action.target_bid is not None:
            seen_bids.add(action.target_bid)

        if level == Level.L2:
            n_l2_actions += 1
            if action.target_bid is not None:
                l2_bids.add(action.target_bid)
        elif level == Level.L3:
            last_l3_idx = i

    # L2 steps at or before the last L3 step; all L2 steps if no L3 seen yet
    if last_l3_idx is not None:
        n_l2_actions_pre_se = sum(
            1 for i, lv in enumerate(levels) if lv == Level.L2 and i <= last_l3_idx
        )
    else:
        n_l2_actions_pre_se = n_l2_actions

    pi = weighted_residual / total_weight if total_weight > 0 else 0.0
    return RiskProfile(
        f=f,
        d_I=d_I,
        pi=pi,
        n_l2_actions=n_l2_actions,
        n_distinct_l2_bids=len(l2_bids),
        n_l2_actions_pre_se=n_l2_actions_pre_se,
    )
