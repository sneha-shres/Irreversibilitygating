from __future__ import annotations

from dataclasses import dataclass

from irrgate.actions import Action
from irrgate.config import ALPHA, BETA
from irrgate.taxonomy import Level, severity_weight


@dataclass
class RiskProfile:
    f: int        # irreversibility presence: 1 iff any L2/L3 action
    d_I: float    # irreversibility density: total (absolute) accumulated severity across all steps
    irr_pos: int  # irreversibility positional risk: distinct pages up to and including the last L2/L3 step


def compute_risk_profile(levels: list[Level], actions: list[Action]) -> RiskProfile:
    severity_values = [severity_weight(level, ALPHA, BETA) for level in levels]

    # Presence: L2 or L3 actions indicate side-effect potential
    f = int(any(level in {Level.L2, Level.L3} for level in levels))

    # Density: absolute (cumulative) severity — sum over all steps, not divided by n_steps.
    # Avoids the length-dilution problem where a single L3 in a long trajectory looks low-risk.
    d_I = sum(severity_values)

    # Positional risk: distinct pages visited up to and including the last L2/L3 step.
    # Snapshots the page set at each L2/L3 step; the final snapshot covers the widest
    # exploration window, capturing how much the agent traversed before its final risky act.
    pages_pre_last_se: set[str] = set()
    current_pages: set[str] = set()
    last_se_found = False
    for level, action in zip(levels, actions):
        if action.page_url:
            current_pages.add(action.page_url)
        if level in (Level.L2, Level.L3):
            last_se_found = True
            pages_pre_last_se = set(current_pages)
    irr_pos = len(pages_pre_last_se) if last_se_found else 0

    return RiskProfile(f=f, d_I=d_I, irr_pos=irr_pos)
