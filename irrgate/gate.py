from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from irrgate.actions import Action
from irrgate.config import Config
from irrgate.profile import RiskProfile, compute_risk_profile
from irrgate.taxonomy import Level


@dataclass
class GateDecision:
    step_index: int
    decision: Literal["approve", "block"]
    profile: RiskProfile


def gate_decision(profile: RiskProfile, tau_d: float, tau_pi: int) -> Literal["approve", "block"]:
    """Return 'block' if `profile` crosses either threshold, else 'approve'."""
    if profile.f == 0:
        return "approve"
    return "block" if (profile.d_I >= tau_d or profile.irr_pos >= tau_pi) else "approve"


def make_gate_decision(
    actions: list[Action],
    levels: list[Level],
    config: Config,
) -> GateDecision:
    step_index = len(actions) - 1
    profile = compute_risk_profile(levels, actions)
    decision = gate_decision(profile, tau_d=config.tau_d, tau_pi=config.tau_pi)
    return GateDecision(step_index=step_index, decision=decision, profile=profile)
