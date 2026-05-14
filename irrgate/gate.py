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


def gate_decision(
    profile: RiskProfile,
    tau_d: float,
    tau_pi: float,
    use_conjunction: bool = False,
) -> Literal["approve", "block"]:
    """Block iff f=1 and threshold condition is met.

    Disjunction (default, primary policy): d_I >= tau_d OR pi >= tau_pi
    Conjunction: d_I >= tau_d AND pi >= tau_pi
    """
    if profile.f == 0:
        return "approve"
    if use_conjunction:
        if profile.d_I >= tau_d and profile.pi >= tau_pi:
            return "block"
    else:
        if profile.d_I >= tau_d or profile.pi >= tau_pi:
            return "block"
    return "approve"


def make_gate_decision(
    actions: list[Action],
    levels: list[Level],
    config: Config,
) -> GateDecision:
    step_index = len(actions) - 1
    profile = compute_risk_profile(levels, actions)
    decision = gate_decision(
        profile,
        tau_d=config.tau_d,
        tau_pi=config.tau_pi,
        use_conjunction=config.use_conjunction,
    )
    return GateDecision(step_index=step_index, decision=decision, profile=profile)


