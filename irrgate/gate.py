from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from irrgate.actions import Action
from irrgate.classifier import classify
from irrgate.config import Config
from irrgate.data.loader import Trajectory
from irrgate.profile import RiskProfile, compute_risk_profile
from irrgate.routing import Regime, route
from irrgate.rubric import evaluate_rubric
from irrgate.taxonomy import Level


@dataclass
class GateDecision:
    step_index: int
    decision: Literal["approve", "block"]
    regime: Regime
    profile: RiskProfile
    rubric_verdicts: dict[str, bool] | None
    blocking_reason: str | None


def make_gate_decision(
    actions: list[Action],
    levels: list[Level],
    axtrees: list[str],
    config: Config,
) -> GateDecision:
    """Compute a gate decision given pre-built trajectory state up to the current step.

    `actions`, `levels`, and `axtrees` must all have the same length; the last element
    is the current step being evaluated.
    """
    step_index = len(actions) - 1
    profile = compute_risk_profile(levels, actions, axtrees)
    regime = route(profile, tau_d=config.tau_d, tau_pi=config.tau_pi)
    rubric_verdicts = evaluate_rubric(actions, levels, axtrees, regime)

    failed_items = [item for item, passed in rubric_verdicts.items() if not passed]
    blocking_reason = None

    if regime == Regime.BYPASS:
        decision = "approve"
    elif regime == Regime.LOW:
        if failed_items:
            decision = "block"
            blocking_reason = f"Low regime rubric failures: {', '.join(failed_items)}"
        else:
            decision = "approve"
    else:  # GATED
        if failed_items:
            decision = "block"
            blocking_reason = f"Gated regime rubric failures: {', '.join(failed_items)}"
        else:
            decision = "approve"

    return GateDecision(
        step_index=step_index,
        decision=decision,
        regime=regime,
        profile=profile,
        rubric_verdicts=rubric_verdicts,
        blocking_reason=blocking_reason,
    )


def gate_step(trajectory: Trajectory, step_index: int, config: Config) -> GateDecision:
    if step_index < 0 or step_index >= len(trajectory.steps):
        raise IndexError("step_index is out of range for the trajectory")

    actions: list[Action] = []
    levels: list[Level] = []
    axtrees: list[str] = []
    prior_axtrees: list[str] = []

    for idx, step in enumerate(trajectory.steps[: step_index + 1]):
        action = Action.from_step(step, step_index=idx)
        level = classify(action, prior_axtrees=prior_axtrees)
        actions.append(action)
        levels.append(level)
        axtree = str(step.get("axtree", ""))
        axtrees.append(axtree)
        prior_axtrees.append(axtree)

    return make_gate_decision(actions, levels, axtrees, config)
