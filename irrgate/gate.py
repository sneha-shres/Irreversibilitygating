from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from irrgate.config import Config
from irrgate.data.loader import Trajectory
from irrgate.actions import Action
from irrgate.classifier import classify
from irrgate.profile import RiskProfile, compute_risk_profile
from irrgate.routing import Regime, route
from irrgate.rubric import evaluate_rubric


@dataclass
class GateDecision:
    step_index: int
    decision: Literal["approve", "block"]
    regime: Regime
    profile: RiskProfile
    rubric_verdicts: dict[str, bool] | None
    blocking_reason: str | None


def gate_step(trajectory: Trajectory, step_index: int, config: Config) -> GateDecision:
    if step_index < 0 or step_index >= len(trajectory.steps):
        raise IndexError("step_index is out of range for the trajectory")

    actions: list[Action] = []
    levels: list[int] = []
    axtrees: list[str] = []
    prior_axtrees: list[str] = []

    for idx, step in enumerate(trajectory.steps[: step_index + 1]):
        action = Action.from_step(step, step_index=idx)
        level = classify(action, prior_axtrees=prior_axtrees, mode=config.rubric_mode)
        actions.append(action)
        levels.append(level)
        axtree = str(step.get("axtree", ""))
        axtrees.append(axtree)
        prior_axtrees.append(axtree)

    profile = compute_risk_profile(levels, actions, axtrees)
    regime = route(profile, tau_d=config.tau_d, tau_pi=config.tau_pi)
    rubric_verdicts = evaluate_rubric(actions, levels, axtrees, regime, mode=config.rubric_mode)

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
    else:
        if failed_items:
            decision = "block"
            blocking_reason = f"{regime.value.capitalize()} regime rubric failures: {', '.join(failed_items)}"
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
