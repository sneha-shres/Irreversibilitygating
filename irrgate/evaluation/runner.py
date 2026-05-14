from __future__ import annotations

from dataclasses import dataclass

from irrgate.actions import Action
from irrgate.classifier import classify
from irrgate.config import Config
from irrgate.data.loader import Trajectory
from irrgate.gate import GateDecision, make_gate_decision
from irrgate.taxonomy import Level


@dataclass
class TrajectoryResult:
    trajectory: Trajectory
    step_decisions: list[GateDecision]
    first_blocking_step: int | None
    reached_completion: bool


def evaluate_trajectory(trajectory: Trajectory, config: Config) -> TrajectoryResult:
    """Evaluate each step incrementally, classifying each action exactly once."""
    step_decisions: list[GateDecision] = []
    first_blocking_step: int | None = None

    actions: list[Action] = []
    levels: list[Level] = []

    for step_index, step in enumerate(trajectory.steps):
        action = Action.from_step(step, step_index=step_index)
        level = classify(action)

        actions.append(action)
        levels.append(level)

        decision = make_gate_decision(actions, levels, config)
        step_decisions.append(decision)

        if decision.decision == "block":
            first_blocking_step = step_index
            break

    reached_completion = first_blocking_step is None

    return TrajectoryResult(
        trajectory=trajectory,
        step_decisions=step_decisions,
        first_blocking_step=first_blocking_step,
        reached_completion=reached_completion,
    )
