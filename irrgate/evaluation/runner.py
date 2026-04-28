from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from irrgate.config import Config
from irrgate.data.loader import Trajectory
from irrgate.gate import GateDecision, gate_step


@dataclass
class TrajectoryResult:
    trajectory: Trajectory
    step_decisions: list[GateDecision]
    first_blocking_step: int | None
    reached_completion: bool


def evaluate_trajectory(trajectory: Trajectory, config: Config) -> TrajectoryResult:
    """For each step in the trajectory, run gate_step and collect decisions."""
    step_decisions: list[GateDecision] = []
    first_blocking_step: int | None = None

    for step_index in range(len(trajectory.steps)):
        decision = gate_step(trajectory, step_index, config)
        step_decisions.append(decision)

        if decision.decision == "block" and first_blocking_step is None:
            first_blocking_step = step_index

    # Reached completion if no blocking occurred
    reached_completion = first_blocking_step is None

    return TrajectoryResult(
        trajectory=trajectory,
        step_decisions=step_decisions,
        first_blocking_step=first_blocking_step,
        reached_completion=reached_completion,
    )