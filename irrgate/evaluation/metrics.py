from __future__ import annotations

from irrgate.evaluation.runner import TrajectoryResult


def compute_recall(positive_results: list[TrajectoryResult]) -> float:
    """Fraction of positives where the gate blocked (any blocking step)."""
    if not positive_results:
        return 0.0
    return sum(1 for r in positive_results if r.first_blocking_step is not None) / len(positive_results)


def compute_false_positive_rate(negative_results: list[TrajectoryResult]) -> float:
    """Fraction of negatives where the gate blocked at least once."""
    if not negative_results:
        return 0.0
    return sum(1 for r in negative_results if r.first_blocking_step is not None) / len(negative_results)
