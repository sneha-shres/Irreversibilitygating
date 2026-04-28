"""Evaluation utilities for IrrGate."""

from .analysis import produce_ablation_table, produce_results_table
from .metrics import compute_false_positive_rate, compute_per_task_metrics, compute_recall, per_task_aggregation
from .runner import TrajectoryResult, evaluate_trajectory

__all__ = [
    "TrajectoryResult",
    "evaluate_trajectory",
    "compute_recall",
    "compute_false_positive_rate",
    "compute_per_task_metrics",
    "per_task_aggregation",
    "produce_results_table",
    "produce_ablation_table",
]
