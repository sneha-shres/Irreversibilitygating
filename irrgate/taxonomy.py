from __future__ import annotations

from enum import IntEnum


class Level(IntEnum):
    L0 = 0  # read-only
    L1 = 1  # agent-reversible
    L2 = 2  # cost-reversible
    L3 = 3  # irreversible


def severity_weight(level: Level, alpha: float = 0.5, beta: float = 0.0) -> float:
    """Return the severity weight for a level.

    L0=0, L1=beta (default 0 for backward compat), L2=alpha, L3=1.0.
    """
    if level == Level.L0:
        return 0.0
    if level == Level.L1:
        return beta
    if level == Level.L2:
        return alpha
    if level == Level.L3:
        return 1.0
    raise ValueError(f"Unsupported level: {level}")
