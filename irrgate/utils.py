"""Small shared utilities used by scripts.

Keep helpers minimal and well-tested to avoid duplication across scripts.
"""
from __future__ import annotations

import math
from typing import List, Tuple


def percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile (linear interpolation) of a list.

    Returns 0.0 for empty input to preserve previous callers' behaviour.
    """
    if not values:
        return 0.0
    sv = sorted(values)
    idx = p / 100.0 * (len(sv) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sv) - 1)
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95%% confidence interval for a binomial proportion.

    Matches the previous implementations used across scripts.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)
