from __future__ import annotations

from enum import Enum

from irrgate.profile import RiskProfile


class Regime(Enum):
    BYPASS = "bypass"
    LOW = "low"
    GATED = "gated"


def route(profile: RiskProfile, tau_d: float = 0.15, tau_pi: float = 0.30) -> Regime:
    if profile.f == 0:
        return Regime.BYPASS
    if profile.d_I < tau_d and profile.pi < tau_pi:
        return Regime.LOW
    return Regime.GATED
