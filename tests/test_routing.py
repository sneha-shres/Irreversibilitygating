from __future__ import annotations

from irrgate.profile import RiskProfile
from irrgate.routing import Regime, route


def test_route_bypass_with_no_nonzero_severity():
    profile = RiskProfile(f=0, d_I=0.0, pi=0.0)
    assert route(profile) == Regime.BYPASS


def test_route_low_when_both_below_threshold():
    profile = RiskProfile(f=1, d_I=0.1, pi=0.1)
    assert route(profile, tau_d=0.15, tau_pi=0.30) == Regime.LOW


def test_route_high_when_both_above_threshold():
    profile = RiskProfile(f=1, d_I=0.2, pi=0.4)
    assert route(profile, tau_d=0.15, tau_pi=0.30) == Regime.HIGH


def test_route_medium_when_one_above_and_one_below():
    profile = RiskProfile(f=1, d_I=0.2, pi=0.2)
    assert route(profile, tau_d=0.15, tau_pi=0.30) == Regime.MEDIUM
