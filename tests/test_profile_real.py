"""Profile + routing tests against real cached trajectories.

Four trajectories chosen for distinct structural properties:
  1. knowledge-base-search/llama  — all L0/L1, 7 steps  (BYPASS gap)
  2. webarena.666/Qwen            — single L3 at step 0, 31 steps
  3. create-problem/gpt-4o        — dense alternating L1/L3, 31 steps
  4. webarena.788/gpt-4o          — 3 steps all L0 (shortest BYPASS)

Plus three isolated edge-case unit tests for the π denominator.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from irrgate.actions import Action
from irrgate.data.loader import load_trajectory
from irrgate.profile import RiskProfile, compute_risk_profile
from irrgate.routing import Regime, route
from irrgate.taxonomy import Level

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CACHE: pd.DataFrame | None = None
_TRAJ_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def _cache() -> pd.DataFrame:
    global _CACHE
    if _CACHE is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "classification_cache.parquet")
        _CACHE = pd.read_parquet(path)
    return _CACHE


def _find_traj_file(task_id: str, model_frag: str) -> str:
    cleaned = os.path.join(_TRAJ_DIR, "cleaned")
    for root, _dirs, files in os.walk(cleaned):
        if f"{task_id}.json" in files and model_frag in root:
            return os.path.join(root, f"{task_id}.json")
    raise FileNotFoundError(f"{task_id} / {model_frag}")


def _build_prefix(task_id: str, model_frag: str, traj_id: str):
    """Return (actions, levels, axtrees) for every step of the trajectory."""
    path = _find_traj_file(task_id, model_frag)
    traj = load_trajectory(path)
    rows = _cache()[_cache()["trajectory_id"] == traj_id].sort_values("step_index")
    cached_levels = [Level(v) for v in rows["final_level"].tolist()]
    assert len(cached_levels) == len(traj.steps), (
        f"Cache has {len(cached_levels)} rows but trajectory has {len(traj.steps)} steps"
    )
    actions = [Action.from_step(s, step_index=i) for i, s in enumerate(traj.steps)]
    axtrees = [str(s.get("axtree", "")) for s in traj.steps]
    return actions, cached_levels, axtrees


def _profiles_at_all_steps(actions, levels, axtrees):
    """Compute cumulative profile at every step index."""
    profiles = []
    for k in range(len(actions)):
        p = compute_risk_profile(levels[: k + 1], actions[: k + 1], axtrees[: k + 1])
        profiles.append(p)
    return profiles


# ---------------------------------------------------------------------------
# π denominator edge-case units (no I/O)
# ---------------------------------------------------------------------------

def _make_action(
    action_type: str = "click",
    target_bid: str | None = None,
    fill_text: str | None = None,
    page_url: str = "",
) -> Action:
    return Action(
        action_type=action_type,
        raw="",
        target_bid=target_bid,
        fill_text=fill_text,
        target_url=None,
        target_element_text=None,
        page_url=page_url,
        reasoning="",
        step_index=0,
    )


def test_pi_zero_when_W_is_zero():
    """All L0/L1 → total_weight=0 → π must be exactly 0."""
    actions = [_make_action(target_bid=str(i)) for i in range(5)]
    p = compute_risk_profile([Level.L0] * 5, actions, [""] * 5)
    assert p.f == 0
    assert p.d_I == 0.0
    assert p.pi == 0.0


def test_pi_args_total_zero_uses_trace_fallback():
    """L3 with no bid and no fill_text → args_total=0 → trace_term=1.0 fallback.

    distinct_bids is also 0 (no bid in any step) → bid_term=1.0 fallback.
    u_i = 1.0 → weighted_residual = 1.0*(1-1.0) = 0 → π = 0.
    """
    a = _make_action(action_type="None", target_bid=None, fill_text=None,
                     page_url="https://example.com/checkout/")
    p = compute_risk_profile([Level.L3], [a], [""])
    assert p.f == 1
    assert p.d_I == pytest.approx(1.0)
    assert p.pi == pytest.approx(0.0)   # fully covered by the two fallbacks


def test_pi_fill_only_L3_untraced():
    """L3 with fill_text but no bid, fill not seen in prior axtrees.

    args_total=1, trace_count=0 → trace_term=0.
    distinct_bids=0 → bid_term=1.0 fallback.
    u_i = 0.5 → weighted_residual = 0.5 → π = 0.5.
    """
    a = _make_action(action_type="select_option", target_bid=None, fill_text="Delete")
    p = compute_risk_profile([Level.L3], [a], [""])
    assert p.f == 1
    assert p.d_I == pytest.approx(1.0)
    assert p.pi == pytest.approx(0.5)


def test_pi_first_step_L3_no_prior_axtrees():
    """L3 at step 0 (empty prior axtrees): bid unseen, trace_term=0, bid_term=0 → u=0 → π=1."""
    a = _make_action(target_bid="1976")
    p = compute_risk_profile([Level.L3], [a], [""])
    assert p.f == 1
    assert p.d_I == pytest.approx(1.0)
    assert p.pi == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Real trajectory 1: knowledge-base-search / Llama — all L0/L1 (BYPASS gap)
# ---------------------------------------------------------------------------

BYPASS_TRAJ = "workarena.servicenow.knowledge-base-search::GenericAgent-meta-llama_Llama-3.3-70B-Instruct"


@pytest.fixture(scope="module")
def bypass_profiles():
    actions, levels, axtrees = _build_prefix(
        "workarena.servicenow.knowledge-base-search",
        "Llama-3.3-70B-Instruct",
        BYPASS_TRAJ,
    )
    return _profiles_at_all_steps(actions, levels, axtrees), levels


def test_bypass_traj_f_always_zero(bypass_profiles):
    profiles, _ = bypass_profiles
    assert all(p.f == 0 for p in profiles), "Pure L0/L1 trajectory must have f=0 at every step"


def test_bypass_traj_d_I_always_zero(bypass_profiles):
    profiles, _ = bypass_profiles
    assert all(p.d_I == pytest.approx(0.0) for p in profiles)


def test_bypass_traj_pi_always_zero(bypass_profiles):
    profiles, _ = bypass_profiles
    assert all(p.pi == pytest.approx(0.0) for p in profiles)


def test_bypass_traj_routes_to_bypass(bypass_profiles):
    profiles, _ = bypass_profiles
    for p in profiles:
        assert route(p) == Regime.BYPASS


# ---------------------------------------------------------------------------
# Real trajectory 2: webarena.666 / Qwen — single L3 at step 0
# ---------------------------------------------------------------------------

SINGLE_L3_TRAJ = "webarena.666::GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct"


@pytest.fixture(scope="module")
def single_l3_profiles():
    actions, levels, axtrees = _build_prefix(
        "webarena.666", "Qwen_Qwen2.5-VL-72B-Instruct", SINGLE_L3_TRAJ
    )
    return _profiles_at_all_steps(actions, levels, axtrees), levels


def test_single_l3_step0_profile(single_l3_profiles):
    """Step 0 is L3 with no prior axtrees → d_I=1, π=1."""
    profiles, levels = single_l3_profiles
    assert levels[0] == Level.L3
    p0 = profiles[0]
    assert p0.f == 1
    assert p0.d_I == pytest.approx(1.0)
    assert p0.pi == pytest.approx(1.0)


def test_single_l3_d_I_dilutes_over_time(single_l3_profiles):
    """After 30 all-L1 steps, d_I drops to ~0.032 (1 L3 / 31 steps × severity 1)."""
    profiles, _ = single_l3_profiles
    last = profiles[-1]
    assert last.f == 1
    assert last.d_I == pytest.approx(1.0 / 31, rel=1e-3)


def test_single_l3_pi_stays_pinned_at_one(single_l3_profiles):
    """π stays at 1.0 after step 0: the L3 was ungrounded and R stays ungrounded forever."""
    profiles, _ = single_l3_profiles
    # Every step after the first has the same weighted_residual / total_weight because
    # L1 steps contribute severity=0 and don't change the numerator or denominator.
    for p in profiles:
        assert p.pi == pytest.approx(1.0), f"π drifted: {p.pi}"


def test_single_l3_d_I_monotone_decreasing(single_l3_profiles):
    """d_I can only decrease after the first step if subsequent actions are L0/L1."""
    profiles, _ = single_l3_profiles
    d_I_vals = [p.d_I for p in profiles]
    for prev, curr in zip(d_I_vals, d_I_vals[1:]):
        assert curr <= prev + 1e-9, f"d_I increased: {prev} → {curr}"


def test_single_l3_routes_to_low_before_dilution(single_l3_profiles):
    """At step 0, d_I=1 and π=1 → GATED regime at default thresholds."""
    profiles, _ = single_l3_profiles
    assert route(profiles[0], tau_d=0.15, tau_pi=0.30) == Regime.GATED


def test_single_l3_routes_to_low_after_dilution(single_l3_profiles):
    """At step 30, d_I≈0.032 < τ_d=0.15 but π=1.0 ≥ τ_π=0.30 → GATED."""
    profiles, _ = single_l3_profiles
    assert route(profiles[-1], tau_d=0.15, tau_pi=0.30) == Regime.GATED


# ---------------------------------------------------------------------------
# Real trajectory 3: create-problem / gpt-4o — dense alternating L1/L3
# ---------------------------------------------------------------------------

DENSE_TRAJ = "workarena.servicenow.create-problem::GenericAgent-gpt-4o-2024-11-20"


@pytest.fixture(scope="module")
def dense_profiles():
    actions, levels, axtrees = _build_prefix(
        "workarena.servicenow.create-problem", "gpt-4o-2024-11-20", DENSE_TRAJ
    )
    return _profiles_at_all_steps(actions, levels, axtrees), levels


def test_dense_traj_f_zero_before_first_L3(dense_profiles):
    """Steps 0-2 are L1 → f=0 until the first L3 at step 3."""
    profiles, levels = dense_profiles
    for k in range(3):
        assert profiles[k].f == 0, f"Expected f=0 at step {k}, got {profiles[k].f}"


def test_dense_traj_f_one_from_step3(dense_profiles):
    profiles, levels = dense_profiles
    for k in range(3, len(profiles)):
        assert profiles[k].f == 1, f"Expected f=1 at step {k}"


def test_dense_traj_step3_d_I(dense_profiles):
    """At step 3 (first L3): d_I = severity(L3)/4 = 1.0/4 = 0.25."""
    profiles, _ = dense_profiles
    assert profiles[3].d_I == pytest.approx(0.25, rel=1e-3)


def test_dense_traj_step3_pi(dense_profiles):
    """At step 3 (first L3): verified value 0.625 from live run."""
    profiles, _ = dense_profiles
    assert profiles[3].pi == pytest.approx(0.625, rel=1e-3)


def test_dense_traj_final_d_I(dense_profiles):
    """At step 30: d_I ≈ 0.452 (14 L3 steps × 1.0 + 17 L1 × 0 / 31 steps)."""
    profiles, _ = dense_profiles
    assert profiles[-1].d_I == pytest.approx(14 / 31, rel=1e-3)


def test_dense_traj_invariants(dense_profiles):
    """Invariants must hold at every step."""
    profiles, _ = dense_profiles
    for k, p in enumerate(profiles):
        assert p.f in (0, 1), f"f not 0/1 at step {k}"
        assert 0.0 <= p.d_I <= 1.0 + 1e-9, f"d_I out of [0,1] at step {k}: {p.d_I}"
        assert 0.0 <= p.pi <= 1.0 + 1e-9, f"π out of [0,1] at step {k}: {p.pi}"
        if p.f == 0:
            assert p.d_I == pytest.approx(0.0)
            assert p.pi == pytest.approx(0.0)
