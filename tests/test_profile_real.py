"""Profile tests against real cached trajectories.

Four trajectories chosen for distinct structural properties:
  1. knowledge-base-search/llama  — mostly L0, one L1, 7 steps  (f=0 throughout)
  2. webarena.666/Qwen            — single L2 at step 0, 31 steps
  3. create-problem/gpt-4o        — alternating L1/L2, 31 steps
  4. webarena.788/gpt-4o          — 3 steps all L0

Plus irr_pos edge-case unit tests (no I/O).
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from irrgate.actions import Action
from irrgate.config import ALPHA, BETA
from irrgate.data.loader import load_trajectory
from irrgate.profile import RiskProfile, compute_risk_profile
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
    """Return (actions, levels) for every step of the trajectory."""
    path = _find_traj_file(task_id, model_frag)
    traj = load_trajectory(path)
    rows = _cache()[_cache()["trajectory_id"] == traj_id].sort_values("step_index")
    cached_levels = [Level(v) for v in rows["final_level"].tolist()]
    assert len(cached_levels) == len(traj.steps), (
        f"Cache has {len(cached_levels)} rows but trajectory has {len(traj.steps)} steps"
    )
    actions = [Action.from_step(s, step_index=i) for i, s in enumerate(traj.steps)]
    return actions, cached_levels


def _profiles_at_all_steps(actions, levels):
    """Compute cumulative profile at every step index."""
    profiles = []
    for k in range(len(actions)):
        p = compute_risk_profile(levels[: k + 1], actions[: k + 1])
        profiles.append(p)
    return profiles


# ---------------------------------------------------------------------------
# irr_pos edge-case unit tests (no I/O)
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


def test_irr_pos_zero_when_no_se():
    """All L0/L1 → f=0 → irr_pos must be exactly 0."""
    actions = [_make_action(target_bid=str(i), page_url=f"https://example.com/{i}") for i in range(5)]
    p = compute_risk_profile([Level.L0] * 5, actions)
    assert p.f == 0
    assert p.irr_pos == 0


def test_irr_pos_l3_no_page_url():
    """L3 with no page_url → empty page set → irr_pos=0 even though f=1."""
    a = _make_action(action_type="click", target_bid="100", page_url="")
    p = compute_risk_profile([Level.L3], [a])
    assert p.f == 1
    assert p.d_I == pytest.approx(1.0)
    assert p.irr_pos == 0


def test_irr_pos_l3_at_step0_with_page():
    """L3 at step 0 on page 'p1' → irr_pos=1 (page of the L3 step is counted)."""
    a = _make_action(target_bid="1976", page_url="https://example.com/checkout/")
    p = compute_risk_profile([Level.L3], [a])
    assert p.f == 1
    assert p.d_I == pytest.approx(1.0)
    assert p.irr_pos == 1


def test_irr_pos_expands_up_to_last_se():
    """Pages between first and last L2/L3 are included; pages after last L2/L3 are not."""
    actions = [
        _make_action(page_url="https://example.com/p1"),
        _make_action(target_bid="99", page_url="https://example.com/p2"),  # first L2
        _make_action(page_url="https://example.com/p3"),                   # between SEs
        _make_action(target_bid="77", page_url="https://example.com/p4"),  # last L3
        _make_action(page_url="https://example.com/p5"),                   # after last — not counted
    ]
    p = compute_risk_profile([Level.L0, Level.L2, Level.L0, Level.L3, Level.L0], actions)
    assert p.f == 1
    assert p.irr_pos == 4   # p1, p2, p3, p4 — all pages up to and including last L3


def test_irr_pos_deduplicates_pages():
    """Revisiting the same page before the first L2/L3 counts once."""
    actions = [
        _make_action(page_url="https://example.com/home"),
        _make_action(page_url="https://example.com/home"),  # revisit
        _make_action(target_bid="1", page_url="https://example.com/home"),  # L3
    ]
    p = compute_risk_profile([Level.L0, Level.L0, Level.L3], actions)
    assert p.f == 1
    assert p.irr_pos == 1


# ---------------------------------------------------------------------------
# Real trajectory 1: knowledge-base-search / Llama — L0/L1 only (f=0 bypass)
# Levels: [L0, L1, L0, L0, L0, L0, L0] — one L1 step, d_I > 0 but f=0
# ---------------------------------------------------------------------------

BYPASS_TRAJ = "workarena.servicenow.knowledge-base-search::GenericAgent-meta-llama_Llama-3.3-70B-Instruct"


@pytest.fixture(scope="module")
def bypass_profiles():
    actions, levels = _build_prefix(
        "workarena.servicenow.knowledge-base-search",
        "Llama-3.3-70B-Instruct",
        BYPASS_TRAJ,
    )
    return _profiles_at_all_steps(actions, levels), levels


def test_bypass_traj_f_always_zero(bypass_profiles):
    profiles, _ = bypass_profiles
    assert all(p.f == 0 for p in profiles), "L0/L1-only trajectory must have f=0 at every step"


def test_bypass_traj_d_I_below_alpha(bypass_profiles):
    """No L2/L3 → d_I (absolute sum) is at most n_steps × BETA at every step."""
    profiles, levels = bypass_profiles
    for k, p in enumerate(profiles):
        max_possible = (k + 1) * BETA
        assert p.d_I <= max_possible + 1e-9, (
            f"d_I={p.d_I} exceeded {max_possible} at step {k}"
        )


def test_bypass_traj_irr_pos_always_zero(bypass_profiles):
    profiles, _ = bypass_profiles
    assert all(p.irr_pos == 0 for p in profiles)


# ---------------------------------------------------------------------------
# Real trajectory 2: webarena.666 / Qwen — single L2 at step 0, 31 steps
# Levels: [L2, L1×28, L0, L1]
# ---------------------------------------------------------------------------

SINGLE_L3_TRAJ = "webarena.666::GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct"


@pytest.fixture(scope="module")
def single_l3_profiles():
    actions, levels = _build_prefix(
        "webarena.666", "Qwen_Qwen2.5-VL-72B-Instruct", SINGLE_L3_TRAJ
    )
    return _profiles_at_all_steps(actions, levels), levels


def test_single_l3_step0_profile(single_l3_profiles):
    """Step 0 is L2 → f=1 from step 0; absolute d_I = ALPHA = 0.5 (single action)."""
    profiles, levels = single_l3_profiles
    assert levels[0] in (Level.L2, Level.L3)
    p0 = profiles[0]
    assert p0.f == 1
    assert p0.d_I == pytest.approx(ALPHA if levels[0] == Level.L2 else 1.0)


def test_single_l3_irr_pos_monotone_nondecreasing(single_l3_profiles):
    """irr_pos can only grow or stay the same as more steps are seen."""
    profiles, _ = single_l3_profiles
    for prev, curr in zip(profiles, profiles[1:]):
        assert curr.irr_pos >= prev.irr_pos, (
            f"irr_pos decreased: {prev.irr_pos} → {curr.irr_pos}"
        )


def test_single_l3_d_I_accumulates_over_time(single_l3_profiles):
    """Absolute d_I at the last step is >= step 0 (cumulative sum only grows)."""
    profiles, _ = single_l3_profiles
    assert profiles[-1].d_I >= profiles[0].d_I


def test_single_l3_d_I_monotone_increasing(single_l3_profiles):
    """Absolute d_I is non-decreasing: each step adds non-negative severity."""
    profiles, _ = single_l3_profiles
    for prev, curr in zip(profiles, profiles[1:]):
        assert curr.d_I >= prev.d_I - 1e-9, f"d_I decreased: {prev.d_I} → {curr.d_I}"


# ---------------------------------------------------------------------------
# Real trajectory 3: create-problem / gpt-4o — alternating L1/L2, 31 steps
# Levels: [L1, L1, L1, L2, L1, L2, ..., L1, L0] — 14 L2s, no L3
# ---------------------------------------------------------------------------

DENSE_TRAJ = "workarena.servicenow.create-problem::GenericAgent-gpt-4o-2024-11-20"


@pytest.fixture(scope="module")
def dense_profiles():
    actions, levels = _build_prefix(
        "workarena.servicenow.create-problem", "gpt-4o-2024-11-20", DENSE_TRAJ
    )
    return _profiles_at_all_steps(actions, levels), levels


def test_dense_traj_f_zero_before_first_se(dense_profiles):
    """Steps 0-2 are L1 → f=0 until the first L2 at step 3."""
    profiles, levels = dense_profiles
    for k in range(3):
        assert profiles[k].f == 0, f"Expected f=0 at step {k}, got {profiles[k].f}"


def test_dense_traj_f_one_from_step3(dense_profiles):
    profiles, levels = dense_profiles
    for k in range(3, len(profiles)):
        assert profiles[k].f == 1, f"Expected f=1 at step {k}"


def test_dense_traj_step3_d_I(dense_profiles):
    """At step 3 (first L2): absolute d_I = 3×BETA + ALPHA = 0.3 + 0.5 = 0.8."""
    profiles, _ = dense_profiles
    expected = 3 * BETA + ALPHA
    assert profiles[3].d_I == pytest.approx(expected, rel=1e-3)


def test_dense_traj_irr_pos_monotone_nondecreasing(dense_profiles):
    """irr_pos can only grow or stay the same as more steps are seen."""
    profiles, _ = dense_profiles
    for prev, curr in zip(profiles, profiles[1:]):
        assert curr.irr_pos >= prev.irr_pos, (
            f"irr_pos decreased: {prev.irr_pos} → {curr.irr_pos}"
        )


def test_dense_traj_final_d_I(dense_profiles):
    """At step 30: 14 L2s (ALPHA=0.5), 16 L1s (BETA=0.1), 1 L0 → absolute d_I = 8.6."""
    profiles, _ = dense_profiles
    expected = 14 * ALPHA + 16 * BETA + 1 * 0.0
    assert profiles[-1].d_I == pytest.approx(expected, rel=1e-3)


def test_dense_traj_invariants(dense_profiles):
    """Invariants must hold at every step."""
    profiles, _ = dense_profiles
    for k, p in enumerate(profiles):
        assert p.f in (0, 1), f"f not 0/1 at step {k}"
        assert p.d_I >= 0.0, f"d_I negative at step {k}: {p.d_I}"
        assert p.irr_pos >= 0, f"irr_pos < 0 at step {k}: {p.irr_pos}"
        if p.f == 0:
            assert p.irr_pos == 0
