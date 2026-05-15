"""Compute full-trajectory risk profiles for all 870 trajectories.

Reads classification levels from data/classification_cache.parquet (no LLM calls)
and action BID data from the raw trajectory files, then applies the current
compute_risk_profile formula at every step prefix to derive:

  peak_d_I               — max d_I reached across all steps
  peak_pi                — max pi reached across all steps
  side_effect_step       — latest step with final_level >= 2 (L2 or L3)
  d_I_at_side_effect_step  — d_I at that step (None if side_effect_step is None)
  pi_at_side_effect_step   — pi at that step

Shape features (whole-trajectory, complement density features):
  n_l3_actions           — count of L3 steps
  n_distinct_l3_bids     — distinct non-None target_bid values among L3 steps
  n_distinct_bids_pre_se — distinct non-None target_bid values up to side_effect_step (NaN if f=0)
  n_distinct_pages_pre_se — distinct non-empty page_url values up to side_effect_step (NaN if f=0)
  n_steps_after_last_l3  — steps remaining after last L3 step (NaN if no L3)
  hit_step_cap           — 1 if n_steps >= 30 (BrowserGym budget proxy)
  action_type_entropy    — Shannon entropy (base 2) over action_type distribution

Output: results/profiles/profiles.parquet  (one row per trajectory)

Usage:
    PYTHONPATH=. python3 scripts/compute_profiles.py
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import pandas as pd

from irrgate.actions import Action
from irrgate.data.loader import load_trajectory
from irrgate.profile import compute_risk_profile
from irrgate.taxonomy import Level


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_trajectory_file(task_id: str, model: str, trajectory_dir: str) -> str:
    cleaned = os.path.join(trajectory_dir, "cleaned")
    if os.path.exists(cleaned):
        for root, _dirs, files in os.walk(cleaned):
            if f"{task_id}.json" in files and model in root:
                return os.path.join(root, f"{task_id}.json")
    direct = os.path.join(trajectory_dir, f"{task_id}.json")
    if os.path.exists(direct):
        return direct
    raise FileNotFoundError(f"Trajectory for {task_id!r} / {model!r} not found in {trajectory_dir}")


def _mannwhitney_auc(pos_vals: list[float], neg_vals: list[float]) -> float:
    """AUC = P(pos_feature > neg_feature) via Mann-Whitney U. Filters NaN."""
    p = np.asarray([v for v in pos_vals if v == v], dtype=float)
    n = np.asarray([v for v in neg_vals if v == v], dtype=float)
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    u = sum(float(np.sum(pi > n)) + 0.5 * float(np.sum(pi == n)) for pi in p)
    return u / (len(p) * len(n))


def _process_trajectory(
    traj_id: str,
    meta: dict,
    is_positive: bool,
    cache_rows: pd.DataFrame,
) -> dict:
    task_id = meta["task_id"]
    model = meta.get("model", "")
    benchmark = meta.get("benchmark", "")

    steps = cache_rows.sort_values("step_index")
    levels = [Level(v) for v in steps["final_level"].tolist()]
    n_steps = len(levels)

    # Load trajectory for Action objects (needed for target_bid in pi formula)
    traj_dir = "data/raw"
    path = _find_trajectory_file(task_id, model, traj_dir)
    traj = load_trajectory(path)
    assert len(traj.steps) == n_steps, (
        f"{traj_id}: cache has {n_steps} rows but trajectory has {len(traj.steps)} steps"
    )
    actions = [Action.from_step(s, step_index=i) for i, s in enumerate(traj.steps)]

    # Compute cumulative profile at every step; track peaks
    peak_d_I = 0.0
    peak_pi = 0.0
    side_effect_step: Optional[int] = None
    d_I_at_se: Optional[float] = None
    pi_at_se: Optional[float] = None

    step_profiles: list[tuple[float, float]] = []  # (d_I, pi) per step

    for k in range(n_steps):
        p = compute_risk_profile(levels[: k + 1], actions[: k + 1])
        step_profiles.append((p.d_I, p.pi))
        if p.d_I > peak_d_I:
            peak_d_I = p.d_I
        if p.pi > peak_pi:
            peak_pi = p.pi
        if levels[k].value >= 2:
            side_effect_step = k

    # Profile at the side-effect step (latest L2/L3)
    if side_effect_step is not None:
        d_I_at_se, pi_at_se = step_profiles[side_effect_step]

    # Shape features — single pass over levels and actions
    n_l3_actions = 0
    l3_bids: set[str] = set()
    pre_se_bid_set: set[str] = set()
    pre_se_page_set: set[str] = set()
    last_l3_idx: Optional[int] = None
    action_type_counts: dict[str, int] = {}
    n_l2_actions = 0
    l2_bids: set[str] = set()

    for k, (lv, act) in enumerate(zip(levels, actions)):
        action_type_counts[act.action_type] = action_type_counts.get(act.action_type, 0) + 1
        if lv == Level.L3:
            n_l3_actions += 1
            if act.target_bid is not None:
                l3_bids.add(act.target_bid)
            last_l3_idx = k
        if lv == Level.L2:
            n_l2_actions += 1
            if act.target_bid is not None:
                l2_bids.add(act.target_bid)
        if side_effect_step is not None and k <= side_effect_step:
            if act.target_bid is not None:
                pre_se_bid_set.add(act.target_bid)
            if act.page_url:
                pre_se_page_set.add(act.page_url)

    n_distinct_l2_bids: int = len(l2_bids)
    n_l2_actions_pre_se: int = (
        sum(1 for k, lv in enumerate(levels) if lv == Level.L2 and k <= last_l3_idx)
        if last_l3_idx is not None
        else n_l2_actions
    )
    n_distinct_l3_bids: int = len(l3_bids)
    n_distinct_bids_pre_se: float = (
        float(len(pre_se_bid_set)) if side_effect_step is not None else float("nan")
    )
    n_distinct_pages_pre_se: float = (
        float(len(pre_se_page_set)) if side_effect_step is not None else float("nan")
    )
    n_steps_after_last_l3: float = (
        float(n_steps - 1 - last_l3_idx) if last_l3_idx is not None else float("nan")
    )
    hit_step_cap: int = int(n_steps >= 30)

    total_acts = len(actions)
    if total_acts == 0:
        action_type_entropy = 0.0
    else:
        action_type_entropy = -sum(
            (c / total_acts) * math.log2(c / total_acts)
            for c in action_type_counts.values()
        )

    return {
        "trajectory_id": traj_id,
        "task_id": task_id,
        "model": model,
        "benchmark": benchmark,
        "is_positive": is_positive,
        "n_steps": n_steps,
        "side_effect_step": side_effect_step,
        "peak_d_I": peak_d_I,
        "peak_pi": peak_pi,
        "d_I_at_side_effect_step": d_I_at_se,
        "pi_at_side_effect_step": pi_at_se,
        # f flag: 1 iff any step has severity > 0  (L2 or L3 only)
        "f": int(any(lv.value >= 2 for lv in levels)),
        # L2 counters
        "n_l2_actions": n_l2_actions,
        "n_distinct_l2_bids": n_distinct_l2_bids,
        "n_l2_actions_pre_se": n_l2_actions_pre_se,
        # Shape features
        "n_l3_actions": n_l3_actions,
        "n_distinct_l3_bids": n_distinct_l3_bids,
        "n_distinct_bids_pre_se": n_distinct_bids_pre_se,
        "n_distinct_pages_pre_se": n_distinct_pages_pre_se,
        "n_steps_after_last_l3": n_steps_after_last_l3,
        "hit_step_cap": hit_step_cap,
        "action_type_entropy": action_type_entropy,
        # Not saved to parquet — used only for validation spot-check
        "_level_seq": "".join(str(lv.value) for lv in levels),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_NEW_FEAT_COLS = [
    "n_l2_actions",
    "n_distinct_l2_bids",
    "n_l2_actions_pre_se",
    "n_l3_actions",
    "n_distinct_l3_bids",
    "n_distinct_bids_pre_se",
    "n_distinct_pages_pre_se",
    "n_steps_after_last_l3",
    "hit_step_cap",
    "action_type_entropy",
]

_ORIGINAL_COLS = [
    "trajectory_id", "task_id", "model", "benchmark", "is_positive",
    "n_steps", "side_effect_step", "peak_d_I", "peak_pi",
    "d_I_at_side_effect_step", "pi_at_side_effect_step", "f",
]


def main() -> None:
    output_dir = "results/profiles"
    os.makedirs(output_dir, exist_ok=True)

    with open("data/eval_set.json", encoding="utf-8") as fh:
        eval_set = json.load(fh)

    cache = pd.read_parquet("data/classification_cache.parquet")
    cache_by_traj = {tid: grp for tid, grp in cache.groupby("trajectory_id")}

    all_meta: list[tuple[dict, bool]] = (
        [(m, True)  for m in eval_set["positives"]] +
        [(m, False) for m in eval_set["negatives"]]
    )
    total = len(all_meta)

    # Preserve old dtypes for assertion (read before overwriting)
    out = os.path.join(output_dir, "profiles.parquet")
    old_dtypes: dict = {}
    if os.path.exists(out):
        old_dtypes = pd.read_parquet(out).dtypes.to_dict()

    rows: list[dict] = []
    level_seqs: dict[str, str] = {}
    for i, (meta, is_positive) in enumerate(all_meta, start=1):
        traj_id = f"{meta['task_id']}::{meta['model']}"
        cache_rows = cache_by_traj.get(traj_id)
        if cache_rows is None or len(cache_rows) == 0:
            print(f"[profiles] [{i}/{total}] MISSING cache for {traj_id}")
            continue
        row = _process_trajectory(traj_id, meta, is_positive, cache_rows)
        level_seqs[traj_id] = row.pop("_level_seq")
        rows.append(row)
        label = "+pos" if is_positive else "-neg"
        print(
            f"[profiles] [{i}/{total}] {label} {traj_id[:70]}"
            f"  n={row['n_steps']}  peak_d_I={row['peak_d_I']:.3f}"
            f"  peak_pi={row['peak_pi']:.3f}"
            f"  se_step={row['side_effect_step']}"
        )

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"\n[profiles] saved {len(df)} trajectories → {out}")

    pos = df[df["is_positive"]]
    neg = df[~df["is_positive"]]
    print(f"[profiles] positives: {len(pos)},  negatives: {len(neg)}")
    print(f"[profiles] f=1 positives: {(pos['f']==1).sum()},  f=1 negatives: {(neg['f']==1).sum()}")
    print(f"[profiles] peak_d_I  pos median={pos['peak_d_I'].median():.3f}  "
          f"neg median={neg['peak_d_I'].median():.3f}")
    print(f"[profiles] peak_pi   pos median={pos['peak_pi'].median():.3f}  "
          f"neg median={neg['peak_pi'].median():.3f}")
    for col in _NEW_FEAT_COLS:
        print(f"[profiles] {col:<30}  pos median={pos[col].median():.3f}  "
              f"neg median={neg[col].median():.3f}")

    # ------------------------------------------------------------------
    # AUC (Mann-Whitney) on f=1 subset
    # ------------------------------------------------------------------
    f1 = df[df["f"] == 1]
    f1_pos = f1[f1["is_positive"]]
    f1_neg = f1[~f1["is_positive"]]
    print(f"\n[profiles] === AUC (Mann-Whitney, f=1 subset: "
          f"n_pos={len(f1_pos)}, n_neg={len(f1_neg)}) ===")
    for col in ["peak_d_I", "peak_pi"] + _NEW_FEAT_COLS:
        auc = _mannwhitney_auc(f1_pos[col].tolist(), f1_neg[col].tolist())
        direction = "^" if (not math.isnan(auc) and auc >= 0.5) else "v"
        flag = "  [<0.55 — low discrim]" if (not math.isnan(auc) and auc < 0.55) else ""
        print(f"[profiles]   {col:<30}: AUC = {auc:.3f} {direction}{flag}")

    # ------------------------------------------------------------------
    # Spot-check: 3 representative trajectories
    # ------------------------------------------------------------------
    easy_pos_pool = df[(df["is_positive"]) & (df["peak_d_I"] > 0.5)]
    hard_pos_pool = df[(df["is_positive"]) & (df["peak_d_I"] < 0.2)]
    hard_neg_pool = df[(~df["is_positive"]) & (df["peak_d_I"] > 0.5)]

    print("\n[profiles] === Spot-check trajectories ===")
    for label, pool in [
        ("easy_pos (peak_d_I > 0.5)", easy_pos_pool),
        ("hard_pos (peak_d_I < 0.2)", hard_pos_pool),
        ("hard_neg (peak_d_I > 0.5)", hard_neg_pool),
    ]:
        if len(pool) == 0:
            print(f"[profiles]   {label}: no trajectories found")
            continue
        sample = pool.sample(1, random_state=42).iloc[0]
        tid = sample["trajectory_id"]
        seq = level_seqs.get(tid, "?")
        print(f"[profiles]   {label}")
        print(f"[profiles]     id    : {tid}")
        print(f"[profiles]     levels: {seq}")
        for col in _NEW_FEAT_COLS:
            print(f"[profiles]     {col:<30}: {sample[col]}")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert len(df) == 870, f"Expected 870 rows, got {len(df)}"
    for col in _ORIGINAL_COLS:
        assert col in df.columns, f"Missing original column: {col}"
        if col in old_dtypes:
            assert df[col].dtype == old_dtypes[col], (
                f"Dtype changed for {col}: was {old_dtypes[col]}, now {df[col].dtype}"
            )
    print("\n[profiles] Assertions passed: 870 rows, all original columns present with correct dtypes.")


if __name__ == "__main__":
    main()
