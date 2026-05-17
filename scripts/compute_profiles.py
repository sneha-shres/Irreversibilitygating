"""Compute full-trajectory risk profiles for all 870 trajectories.

Reads classification levels from data/classification_cache.parquet (no LLM calls)
and action data from the raw trajectory files, then computes the three IrrGate features:

  f                        — irreversibility presence (1 iff any L2/L3 step)
  d_I                      — irreversibility density: absolute cumulative severity (sum, not mean)
  irr_pos                  — irreversibility positional risk: distinct pages up to last L2/L3
  side_effect_step         — index of last L2/L3 step (None if f=0)
  d_I_at_side_effect_step  — density at that step (None if f=0)

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

    path = _find_trajectory_file(task_id, model, "data/raw")
    traj = load_trajectory(path)
    assert len(traj.steps) == n_steps, (
        f"{traj_id}: cache has {n_steps} rows but trajectory has {len(traj.steps)} steps"
    )
    actions = [Action.from_step(s, step_index=i) for i, s in enumerate(traj.steps)]

    # Full-trajectory profile: f, d_I (absolute cumulative), irr_pos
    # d_I is monotone increasing so no step loop needed for peak — it equals the full value.
    full = compute_risk_profile(levels, actions)

    # Find last L2/L3 step and d_I at that prefix
    side_effect_step: Optional[int] = None
    d_I_at_se: Optional[float] = None
    for k in range(n_steps):
        if levels[k].value >= 2:
            side_effect_step = k
    if side_effect_step is not None:
        p_at_se = compute_risk_profile(
            levels[:side_effect_step + 1], actions[:side_effect_step + 1]
        )
        d_I_at_se = p_at_se.d_I

    return {
        "trajectory_id": traj_id,
        "task_id": task_id,
        "model": model,
        "benchmark": benchmark,
        "is_positive": is_positive,
        "n_steps": n_steps,
        "f": full.f,
        "side_effect_step": side_effect_step,
        "d_I": full.d_I,
        "d_I_at_side_effect_step": d_I_at_se,
        "irr_pos": full.irr_pos,
        "_level_seq": "".join(str(lv.value) for lv in levels),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_PROFILE_COLS = [
    "trajectory_id", "task_id", "model", "benchmark", "is_positive",
    "n_steps", "f", "side_effect_step", "d_I", "d_I_at_side_effect_step", "irr_pos",
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
            f"  n={row['n_steps']}  d_I={row['d_I']:.3f}"
            f"  irr_pos={row['irr_pos']}"
            f"  se_step={row['side_effect_step']}"
        )

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"\n[profiles] saved {len(df)} trajectories → {out}")

    pos = df[df["is_positive"]]
    neg = df[~df["is_positive"]]
    print(f"[profiles] f=1 positives: {(pos['f']==1).sum()},  f=1 negatives: {(neg['f']==1).sum()}")
    print(f"[profiles] d_I       pos median={pos['d_I'].median():.3f}  "
          f"neg median={neg['d_I'].median():.3f}")
    print(f"[profiles] irr_pos   pos median={pos['irr_pos'].median():.1f}  "
          f"neg median={neg['irr_pos'].median():.1f}")

    # AUC on f=1 subset
    f1 = df[df["f"] == 1]
    f1_pos = f1[f1["is_positive"]]
    f1_neg = f1[~f1["is_positive"]]
    print(f"\n[profiles] === AUC (Mann-Whitney, f=1 subset: "
          f"n_pos={len(f1_pos)}, n_neg={len(f1_neg)}) ===")
    for col in ["d_I", "irr_pos"]:
        auc = _mannwhitney_auc(f1_pos[col].tolist(), f1_neg[col].tolist())
        direction = "^" if (not math.isnan(auc) and auc >= 0.5) else "v"
        flag = "  [<0.55 — low discrim]" if (not math.isnan(auc) and auc < 0.55) else ""
        print(f"[profiles]   {col:<30}: AUC = {auc:.3f} {direction}{flag}")

    # Spot-check
    easy_pos_pool = df[(df["is_positive"]) & (df["d_I"] > 5.0)]
    hard_pos_pool = df[(df["is_positive"]) & (df["d_I"] < 1.0)]
    hard_neg_pool = df[(~df["is_positive"]) & (df["d_I"] > 5.0)]

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
        print(f"[profiles]     id      : {tid}")
        print(f"[profiles]     levels  : {seq}")
        print(f"[profiles]     d_I     : {sample['d_I']:.3f}")
        print(f"[profiles]     irr_pos : {sample['irr_pos']}")

    # Assertions
    assert len(df) == 870, f"Expected 870 rows, got {len(df)}"
    for col in _PROFILE_COLS:
        assert col in df.columns, f"Missing column: {col}"
    print("\n[profiles] Assertions passed.")


if __name__ == "__main__":
    main()
