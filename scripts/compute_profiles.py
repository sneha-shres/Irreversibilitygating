"""Compute full-trajectory risk profiles for all 870 trajectories.

Reads classification levels from data/classification_cache.parquet (no LLM calls)
and action BID data from the raw trajectory files, then applies the current
compute_risk_profile formula at every step prefix to derive:

  peak_d_I               — max d_I reached across all steps
  peak_pi                — max pi reached across all steps
  side_effect_step       — latest step with final_level >= 2 (L2 or L3)
  d_I_at_side_effect_step  — d_I at that step (None if side_effect_step is None)
  pi_at_side_effect_step   — pi at that step

Output: results/profiles/profiles.parquet  (one row per trajectory)

Usage:
    PYTHONPATH=. python3 scripts/compute_profiles.py
"""

from __future__ import annotations

import json
import os
from typing import Optional

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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    rows: list[dict] = []
    for i, (meta, is_positive) in enumerate(all_meta, start=1):
        traj_id = f"{meta['task_id']}::{meta['model']}"
        cache_rows = cache_by_traj.get(traj_id)
        if cache_rows is None or len(cache_rows) == 0:
            print(f"[profiles] [{i}/{total}] MISSING cache for {traj_id}")
            continue
        row = _process_trajectory(traj_id, meta, is_positive, cache_rows)
        rows.append(row)
        label = "+pos" if is_positive else "-neg"
        print(
            f"[profiles] [{i}/{total}] {label} {traj_id[:70]}"
            f"  n={row['n_steps']}  peak_d_I={row['peak_d_I']:.3f}"
            f"  peak_pi={row['peak_pi']:.3f}"
            f"  se_step={row['side_effect_step']}"
        )

    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, "profiles.parquet")
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


if __name__ == "__main__":
    main()
