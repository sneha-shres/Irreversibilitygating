"""Patch an existing classification_cache.parquet to add f, d_I, pi columns.

Reads cached final_level values (no LLM calls), loads trajectory JSON files
to get actions and axtrees, then computes the running risk profile at each step.

Usage (from repo root):
    PYTHONPATH=. python3 scripts/patch_cache_add_profiles.py
    PYTHONPATH=. python3 scripts/patch_cache_add_profiles.py \\
        --cache data/classification_cache.parquet \\
        --trajectory-dir data/raw
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

from irrgate.actions import Action
from irrgate.data.loader import load_trajectory
from irrgate.profile import compute_risk_profile
from irrgate.taxonomy import Level


def find_trajectory_file(task_id: str, trajectory_dir: str, model: str | None) -> str:
    candidate = os.path.join(trajectory_dir, f"{task_id}.json")
    if os.path.exists(candidate):
        return candidate
    cleaned = os.path.join(trajectory_dir, "cleaned")
    if os.path.exists(cleaned):
        for root, _dirs, files in os.walk(cleaned):
            if f"{task_id}.json" in files:
                if model is None or model in root:
                    return os.path.join(root, f"{task_id}.json")
    raise FileNotFoundError(f"Trajectory file for task_id='{task_id}' not found in {trajectory_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cache", default="data/classification_cache.parquet")
    parser.add_argument("--trajectory-dir", default="data/raw")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: cache not found at {cache_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(cache_path)

    if "f" in df.columns and "d_I" in df.columns and "pi" in df.columns:
        print("Cache already has f, d_I, pi columns — nothing to do.")
        return

    print(f"Patching {len(df):,} rows across {df['trajectory_id'].nunique()} trajectories …")
    start = time.time()

    f_vals: list[int] = []
    d_I_vals: list[float] = []
    pi_vals: list[float] = []

    traj_ids = df["trajectory_id"].unique()
    for i, traj_id in enumerate(traj_ids, start=1):
        traj_rows = df[df["trajectory_id"] == traj_id].sort_values("step_index")

        # Parse task_id and model from trajectory_id (format: "task_id::model")
        parts = traj_id.split("::", 1)
        task_id = parts[0]
        model = parts[1] if len(parts) > 1 and parts[1] else None

        try:
            path = find_trajectory_file(task_id, args.trajectory_dir, model)
            traj = load_trajectory(path)
        except FileNotFoundError as exc:
            print(f"  [{i}/{len(traj_ids)}] SKIP (missing file) {traj_id}: {exc}", file=sys.stderr)
            for _ in range(len(traj_rows)):
                f_vals.append(0)
                d_I_vals.append(0.0)
                pi_vals.append(0.0)
            continue

        cached_levels = [Level(int(v)) for v in traj_rows["final_level"].tolist()]
        actions = [Action.from_step(s, step_index=k) for k, s in enumerate(traj.steps)]
        axtrees = [str(s.get("axtree", "")) for s in traj.steps]

        if len(cached_levels) != len(traj.steps):
            print(
                f"  [{i}/{len(traj_ids)}] WARNING: cache has {len(cached_levels)} rows "
                f"but trajectory has {len(traj.steps)} steps — using trajectory length",
                file=sys.stderr,
            )
            n = min(len(cached_levels), len(traj.steps))
            cached_levels = cached_levels[:n]
            actions = actions[:n]
            axtrees = axtrees[:n]

        for k in range(len(cached_levels)):
            p = compute_risk_profile(cached_levels[: k + 1], actions[: k + 1], axtrees[: k + 1])
            f_vals.append(p.f)
            d_I_vals.append(p.d_I)
            pi_vals.append(p.pi)

        if i % 100 == 0 or i == len(traj_ids):
            print(f"  [{i}/{len(traj_ids)}] done  ({time.time()-start:.0f}s)", flush=True)

    df = df.sort_values(["trajectory_id", "step_index"]).reset_index(drop=True)
    df["f"] = pd.array(f_vals, dtype="int8")
    df["d_I"] = d_I_vals
    df["pi"] = pi_vals

    df.to_parquet(cache_path, index=False)
    print(f"Done — wrote {len(df):,} rows to {cache_path}  ({time.time()-start:.1f}s total)")


if __name__ == "__main__":
    main()
