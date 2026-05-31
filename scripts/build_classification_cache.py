"""Build per-step classification cache (Parquet).

Runs the classifier (stage1 + stage2) over each trajectory step and writes
`data/classification_cache.parquet`. Safe to resume — existing trajectories
are skipped when `--no-resume` is not used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

import irrgate.config  # noqa: F401 — triggers .env loading at import time
from irrgate.actions import Action
from irrgate.classifier import CLASSIFIER_VERSION, classify_with_details
from irrgate.data.loader import load_trajectory
from irrgate.profile import compute_risk_profile
from irrgate.taxonomy import Level


def find_trajectory_file(task_id: str, trajectory_dir: str, model: str | None = None) -> str:
    """Locate a trajectory JSON by task_id; prefer top-level, then `cleaned/`.

    If `model` is provided, prefer a path that contains the model name.
    """
    candidate = os.path.join(trajectory_dir, f"{task_id}.json")
    if os.path.exists(candidate):
        return candidate
    cleaned = os.path.join(trajectory_dir, "cleaned")
    if os.path.exists(cleaned):
        for root, _dirs, files in os.walk(cleaned):
            if f"{task_id}.json" in files and (model is None or model in root):
                return os.path.join(root, f"{task_id}.json")
    raise FileNotFoundError(f"Trajectory file for task_id='{task_id}'{(' model='+model) if model else ''} not found in {trajectory_dir}")


def _make_trajectory_id(task_id: str, model: str) -> str:
    return f"{task_id}::{model}"


def _make_dataframe(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            "trajectory_id", "step_index", "benchmark", "action_type",
            "target_bid", "fill_text",
            "stage1_level", "stage2_level", "final_level", "stage_used",
            "stage2_raw_response", "stage2_model", "stage2_prompt_version",
            "classifier_version",
            "f", "d_I", "pi",
        ])
    df = pd.DataFrame(rows)
    df["step_index"] = df["step_index"].astype("int16")
    df["final_level"] = df["final_level"].astype("int8")
    df["stage_used"] = df["stage_used"].astype("int8")
    df["stage1_level"] = df["stage1_level"].astype(pd.Int8Dtype())
    df["stage2_level"] = df["stage2_level"].astype(pd.Int8Dtype())
    df["f"] = df["f"].astype("int8")
    return df


def _flush(output_path: Path, new_rows: list[dict]) -> None:
    new_df = _make_dataframe(new_rows)
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)


def build(
    eval_set_path: str,
    trajectory_dir: str,
    output_path: str,
    resume: bool = True,
    flush_every: int = 10,
) -> None:
    with open(eval_set_path, encoding="utf-8") as f:
        eval_data = json.load(f)
    all_entries = eval_data["positives"] + eval_data["negatives"]

    output_p = Path(output_path)
    done_ids: set[str] = set()
    if not resume and output_p.exists():
        output_p.unlink()
        print(f"[cache] --no-resume: deleted existing {output_path}", flush=True)
    if resume and output_p.exists():
        existing_df = pd.read_parquet(output_p)
        done_ids = set(existing_df["trajectory_id"].unique())
        print(f"[cache] resuming — {len(done_ids)} trajectories already cached", flush=True)

    total = len(all_entries)
    pending = [e for e in all_entries if _make_trajectory_id(e["task_id"], e.get("model", "")) not in done_ids]
    print(f"[cache] {len(pending)} trajectories to process ({total - len(pending)} skipped)", flush=True)

    buffer: list[dict] = []
    start = time.time()

    for i, meta in enumerate(pending, start=1):
        task_id = meta["task_id"]
        model = meta.get("model", "")
        benchmark = meta.get("benchmark", "")
        traj_id = _make_trajectory_id(task_id, model)

        try:
            path = find_trajectory_file(task_id, trajectory_dir, model=model or None)
            traj = load_trajectory(path)
        except FileNotFoundError as exc:
            print(f"[cache] [{i}/{len(pending)}] SKIP (missing file) {traj_id}: {exc}", file=sys.stderr, flush=True)
            continue

        t0 = time.time()
        traj_actions: list[Action] = []
        traj_levels: list[Level] = []
        for step_index, step in enumerate(traj.steps):
            action = Action.from_step(step, step_index=step_index)
            result = classify_with_details(action)
            traj_actions.append(action)
            traj_levels.append(result.final_level)
            profile = compute_risk_profile(traj_levels, traj_actions)
            buffer.append({
                "trajectory_id": traj_id,
                "step_index": step_index,
                "benchmark": benchmark,
                "action_type": action.action_type,
                "target_bid": action.target_bid,
                "fill_text": action.fill_text,
                "stage1_level": result.stage1_level.value if result.stage1_level is not None else None,
                "stage2_level": result.stage2_level.value if result.stage2_level is not None else None,
                "final_level": result.final_level.value,
                "stage_used": result.stage_used,
                "stage2_raw_response": result.stage2_raw_response,
                "stage2_model": result.stage2_model,
                "stage2_prompt_version": result.stage2_prompt_version,
                "classifier_version": result.classifier_version,
                "f": profile.f,
                "d_I": profile.d_I,
                "pi": profile.pi,
            })

        elapsed = time.time() - t0
        total_elapsed = time.time() - start
        print(
            f"[cache] [{i}/{len(pending)}] {traj_id}  steps={len(traj.steps)}  "
            f"step_time={elapsed:.1f}s  total={total_elapsed:.0f}s",
            flush=True,
        )

        if i % flush_every == 0 or i == len(pending):
            _flush(output_p, buffer)
            buffer = []
            print(f"[cache] flushed to {output_path}", flush=True)

    if buffer:
        _flush(output_p, buffer)

    if output_p.exists():
        final_df = pd.read_parquet(output_p)
        print(
            f"[cache] done — {len(final_df)} rows across "
            f"{final_df['trajectory_id'].nunique()} trajectories  "
            f"classifier_version={CLASSIFIER_VERSION}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IrrGate per-step classification cache (Parquet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-set", default="data/eval_set.json")
    parser.add_argument("--trajectory-dir", default="data/raw")
    parser.add_argument("--output", default="data/classification_cache.parquet")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing cache and start fresh")
    parser.add_argument("--flush-every", type=int, default=10, help="Write to disk every N trajectories")
    args = parser.parse_args()

    build(
        eval_set_path=args.eval_set,
        trajectory_dir=args.trajectory_dir,
        output_path=args.output,
        resume=not args.no_resume,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
