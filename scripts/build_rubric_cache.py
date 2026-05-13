"""Build the per-step rubric cache.

For every step k of every trajectory, pre-computes and stores:

  Profile (cumulative, using classification_cache.parquet for levels):
    f    — 1 if any L≥2 action seen so far, else 0
    d_I  — mean severity over steps 0..k
    pi   — plan-level grounding-weighted risk

  Rubric verdicts (using axtrees from the raw trajectory files):
    r1  — all L≥2 bids visible in a prior axtree
    r2  — all L≥2 args traceable to prior axtrees
    r3  — every L3 preceded by a consent step (send_msg / ask_user)
    r4  — Gemini: action consistent with prior state (True = pass); f=1 steps only
    r5  — Gemini: recovery path identifiable (True = pass); f=1 steps only

R4/R5 are only computed for steps where f=1 (f=0 → BYPASS → rubric never applies at runtime).
R4/R5 results are cached to data/rubric_llm_cache.json so reruns are free.

Output: data/rubric_cache.parquet  (one row per trajectory_id × step_index)

Usage (from repo root):
    PYTHONPATH=. python3 scripts/build_rubric_cache.py
    PYTHONPATH=. python3 scripts/build_rubric_cache.py --no-resume
    PYTHONPATH=. python3 scripts/build_rubric_cache.py --skip-llm   # R1-R3 only, no Gemini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

import irrgate.config  # noqa: F401 — triggers .env loading
from irrgate.actions import Action
from irrgate.data.loader import load_trajectory
from irrgate.profile import (
    _target_bid_seen_in_prior_axtrees,
    _fill_text_seen_in_prior_axtrees,
    compute_risk_profile,
)
from irrgate.rubric import (
    RUBRIC_PROMPT_VERSION,
    r3_consent_precedes_L3,
    rubric_llm_check,
    _save_rubric_cache,
)
from irrgate.taxonomy import Level


def find_trajectory_file(task_id: str, trajectory_dir: str, model: str | None = None) -> str:
    candidate = os.path.join(trajectory_dir, f"{task_id}.json")
    if os.path.exists(candidate):
        return candidate
    cleaned = os.path.join(trajectory_dir, "cleaned")
    if os.path.exists(cleaned):
        for root, _dirs, files in os.walk(cleaned):
            if f"{task_id}.json" in files:
                if model is None or model in root:
                    return os.path.join(root, f"{task_id}.json")
    raise FileNotFoundError(
        f"Trajectory file for task_id='{task_id}'"
        + (f" model='{model}'" if model else "")
        + f" not found in {trajectory_dir}"
    )


def _make_traj_id(task_id: str, model: str) -> str:
    return f"{task_id}::{model}"


def _flush(output_path: Path, new_rows: list[dict]) -> None:
    new_df = pd.DataFrame(new_rows)
    if not new_df.empty:
        new_df["step_index"] = new_df["step_index"].astype("int16")
        new_df["f"] = new_df["f"].astype("int8")
        new_df["d_I"] = new_df["d_I"].astype("float32")
        new_df["pi"] = new_df["pi"].astype("float32")
        for col in ["r1", "r2", "r3"]:
            new_df[col] = new_df[col].astype(bool)

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)


def build(
    eval_set_path: str,
    trajectory_dir: str,
    class_cache_path: str,
    output_path: str,
    resume: bool = True,
    flush_every: int = 10,
    skip_llm: bool = False,
) -> None:
    with open(eval_set_path, encoding="utf-8") as f:
        eval_data = json.load(f)
    all_entries = eval_data["positives"] + eval_data["negatives"]

    class_cache = pd.read_parquet(class_cache_path)

    output_p = Path(output_path)
    done_ids: set[str] = set()
    if not resume and output_p.exists():
        output_p.unlink()
        print(f"[rubric] --no-resume: deleted existing {output_path}", flush=True)
    if resume and output_p.exists():
        existing = pd.read_parquet(output_p)
        done_ids = set(existing["trajectory_id"].unique())
        print(f"[rubric] resuming — {len(done_ids)} trajectories already cached", flush=True)

    total = len(all_entries)
    pending = [
        e for e in all_entries
        if _make_traj_id(e["task_id"], e.get("model", "")) not in done_ids
    ]
    print(f"[rubric] {len(pending)} trajectories to process ({total - len(pending)} skipped)", flush=True)
    if skip_llm:
        print("[rubric] --skip-llm: R4/R5 will be stored as null", flush=True)

    buffer: list[dict] = []
    start = time.time()

    for i, meta in enumerate(pending, start=1):
        task_id = meta["task_id"]
        model = meta.get("model", "")
        traj_id = _make_traj_id(task_id, model)

        try:
            path = find_trajectory_file(task_id, trajectory_dir, model=model or None)
            traj = load_trajectory(path)
        except FileNotFoundError as exc:
            print(f"[rubric] [{i}/{len(pending)}] SKIP (missing file) {traj_id}: {exc}",
                  file=sys.stderr, flush=True)
            continue

        # Get pre-classified levels from the classification cache
        traj_rows = (
            class_cache[class_cache["trajectory_id"] == traj_id]
            .sort_values("step_index")
        )
        if len(traj_rows) != len(traj.steps):
            print(
                f"[rubric] [{i}/{len(pending)}] SKIP (cache mismatch: "
                f"{len(traj_rows)} cached vs {len(traj.steps)} steps) {traj_id}",
                file=sys.stderr, flush=True,
            )
            continue

        levels = [Level(v) for v in traj_rows["final_level"].tolist()]

        t0 = time.time()
        # Build prefix incrementally — O(n) per trajectory
        actions_so_far: list[Action] = []
        levels_so_far: list[Level] = []
        axtrees_so_far: list[str] = []
        prior_axtrees: list[str] = []   # axtrees[0:k] for grounding checks at step k

        f_flag = 0          # cumulative f flag
        r1_ok = True        # False once any L≥2 bid is not in prior axtrees
        r2_ok = True        # False once any L≥2 args are not traceable
        r3_ok = True        # False once any L3 lacks a prior consent

        for k, step in enumerate(traj.steps):
            action = Action.from_step(step, step_index=k)
            axtree = str(step.get("axtree", ""))
            level = levels[k]

            actions_so_far.append(action)
            levels_so_far.append(level)
            axtrees_so_far.append(axtree)

            # Profile (cumulative)
            profile = compute_risk_profile(levels_so_far, actions_so_far, axtrees_so_far)

            # Update f flag
            if level.value >= 2:
                f_flag = 1

            # R1 / R2 — incremental: only need to check the new L≥2 action
            if level >= Level.L2:
                bid = action.target_bid
                fill = action.fill_text

                if bid is None or not _target_bid_seen_in_prior_axtrees(bid, prior_axtrees):
                    r1_ok = False

                args_total = int(bid is not None) + int(fill is not None)
                if args_total > 0:
                    trace_count = 0
                    if bid is not None and _target_bid_seen_in_prior_axtrees(bid, prior_axtrees):
                        trace_count += 1
                    if fill is not None and _fill_text_seen_in_prior_axtrees(fill, prior_axtrees):
                        trace_count += 1
                    if trace_count < args_total:
                        r2_ok = False

            # R3 — recompute over full prefix each step (O(k) but k is small in practice)
            r3_ok = r3_consent_precedes_L3(actions_so_far, levels_so_far)

            # R4 / R5 — Gemini, only when f=1 (BYPASS never checks rubric)
            r4: bool | None = None
            r5: bool | None = None
            if f_flag == 1 and not skip_llm:
                llm = rubric_llm_check(actions_so_far, k, axtrees_so_far)
                r4 = llm["R4"]
                r5 = llm["R5"]

            buffer.append({
                "trajectory_id": traj_id,
                "step_index": k,
                "f": profile.f,
                "d_I": profile.d_I,
                "pi": profile.pi,
                "r1": r1_ok,
                "r2": r2_ok,
                "r3": r3_ok,
                "r4": r4,
                "r5": r5,
                "r4_computed": r4 is not None,
                "rubric_prompt_version": RUBRIC_PROMPT_VERSION,
            })

            prior_axtrees.append(axtree)

        elapsed = time.time() - t0
        total_elapsed = time.time() - start
        print(
            f"[rubric] [{i}/{len(pending)}] {traj_id}  steps={len(traj.steps)}  "
            f"step_time={elapsed:.1f}s  total={total_elapsed:.0f}s",
            flush=True,
        )

        if i % flush_every == 0 or i == len(pending):
            _flush(output_p, buffer)
            _save_rubric_cache()   # persist R4/R5 LLM cache to disk
            buffer = []
            print(f"[rubric] flushed to {output_path}", flush=True)

    if buffer:
        _flush(output_p, buffer)
        _save_rubric_cache()

    if output_p.exists():
        final_df = pd.read_parquet(output_p)
        n_r4 = final_df["r4_computed"].sum() if "r4_computed" in final_df.columns else 0
        print(
            f"[rubric] done — {len(final_df)} rows across "
            f"{final_df['trajectory_id'].nunique()} trajectories  "
            f"r4_r5_computed={n_r4}  rubric_prompt_version={RUBRIC_PROMPT_VERSION}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IrrGate per-step rubric cache (Parquet).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-set",      default="data/eval_set.json")
    parser.add_argument("--trajectory-dir", default="data/raw")
    parser.add_argument("--class-cache",   default="data/classification_cache.parquet",
                        help="Pre-built classification cache parquet")
    parser.add_argument("--output",        default="data/rubric_cache.parquet")
    parser.add_argument("--no-resume",     action="store_true")
    parser.add_argument("--flush-every",   type=int, default=10)
    parser.add_argument("--skip-llm",      action="store_true",
                        help="Compute only R1-R3 (no Gemini); R4/R5 stored as null")
    args = parser.parse_args()

    build(
        eval_set_path=args.eval_set,
        trajectory_dir=args.trajectory_dir,
        class_cache_path=args.class_cache,
        output_path=args.output,
        resume=not args.no_resume,
        flush_every=args.flush_every,
        skip_llm=args.skip_llm,
    )


if __name__ == "__main__":
    main()
