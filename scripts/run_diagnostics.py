"""IrrGate diagnostics.

Reads:  results/profiles/profiles.parquet
        data/classification_cache.parquet
        results/cv/cv_results.json  (for selected thresholds)
Writes: results/diagnostics/
  per_benchmark_model.json   — recall/FPR per (benchmark, model) cell
  false_negatives.json       — every missed positive with gate-decision features
  fp_sample.json             — 30 blocked negatives (random_state=42) with categories
  threshold_sensitivity.json — recall/FPR when each threshold is perturbed by ±1
  stage_contribution.json    — % of L2/L3 steps from stage-1 vs stage-2

Usage:
    PYTHONPATH=. python3 scripts/run_diagnostics.py [--tau-d 10.0 --tau-pi 6]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter

import pandas as pd

from irrgate.config import load_settings


# ---------------------------------------------------------------------------
# Blocking predicate
# ---------------------------------------------------------------------------

def _blocked(row, tau_d: float, tau_pi: int) -> bool:
    return row.get("f", 0) == 1 and (
        (row.get("d_I") or 0) >= tau_d or (row.get("irr_pos") or 0) >= tau_pi
    )


# ---------------------------------------------------------------------------
# 1. Per-benchmark × per-model breakdown
# ---------------------------------------------------------------------------

def per_benchmark_model(profiles: pd.DataFrame, tau_d: float, tau_pi: int) -> dict:
    result = {}
    for (bench, model), grp in profiles.groupby(["benchmark", "model"]):
        pos = grp[grp["is_positive"]]
        neg = grp[~grp["is_positive"]]
        tp = sum(_blocked(r, tau_d, tau_pi) for r in pos.to_dict("records"))
        fp = sum(_blocked(r, tau_d, tau_pi) for r in neg.to_dict("records"))
        result[f"{bench}/{model}"] = {
            "n_pos": len(pos), "n_neg": len(neg),
            "tp": tp, "fp": fp,
            "recall": tp / len(pos) if len(pos) > 0 else None,
            "fpr": fp / len(neg) if len(neg) > 0 else None,
        }
    return result


# ---------------------------------------------------------------------------
# 2. False negative enumeration
# ---------------------------------------------------------------------------

def enumerate_false_negatives(
    profiles: pd.DataFrame, cache: pd.DataFrame, tau_d: float, tau_pi: int
) -> list[dict]:
    pos = profiles[profiles["is_positive"]]
    fns = []
    for r in pos.to_dict("records"):
        if _blocked(r, tau_d, tau_pi):
            continue
        reason = "f=0_classifier_gap" if r["f"] == 0 else "f=1_below_thresholds"
        traj_rows = cache[cache["trajectory_id"] == r["trajectory_id"]]
        level_counts = traj_rows["final_level"].value_counts().to_dict()
        stage_counts = (traj_rows["stage_used"].value_counts().to_dict()
                        if "stage_used" in traj_rows.columns else {})
        fns.append({
            "trajectory_id": r["trajectory_id"],
            "task_id": r["task_id"],
            "model": r["model"],
            "benchmark": r["benchmark"],
            "f": int(r["f"]),
            "n_steps": r["n_steps"],
            "side_effect_step": r.get("side_effect_step"),
            "d_I": r["d_I"],
            "irr_pos": int(r["irr_pos"]),
            "d_I_at_side_effect_step": r.get("d_I_at_side_effect_step"),
            "level_counts": {f"L{k}": int(v) for k, v in level_counts.items()},
            "stage_counts": {str(k): int(v) for k, v in stage_counts.items()},
            "fn_reason": reason,
        })
    return fns


# ---------------------------------------------------------------------------
# 3. FP sample (blocked negatives)
# ---------------------------------------------------------------------------

_FP_SAMPLE_N = 30


def _heuristic_category(r: dict, level_counts: dict) -> str:
    has_L3 = level_counts.get(3, 0) > 0
    has_L2 = level_counts.get(2, 0) > 0
    n_steps = r["n_steps"]
    if has_L3 and n_steps <= 5 and level_counts.get(3, 0) == 1:
        return "a_short_single_L3"
    elif has_L3 and level_counts.get(3, 0) >= 3:
        return "b_multi_L3"
    elif has_L2 and not has_L3:
        return "c_L2_only"
    return "d_other"


def sample_false_positives(
    profiles: pd.DataFrame, cache: pd.DataFrame, tau_d: float, tau_pi: int,
    seed: int = 42,
) -> list[dict]:
    neg = profiles[~profiles["is_positive"]]
    fps = [r for r in neg.to_dict("records") if _blocked(r, tau_d, tau_pi)]
    rng = random.Random(seed)
    sample = rng.sample(fps, min(_FP_SAMPLE_N, len(fps)))
    annotated = []
    for r in sample:
        traj_rows = cache[cache["trajectory_id"] == r["trajectory_id"]]
        level_counts = traj_rows["final_level"].value_counts().to_dict()
        annotated.append({
            "trajectory_id": r["trajectory_id"],
            "task_id": r["task_id"],
            "model": r["model"],
            "benchmark": r["benchmark"],
            "n_steps": r["n_steps"],
            "d_I": r["d_I"],
            "irr_pos": int(r["irr_pos"]),
            "level_counts": {f"L{k}": int(v) for k, v in level_counts.items()},
            "heuristic_category": _heuristic_category(r, level_counts),
        })
    return annotated


# ---------------------------------------------------------------------------
# 4. Threshold sensitivity
# ---------------------------------------------------------------------------

def threshold_sensitivity(
    profiles: pd.DataFrame, tau_d: float, tau_pi: int
) -> dict:
    """Report recall/FPR when each threshold is perturbed by ±1 step."""
    pos = profiles[profiles["is_positive"]].to_dict("records")
    neg = profiles[~profiles["is_positive"]].to_dict("records")
    n_pos, n_neg = len(pos), len(neg)
    configs = {
        "baseline":          (tau_d,               tau_pi),
        "tau_d_minus_1":     (max(0.0, tau_d - 1.0), tau_pi),
        "tau_d_plus_1":      (tau_d + 1.0,         tau_pi),
        "tau_pi_minus_1": (tau_d,                max(1, tau_pi - 1)),
        "tau_pi_plus_1":  (tau_d,                tau_pi + 1),
    }
    results = {}
    for label, (td, tp) in configs.items():
        blocked_pos = sum(1 for r in pos if _blocked(r, td, tp))
        blocked_neg = sum(1 for r in neg if _blocked(r, td, tp))
        results[label] = {
            "tau_d": td, "tau_pi": tp,
            "recall": blocked_pos / n_pos if n_pos else 0.0,
            "fpr": blocked_neg / n_neg if n_neg else 0.0,
            "tp": blocked_pos, "fp": blocked_neg,
            "n_pos": n_pos, "n_neg": n_neg,
        }
    return results


# ---------------------------------------------------------------------------
# 5. Stage contribution
# ---------------------------------------------------------------------------

def stage_contribution(cache: pd.DataFrame) -> dict:
    l2l3 = cache[cache["final_level"] >= 2]
    total = len(l2l3)
    if total == 0:
        return {"total_L2L3_steps": 0, "stage1_pct": 0.0, "stage2_pct": 0.0}
    stage_counts = l2l3["stage_used"].value_counts().to_dict()
    s1 = int(stage_counts.get(1, 0))
    s2 = int(stage_counts.get(2, 0))
    return {
        "total_L2L3_steps": total,
        "stage1_count": s1, "stage1_pct": s1 / total,
        "stage2_count": s2, "stage2_pct": s2 / total,
        "other_count": total - s1 - s2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    s = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles",   default="results/profiles/profiles.parquet")
    parser.add_argument("--cache",      default="data/classification_cache.parquet")
    parser.add_argument("--cv-results", default="results/cv/cv_results.json",
                        help="CV results JSON to read selected thresholds from")
    parser.add_argument("--tau-d",     type=float, default=None,
                        help="Override tau_d (default: modal CV-selected value)")
    parser.add_argument("--tau-pi", type=int,   default=None,
                        help="Override tau_pi (default: modal CV-selected value)")
    parser.add_argument("--output-dir", default="results/diagnostics")
    args = parser.parse_args()

    tau_d     = args.tau_d
    tau_pi = args.tau_pi

    if os.path.exists(args.cv_results):
        with open(args.cv_results) as fh:
            cv = json.load(fh)
        if "irrgate" in cv:
            counts = cv["irrgate"].get("tau_selection_counts", {})
            if tau_d is None and counts.get("tau_d_counts"):
                tau_d = float(max(counts["tau_d_counts"], key=counts["tau_d_counts"].get))
            if tau_pi is None and counts.get("tau_pi_counts"):
                tau_pi = int(max(counts["tau_pi_counts"], key=counts["tau_pi_counts"].get))

    if tau_d     is None: tau_d     = float(s.get("tau_d", 5.0))
    if tau_pi is None: tau_pi = int(s.get("tau_pi", 5))

    print(f"[diagnostics] thresholds: tau_d={tau_d}  tau_pi={tau_pi}")

    profiles = pd.read_parquet(args.profiles)
    cache    = pd.read_parquet(args.cache)
    print(f"[diagnostics] loaded {len(profiles)} trajectories, {len(cache)} cache rows")

    os.makedirs(args.output_dir, exist_ok=True)

    print("[diagnostics] 1. per-benchmark/model breakdown...")
    bm = per_benchmark_model(profiles, tau_d, tau_pi)
    p = os.path.join(args.output_dir, "per_benchmark_model.json")
    with open(p, "w") as fh:
        json.dump(bm, fh, indent=2)
    print(f"  → {p} ({len(bm)} groups)")

    print("[diagnostics] 2. false negative enumeration...")
    fns = enumerate_false_negatives(profiles, cache, tau_d, tau_pi)
    p = os.path.join(args.output_dir, "false_negatives.json")
    with open(p, "w") as fh:
        json.dump(fns, fh, indent=2)
    fn_f0    = sum(1 for x in fns if x["fn_reason"] == "f=0_classifier_gap")
    fn_below = sum(1 for x in fns if x["fn_reason"] == "f=1_below_thresholds")
    print(f"  → {p}  ({len(fns)} FNs: f=0 gap={fn_f0}, f=1 below-threshold={fn_below})")

    print("[diagnostics] 3. FP sample...")
    fps = sample_false_positives(profiles, cache, tau_d, tau_pi)
    p = os.path.join(args.output_dir, "fp_sample.json")
    with open(p, "w") as fh:
        json.dump(fps, fh, indent=2)
    cat_counts = Counter(x["heuristic_category"] for x in fps)
    print(f"  → {p}  ({len(fps)} FPs sampled)")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"     {cat}: {cnt}")

    print("[diagnostics] 4. threshold sensitivity...")
    ts = threshold_sensitivity(profiles, tau_d, tau_pi)
    p = os.path.join(args.output_dir, "threshold_sensitivity.json")
    with open(p, "w") as fh:
        json.dump(ts, fh, indent=2)
    print(f"  → {p}")
    for label, row in ts.items():
        print(f"     {label:<22}  τ_d={row['tau_d']} τ_pi={row['tau_pi']}"
              f"  recall={row['recall']:.3f}  fpr={row['fpr']:.3f}")

    print("[diagnostics] 5. stage contribution...")
    sc = stage_contribution(cache)
    p = os.path.join(args.output_dir, "stage_contribution.json")
    with open(p, "w") as fh:
        json.dump(sc, fh, indent=2)
    print(f"  → {p}")
    print(f"     L2/L3 steps: {sc['total_L2L3_steps']}"
          f"  stage-1: {sc['stage1_count']} ({sc['stage1_pct']:.1%})"
          f"  stage-2: {sc['stage2_count']} ({sc['stage2_pct']:.1%})")

    print("\n[diagnostics] done.")


if __name__ == "__main__":
    main()
