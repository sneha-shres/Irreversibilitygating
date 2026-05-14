"""Phase 3 diagnostics for IrrGate.

Reads: results/profiles/profiles.parquet + data/classification_cache.parquet
       + cv_results.json (for the selected threshold)
Writes: results/diagnostics/

Outputs:
  1. per_benchmark_model.json    — recall/FPR breakdown for disjunction
  2. false_negatives.json        — every missed positive with full annotation
  3. fp_sample.json              — 30 blocked negatives labelled by category
  4. alpha_sensitivity.json      — disjunction recall/FPR at α ∈ {0.25, 0.5, 0.75, 1.0}
  5. stage_contribution.json     — % L2/L3 from stage-1 vs stage-2

Usage:
    PYTHONPATH=. python3 scripts/run_diagnostics.py [--tau-d 0.05 --tau-pi 0.10]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import pandas as pd

from irrgate.config import load_settings
from irrgate.taxonomy import severity_weight, Level


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _blocked_disjunction(row, tau_d: float, tau_pi: float) -> bool:
    if (row["peak_d_I"] or 0) <= 0:
        return False
    return (row["peak_d_I"] or 0) >= tau_d or (row["peak_pi"] or 0) >= tau_pi


def _compute_profiles_alpha(
    cache_rows: pd.DataFrame, alpha: float
) -> tuple[float, float]:
    """Re-compute peak_d_I and peak_pi for one trajectory using a different alpha."""
    steps = cache_rows.sort_values("step_index")
    levels = [Level(v) for v in steps["final_level"].tolist()]
    target_bids = steps["target_bid"].tolist()

    if not levels:
        return 0.0, 0.0

    severity_values = [severity_weight(lv, alpha) for lv in levels]
    peak_d_I = 0.0
    peak_pi  = 0.0
    seen_bids: set[str] = set()
    distinct_bids = [b for b in target_bids if b and not pd.isna(b)]
    n_distinct = len(set(distinct_bids))

    for k in range(len(levels)):
        sevs_k = severity_values[: k + 1]
        d_I_k = sum(sevs_k) / len(sevs_k)
        total_w = sum(sevs_k)
        if d_I_k > peak_d_I:
            peak_d_I = d_I_k

        bid_k = target_bids[k] if not pd.isna(target_bids[k]) else None
        bid_term = (len(seen_bids) / n_distinct) if n_distinct > 0 else 1.0
        residual = severity_values[k] * (1.0 - bid_term)

        # Accumulate residuals for pi
        num = sum(
            severity_values[j] * (1.0 - (
                (len(set(target_bids[:j])) / n_distinct) if n_distinct > 0 else 1.0
            ))
            for j in range(k + 1)
        )
        pi_k = num / total_w if total_w > 0 else 0.0
        if pi_k > peak_pi:
            peak_pi = pi_k

        if bid_k is not None:
            seen_bids.add(bid_k)

    return peak_d_I, peak_pi


# ---------------------------------------------------------------------------
# 1. Per-benchmark × per-model breakdown
# ---------------------------------------------------------------------------

def per_benchmark_model(profiles: pd.DataFrame, tau_d: float, tau_pi: float) -> dict:
    result = {}
    for (bench, model), grp in profiles.groupby(["benchmark", "model"]):
        pos = grp[grp["is_positive"]]
        neg = grp[~grp["is_positive"]]
        tp = sum(_blocked_disjunction(r, tau_d, tau_pi) for r in pos.to_dict("records"))
        fp = sum(_blocked_disjunction(r, tau_d, tau_pi) for r in neg.to_dict("records"))
        recall = tp / len(pos) if len(pos) > 0 else None
        fpr    = fp / len(neg) if len(neg) > 0 else None
        result[f"{bench}/{model}"] = {
            "n_pos": len(pos), "n_neg": len(neg),
            "tp": tp, "fp": fp, "recall": recall, "fpr": fpr,
        }
    return result


# ---------------------------------------------------------------------------
# 2. False negative enumeration
# ---------------------------------------------------------------------------

def enumerate_false_negatives(
    profiles: pd.DataFrame, cache: pd.DataFrame, tau_d: float, tau_pi: float
) -> list[dict]:
    pos = profiles[profiles["is_positive"]]
    fns = []
    for r in pos.to_dict("records"):
        if _blocked_disjunction(r, tau_d, tau_pi):
            continue  # correctly blocked → TP
        # Determine reason for miss
        if r["f"] == 0:
            reason = "f=0_classifier_gap"
        else:
            reason = "f=1_below_thresholds"
        # Get per-step level breakdown
        traj_rows = cache[cache["trajectory_id"] == r["trajectory_id"]]
        level_counts = traj_rows["final_level"].value_counts().to_dict()
        stage_counts = traj_rows["stage_used"].value_counts().to_dict() if "stage_used" in traj_rows.columns else {}
        fns.append({
            "trajectory_id": r["trajectory_id"],
            "task_id": r["task_id"],
            "model": r["model"],
            "benchmark": r["benchmark"],
            "n_steps": r["n_steps"],
            "side_effect_step": r["side_effect_step"],
            "peak_d_I": r["peak_d_I"],
            "peak_pi": r["peak_pi"],
            "d_I_at_side_effect_step": r["d_I_at_side_effect_step"],
            "pi_at_side_effect_step": r["pi_at_side_effect_step"],
            "level_counts": {f"L{k}": int(v) for k, v in level_counts.items()},
            "stage_counts": {str(k): int(v) for k, v in stage_counts.items()},
            "fn_reason": reason,
        })
    return fns


# ---------------------------------------------------------------------------
# 3. FP sample (blocked negatives)
# ---------------------------------------------------------------------------

_FP_SAMPLE_N = 30


def sample_false_positives(
    profiles: pd.DataFrame, cache: pd.DataFrame, tau_d: float, tau_pi: float,
    seed: int = 42,
) -> list[dict]:
    neg = profiles[~profiles["is_positive"]]
    fps = [r for r in neg.to_dict("records") if _blocked_disjunction(r, tau_d, tau_pi)]
    rng = random.Random(seed)
    sample = rng.sample(fps, min(_FP_SAMPLE_N, len(fps)))

    annotated = []
    for r in sample:
        traj_rows = cache[cache["trajectory_id"] == r["trajectory_id"]]
        level_counts = traj_rows["final_level"].value_counts().to_dict()
        has_L3 = level_counts.get(3, 0) > 0
        has_L2 = level_counts.get(2, 0) > 0
        n_steps = r["n_steps"]

        # Heuristic category:
        # (a) short trajectory, single L3 → high d_I from one action = over-cautious block
        # (b) many L2/L3 steps in a long trajectory → classifier may be over-firing
        # (c) otherwise → over-cautious block
        if has_L3 and n_steps <= 5 and level_counts.get(3, 0) == 1:
            category = "a_short_single_L3"
        elif has_L3 and level_counts.get(3, 0) >= 3:
            category = "b_multi_L3"
        elif has_L2 and not has_L3:
            category = "c_L2_only"
        else:
            category = "d_other"

        annotated.append({
            "trajectory_id": r["trajectory_id"],
            "task_id": r["task_id"],
            "model": r["model"],
            "benchmark": r["benchmark"],
            "n_steps": n_steps,
            "peak_d_I": r["peak_d_I"],
            "peak_pi": r["peak_pi"],
            "level_counts": {f"L{k}": int(v) for k, v in level_counts.items()},
            "heuristic_category": category,
        })
    return annotated


# ---------------------------------------------------------------------------
# 4. Alpha sensitivity
# ---------------------------------------------------------------------------

def alpha_sensitivity(
    profiles: pd.DataFrame, cache: pd.DataFrame,
    alpha_values: list[float], tau_d: float, tau_pi: float,
) -> list[dict]:
    cache_by_traj = {tid: grp for tid, grp in cache.groupby("trajectory_id")}
    results = []
    for alpha in alpha_values:
        tp = fp = 0
        n_pos = n_neg = 0
        for r in profiles.to_dict("records"):
            traj_rows = cache_by_traj.get(r["trajectory_id"])
            if traj_rows is None:
                continue
            peak_d, peak_pi = _compute_profiles_alpha(traj_rows, alpha)
            # disjunction: f=1 iff peak_d > 0
            blocked = peak_d > 0 and (peak_d >= tau_d or peak_pi >= tau_pi)
            if r["is_positive"]:
                n_pos += 1
                if blocked:
                    tp += 1
            else:
                n_neg += 1
                if blocked:
                    fp += 1
        results.append({
            "alpha": alpha,
            "tau_d": tau_d, "tau_pi": tau_pi,
            "recall": tp / n_pos if n_pos else 0.0,
            "fpr":    fp / n_neg if n_neg else 0.0,
            "tp": tp, "fp": fp, "n_pos": n_pos, "n_neg": n_neg,
        })
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
        "other_count":  total - s1 - s2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    s = load_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="results/profiles/profiles.parquet")
    parser.add_argument("--cache",    default="data/classification_cache.parquet")
    parser.add_argument("--cv-results", default="results/cv/cv_results.json",
                        help="CV results JSON to read selected tau_d/tau_pi from")
    parser.add_argument("--tau-d",  type=float, default=None,
                        help="Override tau_d (else read from cv_results or settings.json)")
    parser.add_argument("--tau-pi", type=float, default=None,
                        help="Override tau_pi")
    parser.add_argument("--output-dir", default="results/diagnostics")
    args = parser.parse_args()

    # Determine thresholds: CLI > cv_results > settings.json
    tau_d = args.tau_d
    tau_pi = args.tau_pi
    if (tau_d is None or tau_pi is None) and os.path.exists(args.cv_results):
        with open(args.cv_results) as fh:
            cv = json.load(fh)
        # Modal selection from CV
        tau_d_counts = cv.get("tau_d_selection_counts", {})
        tau_pi_counts = cv.get("tau_pi_selection_counts", {})
        if tau_d is None and tau_d_counts:
            tau_d = float(max(tau_d_counts, key=tau_d_counts.get))
        if tau_pi is None and tau_pi_counts:
            tau_pi = float(max(tau_pi_counts, key=tau_pi_counts.get))
    if tau_d is None:
        tau_d = s.get("tau_d", 0.15)
    if tau_pi is None:
        tau_pi = s.get("tau_pi", 0.30)

    print(f"[diagnostics] tau_d={tau_d}  tau_pi={tau_pi}")
    os.makedirs(args.output_dir, exist_ok=True)

    profiles = pd.read_parquet(args.profiles)
    cache    = pd.read_parquet(args.cache)
    print(f"[diagnostics] loaded {len(profiles)} trajectories, {len(cache)} cache rows")

    # 1. Per-benchmark/model
    print("[diagnostics] 1. per-benchmark/model breakdown...")
    bm = per_benchmark_model(profiles, tau_d, tau_pi)
    out1 = os.path.join(args.output_dir, "per_benchmark_model.json")
    with open(out1, "w") as fh:
        json.dump(bm, fh, indent=2)
    print(f"  → {out1} ({len(bm)} groups)")

    # 2. False negatives
    print("[diagnostics] 2. false negative enumeration...")
    fns = enumerate_false_negatives(profiles, cache, tau_d, tau_pi)
    out2 = os.path.join(args.output_dir, "false_negatives.json")
    with open(out2, "w") as fh:
        json.dump(fns, fh, indent=2)
    print(f"  → {out2}  ({len(fns)} FNs)")
    fn_f0    = sum(1 for x in fns if x["fn_reason"] == "f=0_classifier_gap")
    fn_below = sum(1 for x in fns if x["fn_reason"] == "f=1_below_thresholds")
    print(f"     f=0 gap: {fn_f0}   f=1 below-threshold: {fn_below}")

    # 3. FP sample
    print("[diagnostics] 3. FP sample...")
    fps = sample_false_positives(profiles, cache, tau_d, tau_pi)
    out3 = os.path.join(args.output_dir, "fp_sample.json")
    with open(out3, "w") as fh:
        json.dump(fps, fh, indent=2)
    from collections import Counter
    cat_counts = Counter(x["heuristic_category"] for x in fps)
    print(f"  → {out3}  ({len(fps)} FPs sampled)")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"     {cat}: {cnt}")

    # 4. Alpha sensitivity
    print("[diagnostics] 4. alpha sensitivity...")
    alpha_vals = [0.25, 0.5, 0.75, 1.0]
    alpha_res = alpha_sensitivity(profiles, cache, alpha_vals, tau_d, tau_pi)
    out4 = os.path.join(args.output_dir, "alpha_sensitivity.json")
    with open(out4, "w") as fh:
        json.dump(alpha_res, fh, indent=2)
    print(f"  → {out4}")
    for row in alpha_res:
        print(f"     α={row['alpha']}  recall={row['recall']:.3f}  fpr={row['fpr']:.3f}")

    # 5. Stage contribution
    print("[diagnostics] 5. stage contribution...")
    sc = stage_contribution(cache)
    out5 = os.path.join(args.output_dir, "stage_contribution.json")
    with open(out5, "w") as fh:
        json.dump(sc, fh, indent=2)
    print(f"  → {out5}")
    print(f"     L2/L3 steps: {sc['total_L2L3_steps']}"
          f"  stage-1: {sc['stage1_count']} ({sc['stage1_pct']:.1%})"
          f"  stage-2: {sc['stage2_count']} ({sc['stage2_pct']:.1%})")

    print("\n[diagnostics] done.")


if __name__ == "__main__":
    main()
