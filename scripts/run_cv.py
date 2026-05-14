"""5×5 repeated stratified cross-validation for IrrGate threshold selection.

Pre-commitment (written before any tuning):
  Procedure : 5×5 repeated stratified CV
  Strata    : benchmark × is_positive × model
  Criterion : maximize recall subject to FPR ≤ 0.10
  Grid      : tau_d ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
               tau_pi ∈ {0.10, 0.20, 0.30, 0.40, 0.50}

Reads: results/profiles/profiles.parquet (output of scripts/compute_profiles.py)
Writes: results/cv/cv_results.json

Usage:
    PYTHONPATH=. python3 scripts/run_cv.py [--profiles results/profiles/profiles.parquet]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

TAU_D_VALUES  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
TAU_PI_VALUES = [0.10, 0.20, 0.30, 0.40, 0.50]

N_FOLDS  = 5
N_REPEATS = 5  # 5 seeds × 5 folds = 25 splits


# ---------------------------------------------------------------------------
# Gate simulation from peak profiles
# ---------------------------------------------------------------------------

def _simulate_disjunction(row: dict, tau_d: float, tau_pi: float) -> bool:
    """True iff the trajectory would be blocked by the disjunction policy."""
    if (row["peak_d_I"] or 0) <= 0:
        return False
    return (row["peak_d_I"] or 0) >= tau_d or (row["peak_pi"] or 0) >= tau_pi


def _simulate_f_only(row: dict, **_) -> bool:
    return (row["peak_d_I"] or 0) > 0


def _simulate_f_plus_d(row: dict, tau_d: float, **_) -> bool:
    return (row["peak_d_I"] or 0) > 0 and (row["peak_d_I"] or 0) >= tau_d


def _simulate_f_plus_pi(row: dict, tau_pi: float, **_) -> bool:
    return (row["peak_d_I"] or 0) > 0 and (row["peak_pi"] or 0) >= tau_pi


def _simulate_conjunction(row: dict, tau_d: float, tau_pi: float) -> bool:
    return (
        (row["peak_d_I"] or 0) > 0
        and (row["peak_d_I"] or 0) >= tau_d
        and (row["peak_pi"] or 0) >= tau_pi
    )


_VARIANTS = {
    "f_only":       _simulate_f_only,
    "f_plus_d":     _simulate_f_plus_d,
    "f_plus_pi":    _simulate_f_plus_pi,
    "disjunction":  _simulate_disjunction,
    "conjunction":  _simulate_conjunction,
}


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _stratum(row: dict) -> str:
    return f"{row['benchmark']}|{int(row['is_positive'])}|{row['model']}"


def stratified_kfold(records: list[dict], k: int, seed: int) -> list[tuple[list[dict], list[dict]]]:
    """Return k (tuning, held_out) pairs via stratified k-fold."""
    rng = random.Random(seed)
    strata: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        strata[_stratum(r)].append(r)

    # Shuffle each stratum independently
    for lst in strata.values():
        rng.shuffle(lst)

    # Round-robin assign fold indices within each stratum
    fold_indices: list[list[dict]] = [[] for _ in range(k)]
    for lst in strata.values():
        for i, r in enumerate(lst):
            fold_indices[i % k].append(r)

    splits = []
    for held_fold in range(k):
        held_out = fold_indices[held_fold]
        tuning = [r for i, fold in enumerate(fold_indices) if i != held_fold for r in fold]
        splits.append((tuning, held_out))
    return splits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _evaluate(records: list[dict], variant_fn, tau_d: float, tau_pi: float) -> dict:
    pos = [r for r in records if r["is_positive"]]
    neg = [r for r in records if not r["is_positive"]]
    tp = sum(1 for r in pos if variant_fn(r, tau_d=tau_d, tau_pi=tau_pi))
    fp = sum(1 for r in neg if variant_fn(r, tau_d=tau_d, tau_pi=tau_pi))
    recall = tp / len(pos) if pos else 0.0
    fpr    = fp / len(neg) if neg else 0.0
    return {"tp": tp, "fp": fp, "n_pos": len(pos), "n_neg": len(neg),
            "recall": recall, "fpr": fpr}


# ---------------------------------------------------------------------------
# Grid search: maximize recall subject to FPR ≤ FPR_BUDGET
# ---------------------------------------------------------------------------

FPR_BUDGET = 0.10


def select_threshold(tuning: list[dict]) -> tuple[float, float]:
    """Pick (tau_d, tau_pi) that maximizes recall on tuning set with FPR ≤ 0.10.

    If no config meets the FPR budget, return the config with the lowest FPR.
    Tie-break on recall: prefer higher recall; then on FPR: prefer lower FPR;
    then on tau_d: prefer larger (more conservative) threshold.
    """
    best: Optional[tuple[float, float]] = None
    best_recall = -1.0
    best_fpr    = float("inf")

    for tau_d in TAU_D_VALUES:
        for tau_pi in TAU_PI_VALUES:
            m = _evaluate(tuning, _simulate_disjunction, tau_d, tau_pi)
            recall, fpr = m["recall"], m["fpr"]
            meets_budget = fpr <= FPR_BUDGET

            if best is None:
                best = (tau_d, tau_pi)
                best_recall = recall
                best_fpr    = fpr
                continue

            current_meets = best_fpr <= FPR_BUDGET

            if meets_budget and not current_meets:
                best = (tau_d, tau_pi); best_recall = recall; best_fpr = fpr
            elif meets_budget and current_meets:
                if (recall, -fpr, tau_d) > (best_recall, -best_fpr, best[0]):
                    best = (tau_d, tau_pi); best_recall = recall; best_fpr = fpr
            elif not meets_budget and not current_meets:
                if (-fpr, recall, tau_d) > (-best_fpr, best_recall, best[0]):
                    best = (tau_d, tau_pi); best_recall = recall; best_fpr = fpr

    return best


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default="results/profiles/profiles.parquet")
    parser.add_argument("--output-dir", default="results/cv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.profiles)
    records = df.to_dict("records")
    print(f"[cv] loaded {len(records)} trajectories  "
          f"({sum(r['is_positive'] for r in records)} pos, "
          f"{sum(not r['is_positive'] for r in records)} neg)")

    # Per-split results
    splits_data: list[dict] = []
    # Pooled held-out counts (for Wilson CIs across all held-out data)
    pooled_tp: dict[str, int] = {v: 0 for v in _VARIANTS}
    pooled_fp: dict[str, int] = {v: 0 for v in _VARIANTS}
    pooled_n_pos: int = 0
    pooled_n_neg: int = 0

    for seed in range(N_REPEATS):
        splits = stratified_kfold(records, N_FOLDS, seed=seed)
        for fold, (tuning, held_out) in enumerate(splits):
            tau_d, tau_pi = select_threshold(tuning)
            tuning_m   = _evaluate(tuning,   _simulate_disjunction, tau_d, tau_pi)
            split_result = {
                "seed": seed, "fold": fold,
                "tau_d_selected": tau_d, "tau_pi_selected": tau_pi,
                "tuning_recall": tuning_m["recall"], "tuning_fpr": tuning_m["fpr"],
                "n_tuning_pos": tuning_m["n_pos"],   "n_tuning_neg": tuning_m["n_neg"],
                "held_out": {},
            }
            # Evaluate all variants on held-out
            for vname, vfn in _VARIANTS.items():
                m = _evaluate(held_out, vfn, tau_d, tau_pi)
                split_result["held_out"][vname] = m
                pooled_tp[vname] += m["tp"]
                pooled_fp[vname] += m["fp"]

            # Track held-out size (same across all variants)
            ho_pos = sum(1 for r in held_out if r["is_positive"])
            ho_neg = sum(1 for r in held_out if not r["is_positive"])
            pooled_n_pos += ho_pos
            pooled_n_neg += ho_neg

            print(
                f"[cv] seed={seed} fold={fold}  τ_d={tau_d:.2f} τ_π={tau_pi:.2f}"
                f"  tune recall={tuning_m['recall']:.3f} fpr={tuning_m['fpr']:.3f}"
                f"  held disj recall={split_result['held_out']['disjunction']['recall']:.3f}"
                f" fpr={split_result['held_out']['disjunction']['fpr']:.3f}"
            )
            splits_data.append(split_result)

    # Aggregate across splits
    def _across_splits(variant: str, key: str) -> list[float]:
        return [s["held_out"][variant][key] for s in splits_data]

    def _percentile(vals: list[float], p: float) -> float:
        sv = sorted(vals)
        idx = p / 100 * (len(sv) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
        return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])

    summary: dict[str, dict] = {}
    for vname in _VARIANTS:
        recalls = _across_splits(vname, "recall")
        fprs    = _across_splits(vname, "fpr")
        # Pooled Wilson CIs (treating all held-out observations as iid)
        recall_lo, recall_hi = wilson_ci(pooled_tp[vname], pooled_n_pos)
        fpr_lo,    fpr_hi    = wilson_ci(pooled_fp[vname], pooled_n_neg)
        summary[vname] = {
            "pooled_recall":    pooled_tp[vname] / pooled_n_pos if pooled_n_pos else 0.0,
            "pooled_fpr":       pooled_fp[vname] / pooled_n_neg if pooled_n_neg else 0.0,
            "recall_wilson_lo": recall_lo, "recall_wilson_hi": recall_hi,
            "fpr_wilson_lo":    fpr_lo,    "fpr_wilson_hi":    fpr_hi,
            "recall_p25":  _percentile(recalls, 25),
            "recall_p75":  _percentile(recalls, 75),
            "fpr_p25":     _percentile(fprs, 25),
            "fpr_p75":     _percentile(fprs, 75),
            "pooled_tp": pooled_tp[vname],
            "pooled_fp": pooled_fp[vname],
            "pooled_n_pos": pooled_n_pos,
            "pooled_n_neg": pooled_n_neg,
        }

    # Tau distribution
    tau_d_counts: dict[str, int]  = defaultdict(int)
    tau_pi_counts: dict[str, int] = defaultdict(int)
    for s in splits_data:
        tau_d_counts[str(s["tau_d_selected"])]  += 1
        tau_pi_counts[str(s["tau_pi_selected"])] += 1

    output = {
        "procedure": {
            "n_folds": N_FOLDS, "n_repeats": N_REPEATS,
            "fpr_budget": FPR_BUDGET,
            "tau_d_grid": TAU_D_VALUES,
            "tau_pi_grid": TAU_PI_VALUES,
            "strata": "benchmark × is_positive × model",
            "criterion": "maximize recall subject to FPR <= 0.10",
        },
        "summary_per_variant": summary,
        "tau_d_selection_counts":  dict(tau_d_counts),
        "tau_pi_selection_counts": dict(tau_pi_counts),
        "splits": splits_data,
    }

    out_path = os.path.join(args.output_dir, "cv_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[cv] results saved → {out_path}")

    print("\n=== CV Summary ===")
    print(f"{'Variant':<15} {'Recall':>8} {'95% CI':>18} {'FPR':>8} {'95% CI':>18}")
    for vname, s in summary.items():
        print(
            f"{vname:<15}"
            f"  {s['pooled_recall']:.3f}"
            f"  [{s['recall_wilson_lo']:.3f}, {s['recall_wilson_hi']:.3f}]"
            f"  {s['pooled_fpr']:.3f}"
            f"  [{s['fpr_wilson_lo']:.3f}, {s['fpr_wilson_hi']:.3f}]"
        )

    print("\n=== Selected tau_d distribution (across 25 splits) ===")
    for td, cnt in sorted(tau_d_counts.items()):
        print(f"  tau_d={td}: {cnt}/25 splits")
    print("=== Selected tau_pi distribution ===")
    for tp, cnt in sorted(tau_pi_counts.items()):
        print(f"  tau_pi={tp}: {cnt}/25 splits")


if __name__ == "__main__":
    main()
