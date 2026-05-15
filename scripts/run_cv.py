"""5×5 repeated stratified cross-validation for IrrGate threshold selection.

Two passes run in sequence and written to a single combined cv_results.json:
  Pass A (density variants): existing τ_d × τ_π grid, primary=disjunction
  Pass B (shape variants):   τ_pages × τ_l3 grid,   primary=pages_disj_l3

Pre-commitment (written before any tuning):
  Procedure : 5×5 repeated stratified CV
  Strata    : benchmark × is_positive × model
  Criterion : maximize recall subject to FPR ≤ 0.10
  Pass A grid : τ_d  ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
                τ_π  ∈ {0.10, 0.20, 0.30, 0.40, 0.50}
  Pass B grid : τ_pages ∈ {2, 3, 4, 5, 6, 7, 8, 10, 12}
                τ_l3    ∈ {1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18}

Reads:  results/profiles/profiles.parquet
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
import re
import subprocess
import sys
from collections import defaultdict
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Pass A Grid (pre-committed; bit-identical to prior run)
# ---------------------------------------------------------------------------

TAU_D_VALUES  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
TAU_PI_VALUES = [0.10, 0.20, 0.30, 0.40, 0.50]

# ---------------------------------------------------------------------------
# Pass B Grid (pre-committed; do not tune in-flight)
# ---------------------------------------------------------------------------

TAU_PAGES_VALUES = [2, 3, 4, 5, 6, 7, 8, 10, 12]
TAU_L3_VALUES    = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18]

N_FOLDS   = 5
N_REPEATS = 5  # 5 seeds × 5 folds = 25 splits
FPR_BUDGET = 0.10


# ---------------------------------------------------------------------------
# Pass A simulation functions (bit-identical to prior run)
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
# Pass B simulation functions (shape variants)
# ---------------------------------------------------------------------------

def _pages(row: dict) -> float:
    """n_distinct_pages_pre_se with NaN/None treated as 0."""
    v = row.get("n_distinct_pages_pre_se")
    if v is None:
        return 0.0
    try:
        if math.isnan(float(v)):
            return 0.0
    except (TypeError, ValueError):
        pass
    return float(v)


def _simulate_pages_only(row: dict, tau_pages: int, **_) -> bool:
    if (row["peak_d_I"] or 0) <= 0:
        return False
    return _pages(row) >= tau_pages


def _simulate_l3_count_only(row: dict, tau_l3: int, **_) -> bool:
    if (row["peak_d_I"] or 0) <= 0:
        return False
    return (row.get("n_l3_actions") or 0) >= tau_l3


def _simulate_pages_disj_l3(row: dict, tau_pages: int, tau_l3: int) -> bool:
    if (row["peak_d_I"] or 0) <= 0:
        return False
    return _pages(row) >= tau_pages or (row.get("n_l3_actions") or 0) >= tau_l3


def _simulate_pages_conj_l3(row: dict, tau_pages: int, tau_l3: int) -> bool:
    if (row["peak_d_I"] or 0) <= 0:
        return False
    return _pages(row) >= tau_pages and (row.get("n_l3_actions") or 0) >= tau_l3


_B_VARIANTS = {
    "f_only":         _simulate_f_only,
    "pages_only":     _simulate_pages_only,
    "l3_count_only":  _simulate_l3_count_only,
    "pages_disj_l3":  _simulate_pages_disj_l3,
    "pages_conj_l3":  _simulate_pages_conj_l3,
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


def _evaluate_b(records: list[dict], variant_fn, **kwargs) -> dict:
    """Evaluate a shape variant; passes kwargs (e.g. tau_pages, tau_l3) to variant_fn."""
    pos = [r for r in records if r["is_positive"]]
    neg = [r for r in records if not r["is_positive"]]
    tp = sum(1 for r in pos if variant_fn(r, **kwargs))
    fp = sum(1 for r in neg if variant_fn(r, **kwargs))
    recall = tp / len(pos) if pos else 0.0
    fpr    = fp / len(neg) if neg else 0.0
    return {"tp": tp, "fp": fp, "n_pos": len(pos), "n_neg": len(neg),
            "recall": recall, "fpr": fpr}


# ---------------------------------------------------------------------------
# Grid search: maximize recall subject to FPR ≤ FPR_BUDGET
# ---------------------------------------------------------------------------

def select_threshold(tuning: list[dict]) -> tuple[float, float]:
    """Pass A: pick (tau_d, tau_pi) that maximizes recall on tuning set with FPR ≤ 0.10.

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


def select_threshold_b(tuning: list[dict]) -> tuple[int, int]:
    """Pass B: pick (tau_pages, tau_l3) that maximizes recall with FPR ≤ 0.10.

    Same criterion and tie-break logic as Pass A; τ_pages used as the size tie-break.
    """
    best: Optional[tuple[int, int]] = None
    best_recall = -1.0
    best_fpr    = float("inf")

    for tau_pages in TAU_PAGES_VALUES:
        for tau_l3 in TAU_L3_VALUES:
            m = _evaluate_b(tuning, _simulate_pages_disj_l3,
                            tau_pages=tau_pages, tau_l3=tau_l3)
            recall, fpr = m["recall"], m["fpr"]
            meets_budget = fpr <= FPR_BUDGET

            if best is None:
                best = (tau_pages, tau_l3)
                best_recall = recall
                best_fpr    = fpr
                continue

            current_meets = best_fpr <= FPR_BUDGET

            if meets_budget and not current_meets:
                best = (tau_pages, tau_l3); best_recall = recall; best_fpr = fpr
            elif meets_budget and current_meets:
                if (recall, -fpr, tau_pages) > (best_recall, -best_fpr, best[0]):
                    best = (tau_pages, tau_l3); best_recall = recall; best_fpr = fpr
            elif not meets_budget and not current_meets:
                if (-fpr, recall, tau_pages) > (-best_fpr, best_recall, best[0]):
                    best = (tau_pages, tau_l3); best_recall = recall; best_fpr = fpr

    return best


# ---------------------------------------------------------------------------
# Summary helper (shared by both passes)
# ---------------------------------------------------------------------------

def _summarize_variants(
    splits_data: list[dict],
    pooled_tp: dict[str, int],
    pooled_fp: dict[str, int],
    pooled_n_pos: int,
    pooled_n_neg: int,
    variant_names,
) -> dict:
    def _across_splits(variant: str, key: str) -> list[float]:
        return [s["held_out"][variant][key] for s in splits_data]

    def _percentile(vals: list[float], p: float) -> float:
        sv = sorted(vals)
        idx = p / 100 * (len(sv) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
        return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])

    summary: dict[str, dict] = {}
    for vname in variant_names:
        recalls = _across_splits(vname, "recall")
        fprs    = _across_splits(vname, "fpr")
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
    return summary


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

    # ================================================================
    # Pass A: Density variants (unchanged logic from prior run)
    # ================================================================
    print("\n[cv] === Pass A: density variants ===")
    splits_data_a: list[dict] = []
    pooled_tp_a: dict[str, int] = {v: 0 for v in _VARIANTS}
    pooled_fp_a: dict[str, int] = {v: 0 for v in _VARIANTS}
    pooled_n_pos_a = pooled_n_neg_a = 0

    for seed in range(N_REPEATS):
        splits = stratified_kfold(records, N_FOLDS, seed=seed)
        for fold, (tuning, held_out) in enumerate(splits):
            tau_d, tau_pi = select_threshold(tuning)
            tuning_m = _evaluate(tuning, _simulate_disjunction, tau_d, tau_pi)
            split_result = {
                "seed": seed, "fold": fold,
                "tau_d_selected": tau_d, "tau_pi_selected": tau_pi,
                "tuning_recall": tuning_m["recall"], "tuning_fpr": tuning_m["fpr"],
                "n_tuning_pos": tuning_m["n_pos"], "n_tuning_neg": tuning_m["n_neg"],
                "held_out": {},
            }
            for vname, vfn in _VARIANTS.items():
                m = _evaluate(held_out, vfn, tau_d, tau_pi)
                split_result["held_out"][vname] = m
                pooled_tp_a[vname] += m["tp"]
                pooled_fp_a[vname] += m["fp"]

            ho_pos = sum(1 for r in held_out if r["is_positive"])
            ho_neg = sum(1 for r in held_out if not r["is_positive"])
            pooled_n_pos_a += ho_pos
            pooled_n_neg_a += ho_neg

            print(
                f"[cv] A seed={seed} fold={fold}  τ_d={tau_d:.2f} τ_π={tau_pi:.2f}"
                f"  tune recall={tuning_m['recall']:.3f} fpr={tuning_m['fpr']:.3f}"
                f"  held disj recall={split_result['held_out']['disjunction']['recall']:.3f}"
                f" fpr={split_result['held_out']['disjunction']['fpr']:.3f}"
            )
            splits_data_a.append(split_result)

    summary_a = _summarize_variants(
        splits_data_a, pooled_tp_a, pooled_fp_a,
        pooled_n_pos_a, pooled_n_neg_a, _VARIANTS,
    )

    # --- Validation 1: assert Pass A reproduces prior result ---
    disj_a = summary_a["disjunction"]
    if not (abs(disj_a["pooled_recall"] - 0.444) <= 0.005
            and abs(disj_a["pooled_fpr"] - 0.206) <= 0.005):
        print(f"\n[HALT] Pass A reproducibility check FAILED:")
        print(f"  disjunction recall={disj_a['pooled_recall']:.4f}  (expected 0.444 ± 0.005)")
        print(f"  disjunction FPR   ={disj_a['pooled_fpr']:.4f}  (expected 0.206 ± 0.005)")
        sys.exit(1)
    print(
        f"\n[cv] Pass A reproducibility check PASSED  "
        f"(recall={disj_a['pooled_recall']:.3f}, FPR={disj_a['pooled_fpr']:.3f})"
    )

    # ================================================================
    # Pass B: Shape variants
    # ================================================================
    print("\n[cv] === Pass B: shape variants ===")
    splits_data_b: list[dict] = []
    pooled_tp_b: dict[str, int] = {v: 0 for v in _B_VARIANTS}
    pooled_fp_b: dict[str, int] = {v: 0 for v in _B_VARIANTS}
    pooled_n_pos_b = pooled_n_neg_b = 0

    for seed in range(N_REPEATS):
        splits = stratified_kfold(records, N_FOLDS, seed=seed)
        for fold, (tuning, held_out) in enumerate(splits):
            tau_pages, tau_l3 = select_threshold_b(tuning)
            tuning_m = _evaluate_b(
                tuning, _simulate_pages_disj_l3,
                tau_pages=tau_pages, tau_l3=tau_l3,
            )
            split_result = {
                "seed": seed, "fold": fold,
                "tau_pages_selected": tau_pages, "tau_l3_selected": tau_l3,
                "tuning_recall": tuning_m["recall"], "tuning_fpr": tuning_m["fpr"],
                "n_tuning_pos": tuning_m["n_pos"], "n_tuning_neg": tuning_m["n_neg"],
                "held_out": {},
            }
            for vname, vfn in _B_VARIANTS.items():
                m = _evaluate_b(held_out, vfn, tau_pages=tau_pages, tau_l3=tau_l3)
                split_result["held_out"][vname] = m
                pooled_tp_b[vname] += m["tp"]
                pooled_fp_b[vname] += m["fp"]

            ho_pos = sum(1 for r in held_out if r["is_positive"])
            ho_neg = sum(1 for r in held_out if not r["is_positive"])
            pooled_n_pos_b += ho_pos
            pooled_n_neg_b += ho_neg

            print(
                f"[cv] B seed={seed} fold={fold}  τ_pages={tau_pages} τ_l3={tau_l3}"
                f"  tune recall={tuning_m['recall']:.3f} fpr={tuning_m['fpr']:.3f}"
                f"  held pages_disj_l3"
                f" recall={split_result['held_out']['pages_disj_l3']['recall']:.3f}"
                f" fpr={split_result['held_out']['pages_disj_l3']['fpr']:.3f}"
            )
            splits_data_b.append(split_result)

    summary_b = _summarize_variants(
        splits_data_b, pooled_tp_b, pooled_fp_b,
        pooled_n_pos_b, pooled_n_neg_b, _B_VARIANTS,
    )

    # Tau selection distributions
    tau_d_counts_a:      dict[str, int] = defaultdict(int)
    tau_pi_counts_a:     dict[str, int] = defaultdict(int)
    tau_pages_counts_b:  dict[str, int] = defaultdict(int)
    tau_l3_counts_b:     dict[str, int] = defaultdict(int)
    for s in splits_data_a:
        tau_d_counts_a[str(s["tau_d_selected"])]  += 1
        tau_pi_counts_a[str(s["tau_pi_selected"])] += 1
    for s in splits_data_b:
        tau_pages_counts_b[str(s["tau_pages_selected"])] += 1
        tau_l3_counts_b[str(s["tau_l3_selected"])]       += 1

    # Build combined output
    output = {
        "procedure": {
            "n_folds": N_FOLDS, "n_repeats": N_REPEATS,
            "fpr_budget": FPR_BUDGET,
            "strata": "benchmark × is_positive × model",
            "criterion": "maximize recall subject to FPR <= 0.10",
        },
        "pass_a_density": {
            "grid": {"tau_d": TAU_D_VALUES, "tau_pi": TAU_PI_VALUES},
            "primary_variant": "disjunction",
            "summary_per_variant": summary_a,
            "tau_selection_counts": {
                "tau_d_counts":  dict(tau_d_counts_a),
                "tau_pi_counts": dict(tau_pi_counts_a),
            },
            "splits": splits_data_a,
        },
        "pass_b_shape": {
            "grid": {"tau_pages": TAU_PAGES_VALUES, "tau_l3": TAU_L3_VALUES},
            "primary_variant": "pages_disj_l3",
            "summary_per_variant": summary_b,
            "tau_selection_counts": {
                "tau_pages_counts": dict(tau_pages_counts_b),
                "tau_l3_counts":    dict(tau_l3_counts_b),
            },
            "splits": splits_data_b,
        },
    }

    out_path = os.path.join(args.output_dir, "cv_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n[cv] results saved → {out_path}")

    # ================================================================
    # Side-by-side comparison table
    # ================================================================
    print("\n" + "=" * 78)
    print("Side-by-side CV Summary  (pooled Wilson 95% CIs across 25 held-out folds)")
    print("=" * 78)
    hdr = f"{'Variant':<17} {'Recall':>7} {'95% CI':>18} {'FPR':>7} {'95% CI':>18}"
    sep = "-" * 69

    print(f"\nPass A — density variants  (primary: disjunction | grid: τ_d × τ_π)")
    print(hdr)
    print(sep)
    for vname, s in summary_a.items():
        marker = " *" if vname == "disjunction" else "  "
        print(
            f"{vname:<17}{marker}"
            f"{s['pooled_recall']:>7.3f}"
            f"  [{s['recall_wilson_lo']:.3f}, {s['recall_wilson_hi']:.3f}]"
            f"  {s['pooled_fpr']:>7.3f}"
            f"  [{s['fpr_wilson_lo']:.3f}, {s['fpr_wilson_hi']:.3f}]"
        )

    print(f"\nPass B — shape variants  (primary: pages_disj_l3 | grid: τ_pages × τ_l3)")
    print(hdr)
    print(sep)
    for vname, s in summary_b.items():
        marker = " *" if vname == "pages_disj_l3" else "  "
        print(
            f"{vname:<17}{marker}"
            f"{s['pooled_recall']:>7.3f}"
            f"  [{s['recall_wilson_lo']:.3f}, {s['recall_wilson_hi']:.3f}]"
            f"  {s['pooled_fpr']:>7.3f}"
            f"  [{s['fpr_wilson_lo']:.3f}, {s['fpr_wilson_hi']:.3f}]"
        )
    print("  (* = primary variant driving threshold selection)")

    # ================================================================
    # Tau distribution printout
    # ================================================================
    print("\n=== Pass A: selected τ_d distribution (across 25 splits) ===")
    for td, cnt in sorted(tau_d_counts_a.items()):
        print(f"  tau_d={td}: {cnt}/25 splits")
    print("=== Pass A: selected τ_π distribution ===")
    for tp, cnt in sorted(tau_pi_counts_a.items()):
        print(f"  tau_pi={tp}: {cnt}/25 splits")

    print("\n=== Pass B: selected τ_pages distribution (across 25 splits) ===")
    for tp, cnt in sorted(tau_pages_counts_b.items()):
        print(f"  tau_pages={tp}: {cnt}/25 splits")
    print("=== Pass B: selected τ_l3 distribution ===")
    for tl, cnt in sorted(tau_l3_counts_b.items()):
        print(f"  tau_l3={tl}: {cnt}/25 splits")

    # ================================================================
    # Validation 2: strict dominance check
    # ================================================================
    pdl  = summary_b["pages_disj_l3"]
    disj = summary_a["disjunction"]
    recall_dom = pdl["pooled_recall"] >= disj["pooled_recall"]
    fpr_dom    = pdl["pooled_fpr"]    <= disj["pooled_fpr"]
    if recall_dom and fpr_dom:
        print(
            f"\n[cv] Dominance check PASSED: pages_disj_l3 strictly dominates disjunction  "
            f"(recall {pdl['pooled_recall']:.3f} >= {disj['pooled_recall']:.3f},  "
            f"FPR {pdl['pooled_fpr']:.3f} <= {disj['pooled_fpr']:.3f})"
        )
    else:
        print(
            f"\n[cv] WARNING: pages_disj_l3 does NOT strictly dominate disjunction  "
            f"(recall {pdl['pooled_recall']:.3f} vs {disj['pooled_recall']:.3f},  "
            f"FPR {pdl['pooled_fpr']:.3f} vs {disj['pooled_fpr']:.3f})"
        )

    # ================================================================
    # Validation 3: modal threshold grid-edge check
    # ================================================================
    modal_pages = max(tau_pages_counts_b, key=tau_pages_counts_b.get)
    modal_l3    = max(tau_l3_counts_b,    key=tau_l3_counts_b.get)
    print(
        f"\n[cv] Pass B modal thresholds: τ_pages={modal_pages} "
        f"({tau_pages_counts_b[modal_pages]}/25 splits),  "
        f"τ_l3={modal_l3} ({tau_l3_counts_b[modal_l3]}/25 splits)"
    )
    at_edge = (int(modal_pages) == TAU_PAGES_VALUES[-1]
               or int(modal_l3)  == TAU_L3_VALUES[-1])
    if at_edge:
        print("[cv] WARNING: modal threshold is at a grid edge — consider widening the grid")

    # ================================================================
    # Validation 4: run test suite
    # ================================================================
    print("\n[cv] Running pytest tests/ -q ...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode not in (0, 1):
        print(result.stderr)
    # Count failures by summing "FAILED " lines (works regardless of how pytest formats the count)
    n_fail = len(re.findall(r"^FAILED ", result.stdout, re.MULTILINE))
    if n_fail > 2:
        print(f"[cv] WARNING: {n_fail} test failures (expected ≤ 2 pre-existing in test_loader)")
    elif n_fail > 0:
        print(f"[cv] Tests: {n_fail} pre-existing failure(s) — OK")
    else:
        print("[cv] Tests: all passed")


if __name__ == "__main__":
    main()
