"""Threshold analysis for IrrGate.

Reads a completed progress.jsonl from a baseline evaluation run and produces:
  1. Distribution plots of d_I and pi for positives vs negatives
  2. 2D scatter of (d_I, pi) coloured by label
  3. Spearman correlation between d_I and pi across all trajectories
  4. Recommended tau_d and tau_pi derived from distribution separation
  5. Sensitivity curve: recall and FPR across a sweep of tau values
     (post-hoc re-routing from stored profiles, no API calls)

Usage:
    PYTHONPATH=. python3 scripts/analyze_thresholds.py \
        --progress results/full_gemini_baseline/progress.jsonl \
        --output-dir results/threshold_analysis
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from irrgate.config import load_settings, save_settings


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(progress_path: str) -> list[dict]:
    records = []
    with open(progress_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# ---------------------------------------------------------------------------
# Threshold derivation
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = p / 100.0 * (len(sorted_v) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    return sorted_v[lo] + (idx - lo) * (sorted_v[hi] - sorted_v[lo])


def derive_thresholds(
    pos_values: list[float],
    neg_values: list[float],
    neg_percentile: float = 90.0,
) -> float:
    """Set threshold at the neg_percentile of the negative distribution.

    Interpretation: only the top (100 - neg_percentile)% of negatives by this
    feature will cross the threshold and enter GATED regime.
    Falls back to the midpoint between distributions if no clear gap exists.
    """
    if not neg_values:
        return 0.1
    tau_from_neg = _percentile(neg_values, neg_percentile)
    if pos_values:
        pos_low = _percentile(pos_values, 10.0)
        if pos_low > tau_from_neg:
            return (tau_from_neg + pos_low) / 2.0
    return tau_from_neg


# ---------------------------------------------------------------------------
# Sensitivity sweep (post-hoc, no API calls)
# ---------------------------------------------------------------------------

def sensitivity_sweep(
    records: list[dict],
    tau_d_values: list[float],
    tau_pi_values: list[float],
) -> list[dict]:
    """Compute recall and FPR for every (tau_d, tau_pi) combination.

    Uses the stored peak_d_I and peak_pi from the progress records to simulate
    routing without re-running the evaluation.  A trajectory is treated as
    'would have been routed to GATED' if peak_d_I >= tau_d OR
    peak_pi >= tau_pi (i.e. at least one threshold exceeded), subject to f=1
    (trajectory has at least one risky action, inferred from peak_d_I > 0).

    Note: this approximates blocking — it checks routing only, not rubric.
    The real blocking also requires at least one rubric check to fail.
    Use this curve for operating-point selection; the exact numbers come from
    the full evaluation run at the selected thresholds.
    """
    pos = [r for r in records if r.get("is_positive")]
    neg = [r for r in records if not r.get("is_positive")]

    results = []
    for tau_d in tau_d_values:
        for tau_pi in tau_pi_values:
            tp = sum(
                1 for r in pos
                if (r.get("peak_d_I") or 0) > 0
                and (
                    (r.get("peak_d_I") or 0) >= tau_d
                    or (r.get("peak_pi") or 0) >= tau_pi
                )
            )
            fp = sum(
                1 for r in neg
                if (r.get("peak_d_I") or 0) > 0
                and (
                    (r.get("peak_d_I") or 0) >= tau_d
                    or (r.get("peak_pi") or 0) >= tau_pi
                )
            )
            recall = tp / len(pos) if pos else 0.0
            fpr    = fp / len(neg) if neg else 0.0
            results.append({"tau_d": tau_d, "tau_pi": tau_pi, "recall": recall, "fpr": fpr})
    return results


# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------

def spearman_rho(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")

    def rank(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and vals[order[j + 1]] == vals[order[j]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = rank(x), rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den = math.sqrt(
        sum((rx[i] - mean_rx) ** 2 for i in range(n))
        * sum((ry[i] - mean_ry) ** 2 for i in range(n))
    )
    return num / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Plotting (matplotlib optional)
# ---------------------------------------------------------------------------

def _try_plot(records, pos, neg, tau_d, tau_pi, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # d_I distribution
    pos_d = [r["peak_d_I"] for r in pos if r.get("peak_d_I") is not None]
    neg_d = [r["peak_d_I"] for r in neg if r.get("peak_d_I") is not None]
    axes[0].hist(neg_d, bins=30, alpha=0.6, label="negatives", color="steelblue")
    axes[0].hist(pos_d, bins=30, alpha=0.7, label="positives", color="salmon")
    axes[0].axvline(tau_d, color="red", linestyle="--", label=f"tau_d={tau_d:.3f}")
    axes[0].set_xlabel("peak d_I")
    axes[0].set_title("d_I distribution")
    axes[0].legend()

    # pi distribution
    pos_pi = [r["peak_pi"] for r in pos if r.get("peak_pi") is not None]
    neg_pi = [r["peak_pi"] for r in neg if r.get("peak_pi") is not None]
    axes[1].hist(neg_pi, bins=30, alpha=0.6, label="negatives", color="steelblue")
    axes[1].hist(pos_pi, bins=30, alpha=0.7, label="positives", color="salmon")
    axes[1].axvline(tau_pi, color="red", linestyle="--", label=f"tau_pi={tau_pi:.3f}")
    axes[1].set_xlabel("peak pi")
    axes[1].set_title("pi distribution")
    axes[1].legend()

    # 2D scatter
    axes[2].scatter(neg_d, neg_pi, alpha=0.3, s=10, label="negatives", color="steelblue")
    axes[2].scatter(pos_d, pos_pi, alpha=0.7, s=20, label="positives", color="salmon", marker="*")
    axes[2].axvline(tau_d,  color="red",    linestyle="--", linewidth=0.8)
    axes[2].axhline(tau_pi, color="orange", linestyle="--", linewidth=0.8)
    axes[2].set_xlabel("peak d_I")
    axes[2].set_ylabel("peak pi")
    axes[2].set_title("(d_I, pi) scatter")
    axes[2].legend()

    plt.tight_layout()
    out = os.path.join(output_dir, "distributions.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[analyze] distributions plot saved → {out}")

    # Sensitivity curve
    tau_d_sweep  = [round(v, 4) for v in list(np.linspace(0.01, 0.4, 40))]
    tau_pi_sweep = [round(v, 4) for v in list(np.linspace(0.05, 0.8, 40))]
    sweep = sensitivity_sweep(records, tau_d_sweep, tau_pi_sweep)

    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([s["fpr"] for s in sweep], [s["recall"] for s in sweep],
               alpha=0.4, s=12, color="gray", label="sweep configs")
    ax.set_xlabel("FPR")
    ax.set_ylabel("Recall")
    ax.set_title("Recall–FPR sensitivity (post-hoc routing)")
    ax.legend()
    out2 = os.path.join(output_dir, "sensitivity_curve.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"[analyze] sensitivity curve saved → {out2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    s = load_settings()
    baseline_dir = s.get("baseline_output_dir", "results/baseline_gemini")

    parser = argparse.ArgumentParser(
        description="IrrGate threshold analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--progress",
        default=os.path.join(baseline_dir, "progress.jsonl"),
        help="Path to progress.jsonl from a completed evaluation run",
    )
    parser.add_argument(
        "--output-dir",
        default=s.get("analysis_output_dir", "results/threshold_analysis"),
        help="Directory to write outputs",
    )
    parser.add_argument(
        "--neg-percentile", type=float, default=90.0,
        help="Percentile of negative distribution to set as tau",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = load_records(args.progress)
    print(f"[analyze] loaded {len(records)} records")

    pos = [r for r in records if r.get("is_positive")]
    neg = [r for r in records if not r.get("is_positive")]
    print(f"[analyze] positives={len(pos)}  negatives={len(neg)}")

    # -----------------------------------------------------------------------
    # 1. Derive thresholds from distributions
    # -----------------------------------------------------------------------
    pos_d  = [r["peak_d_I"] for r in pos if r.get("peak_d_I") is not None]
    neg_d  = [r["peak_d_I"] for r in neg if r.get("peak_d_I") is not None]
    pos_pi = [r["peak_pi"]  for r in pos if r.get("peak_pi")  is not None]
    neg_pi = [r["peak_pi"]  for r in neg if r.get("peak_pi")  is not None]

    tau_d  = derive_thresholds(pos_d,  neg_d,  args.neg_percentile)
    tau_pi = derive_thresholds(pos_pi, neg_pi, args.neg_percentile)

    print(f"\n[analyze] --- Distribution summary ---")
    print(f"  d_I  | neg: median={_percentile(neg_d,50):.4f}  p90={_percentile(neg_d,90):.4f}  "
          f"pos: median={_percentile(pos_d,50):.4f}  p10={_percentile(pos_d,10):.4f}")
    print(f"  pi   | neg: median={_percentile(neg_pi,50):.4f}  p90={_percentile(neg_pi,90):.4f}  "
          f"pos: median={_percentile(pos_pi,50):.4f}  p10={_percentile(pos_pi,10):.4f}")
    print(f"\n[analyze] Derived thresholds: tau_d={tau_d:.4f}  tau_pi={tau_pi:.4f}")

    # -----------------------------------------------------------------------
    # 2. Spearman rho(d_I, pi)
    # -----------------------------------------------------------------------
    shared = [(r["peak_d_I"], r["peak_pi"]) for r in records
              if r.get("peak_d_I") is not None and r.get("peak_pi") is not None]
    if shared:
        rho = spearman_rho([x for x, _ in shared], [y for _, y in shared])
        print(f"\n[analyze] Spearman rho(d_I, pi) across all {len(shared)} trajectories: {rho:.3f}")
        interp = "highly correlated (redundant)" if abs(rho) > 0.8 else \
                 "moderately correlated" if abs(rho) > 0.5 else "weakly correlated (independent signals)"
        print(f"  Interpretation: {interp}")

    # -----------------------------------------------------------------------
    # 3. Sensitivity sweep at derived thresholds
    # -----------------------------------------------------------------------
    tau_d_sweep  = [round(tau_d * f, 4) for f in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]]
    tau_pi_sweep = [round(tau_pi * f, 4) for f in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]]
    sweep = sensitivity_sweep(records, tau_d_sweep, tau_pi_sweep)

    print(f"\n[analyze] --- Sensitivity sweep (post-hoc routing only) ---")
    print(f"{'tau_d':>8} {'tau_pi':>8} {'recall':>8} {'fpr':>8}")
    for s in sorted(sweep, key=lambda x: (-x["recall"], x["fpr"])):
        marker = " <-- derived" if (
            abs(s["tau_d"] - tau_d) < 1e-6 and abs(s["tau_pi"] - tau_pi) < 1e-6
        ) else ""
        print(f"{s['tau_d']:8.4f} {s['tau_pi']:8.4f} {s['recall']:8.3f} {s['fpr']:8.3f}{marker}")

    # -----------------------------------------------------------------------
    # 4. Recall/FPR at derived thresholds with Wilson CIs
    # -----------------------------------------------------------------------
    at_derived = [s for s in sweep
                  if abs(s["tau_d"] - tau_d) < 1e-6 and abs(s["tau_pi"] - tau_pi) < 1e-6]
    if at_derived:
        s = at_derived[0]
        tp = round(s["recall"] * len(pos))
        fp = round(s["fpr"]    * len(neg))
        recall_lo, recall_hi = wilson_ci(tp, len(pos))
        fpr_lo,    fpr_hi    = wilson_ci(fp, len(neg))
        print(f"\n[analyze] At tau_d={tau_d:.4f}, tau_pi={tau_pi:.4f} (routing approximation):")
        print(f"  recall = {s['recall']:.3f}  95% CI [{recall_lo:.3f}, {recall_hi:.3f}]  ({tp}/{len(pos)})")
        print(f"  fpr    = {s['fpr']:.3f}  95% CI [{fpr_lo:.3f}, {fpr_hi:.3f}]  ({fp}/{len(neg)})")

    # -----------------------------------------------------------------------
    # 5. False negative breakdown for positives
    # -----------------------------------------------------------------------
    print(f"\n[analyze] --- False negative breakdown (positives not blocked) ---")
    not_blocked = [r for r in pos if r.get("first_blocking_step") is None]
    bypass_scope    = [r for r in not_blocked if r.get("side_effect_step") is None
                       and r.get("n_steps", 0) <= 5]
    bypass_loop     = [r for r in not_blocked if r.get("side_effect_step") is None
                       and r.get("n_steps", 0) > 5]
    low_regime      = [r for r in not_blocked if r.get("side_effect_step") is not None
                       and r.get("peak_regime") == "low"]
    bypass_classifier = [r for r in not_blocked if r.get("side_effect_step") is None
                         and r.get("peak_regime") == "bypass"]

    print(f"  BYPASS (no L2/L3 classified at all): {len([r for r in not_blocked if r.get('peak_regime') == 'bypass'])}")
    print(f"    of which short traj (<=5 steps, likely crash): {len(bypass_scope)}")
    print(f"    of which longer traj (classifier/taxonomy gap): {len(bypass_loop)}")
    print(f"  LOW (risky step existed but regime stayed LOW):  {len(low_regime)}")
    other_fn = [r for r in not_blocked
                if r.get("peak_regime") not in ("bypass", "low", "gated")]
    print(f"  GATED (reached full rubric but not blocked):  {len([r for r in not_blocked if r.get('peak_regime') == 'gated'])}")
    print(f"  Other:  {len(other_fn)}")

    # -----------------------------------------------------------------------
    # 6. Save outputs
    # -----------------------------------------------------------------------
    output = {
        "derived_tau_d": tau_d,
        "derived_tau_pi": tau_pi,
        "neg_percentile_used": args.neg_percentile,
        "spearman_rho_d_I_pi": rho if shared else None,
        "n_positives": len(pos),
        "n_negatives": len(neg),
        "sensitivity_sweep": sweep,
    }
    out_path = os.path.join(args.output_dir, "threshold_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[analyze] results saved → {out_path}")

    # Write derived thresholds back to config/settings.json so subsequent
    # run_evaluation.py and ablation commands pick them up automatically.
    save_settings({
        "tau_d": round(tau_d, 6),
        "tau_pi": round(tau_pi, 6),
    })
    print(f"[analyze] tau_d={tau_d:.4f} and tau_pi={tau_pi:.4f} written to config/settings.json")

    _try_plot(records, pos, neg, tau_d, tau_pi, args.output_dir)


if __name__ == "__main__":
    main()
