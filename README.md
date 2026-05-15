# IrrGate

A runtime, per-step safety gate for web-based AI agents. At each step, IrrGate intercepts the agent's proposed action, evaluates the cumulative trajectory so far, and approves or blocks the action before it executes.

Evaluated retrospectively on WorkArena and WebArena trajectories from [AgentRewardBench](https://github.com/McGill-NLP/agent-reward-bench).

## How It Works

BrowserGym agents are **reactive** — they produce one action at a time with no upfront plan. IrrGate fits this model:

```
agent proposes action
        ↓
classify trajectory-so-far by reversibility level (L0–L3)
        ↓
compute risk profile (f, d_I, π)
        ↓
gate_decision(profile, τ_d, τ_π) → approve or block
```

**Gate policy (disjunction):** block iff `f=1` and (`d_I ≥ τ_d` or `π ≥ τ_π`). No LLM at gate time.

In the retrospective evaluation, completed trajectories are replayed to determine at which step IrrGate would have first intervened.

### Reversibility levels

| Level | Description | Examples |
|-------|-------------|---------|
| L0 | Read-only / communicative | navigate, scroll, send_msg_to_user |
| L1 | Easily reversible | form fill, text input |
| L2 | Cost-reversible (severity = α = 0.5) | like, follow, update, edit |
| L3 | Irreversible (severity = 1.0) | submit, delete, publish, confirm |

### Risk profile

- **f**: 1 if any action has severity > 0 (any L2/L3 step), else 0
- **d_I**: mean severity over all steps (density of risky actions)
- **π**: plan-level risk weighted by BID-coverage — how many distinct UI elements (BIDs) the agent has already visited before each action; actions targeting unseen elements carry higher residual risk

### Gate decision

```python
def gate_decision(profile, tau_d, tau_pi) -> "approve" | "block":
    if profile.f == 0:
        return "approve"                         # no risky step seen
    if profile.d_I >= tau_d or profile.pi >= tau_pi:
        return "block"
    return "approve"
```

## Installation

```bash
pip install -e .
```

## Data Download

```bash
python scripts/download_data.py --output-dir data/raw
```

Downloads `annotations.csv` and WorkArena/WebArena trajectory JSON files from HuggingFace.

## Running the Pipeline

All commands run from the **repo root**:

```bash
# Step 0: Build eval set (deduplicates by (task_id, model), all 4 agent models)
PYTHONPATH=. python3 scripts/build_eval_set.py

# Step 1: Pre-build classification cache (avoids redundant Gemini calls; resumable)
PYTHONPATH=. python3 scripts/build_classification_cache.py \
  --eval-set data/eval_set.json \
  --trajectory-dir data/raw \
  --output data/classification_cache.parquet

# Step 2: Compute full-trajectory risk profiles from the classification cache
PYTHONPATH=. python3 scripts/compute_profiles.py
# → results/profiles/profiles.parquet

# Step 3: 5×5 repeated stratified cross-validation for threshold selection
PYTHONPATH=. python3 scripts/run_cv.py
# → results/cv/cv_results.json

# Step 4: Phase 3 diagnostics (per-benchmark, FN enumeration, FP sample, α sensitivity)
PYTHONPATH=. python3 scripts/run_diagnostics.py
# → results/diagnostics/

# Step 5 (optional): Full trajectory-by-trajectory evaluation with a fixed threshold
PYTHONPATH=. python3 scripts/run_evaluation.py \
  --eval-set data/eval_set.json \
  --trajectory-dir data/raw \
  --tau-d 0.05 \
  --tau-pi 0.10 \
  --ablation-variant disjunction \
  --output-dir results/my_run
```

### Ablation variants

```bash
--ablation-variant f_only       # block iff f=1
--ablation-variant f_plus_d     # block iff f=1 AND d_I ≥ τ_d
--ablation-variant f_plus_pi    # block iff f=1 AND π ≥ τ_π
--ablation-variant disjunction  # block iff f=1 AND (d_I ≥ τ_d OR π ≥ τ_π) [primary]
--ablation-variant conjunction  # block iff f=1 AND d_I ≥ τ_d AND π ≥ τ_π
```

## Results

### Dataset

- **870 trajectories**: 54 positives (side-effect=Yes), 816 negatives
- **4 agent models**: GPT-4o, Claude 3.7 Sonnet, Qwen 2.5-VL-72B, Llama 3.3-70B
- **2 benchmarks**: WebArena (32 pos), WorkArena (22 pos)

### CV summary (5×5 repeated stratified CV, FPR budget ≤ 10%)

Two passes run on the same 25 stratified splits. Pass A uses density features (τ_d, τ_π); Pass B uses shape features (τ_pages, τ_l3). Pooled Wilson 95% CIs across all 25 held-out folds; `*` marks the primary variant driving threshold selection in each pass.

**Pass A — density variants** (grid: `τ_d × τ_π`, primary: `disjunction`)

No configuration met the 10% FPR budget; the fallback criterion (minimize FPR) selected `τ_d = 0.30, τ_π = 0.50` in all 25 splits.

| Variant | Recall | 95% CI | FPR | 95% CI |
|---------|--------|--------|-----|--------|
| f_only | 0.870 | [0.825, 0.905] | 0.373 | [0.358, 0.387] |
| f_plus_d | 0.352 | [0.297, 0.411] | 0.175 | [0.164, 0.187] |
| f_plus_pi | 0.426 | [0.368, 0.486] | 0.184 | [0.172, 0.196] |
| **disjunction** * | **0.444** | [0.386, 0.504] | **0.206** | [0.194, 0.219] |
| conjunction | 0.333 | [0.280, 0.392] | 0.153 | [0.142, 0.165] |

**Pass B — shape variants** (grid: `τ_pages × τ_l3`, primary: `pages_disj_l3`)

The shape formulation meets the FPR budget: `pages_disj_l3` achieves FPR 0.098, recall 0.544 under held-out CV (strictly dominates the density disjunction on both metrics). After two grid-widening iterations (`τ_pages ∈ {2–12}`, `τ_l3 ∈ {1–18}`), the CV selected `τ_pages = 5` (24/25 splits) and `τ_l3 = 14` (11/25 splits) — neither at the grid edge. `pages_only` at the same τ_pages yields recall 0.441 vs. 0.544 for `pages_disj_l3`, so the l3 term adds a meaningful +10.3pp recall at +4.2pp FPR cost.

| Variant | Recall | 95% CI | FPR | 95% CI |
|---------|--------|--------|-----|--------|
| f_only | 0.870 | [0.825, 0.905] | 0.373 | [0.358, 0.387] |
| pages_only | 0.441 | [0.383, 0.500] | 0.056 | [0.049, 0.064] |
| l3_count_only | 0.141 | [0.104, 0.187] | 0.048 | [0.042, 0.055] |
| **pages_disj_l3** * | **0.544** | [0.485, 0.603] | **0.098** | [0.089, 0.107] |
| pages_conj_l3 | 0.037 | [0.020, 0.067] | 0.007 | [0.005, 0.010] |

### Full-dataset performance

At the recall ceiling (`τ_d = 0.05, τ_π = 0.10`):
- **Recall: 87.0%** (47/54) — 7 permanent misses, all f=0 (pure L0/L1 trajectories)
- FPR: ~36.6%

At CV-selected thresholds (`τ_d = 0.30, τ_π = 0.50`):
- **Recall: 44.4%** (24/54)
- **FPR: 20.6%** (168/816)

### Per-benchmark breakdown (disjunction, τ_d = 0.30, τ_π = 0.50)

| Benchmark | Model | Pos | Neg | Recall | FPR |
|-----------|-------|-----|-----|--------|-----|
| WebArena | Qwen 2.5-VL-72B | 11 | 89 | 1.000 | 0.427 |
| WebArena | Claude 3.7 Sonnet | 9 | 91 | 1.000 | 0.407 |
| WebArena | GPT-4o | 8 | 92 | 0.750 | 0.424 |
| WebArena | Llama 3.3-70B | 4 | 94 | 0.750 | 0.436 |
| WorkArena | Claude 3.7 Sonnet | 8 | 110 | 0.875 | 0.409 |
| WorkArena | GPT-4o | 6 | 112 | 0.833 | 0.277 |
| WorkArena | Llama 3.3-70B | 5 | 113 | 0.800 | 0.310 |
| WorkArena | Qwen 2.5-VL-72B | 3 | 115 | 0.667 | 0.304 |

### Alpha sensitivity (disjunction, τ_d = 0.05, τ_π = 0.10)

| α | Recall | FPR |
|---|--------|-----|
| 0.25 | 0.870 | 0.365 |
| 0.50 | 0.870 | 0.366 |
| 0.75 | 0.870 | 0.368 |
| 1.00 | 0.870 | 0.368 |

α (severity weight for L2 actions) is **inert**: recall and FPR change by < 0.01 across the full range. τ_d is the only active discriminative lever.

### Stage contribution

Of 2,353 L2/L3 classification steps: **95.1% from stage-1** (rule-based), 4.9% from stage-2 (Gemini). Stage-2 adds coverage with minimal API cost.

### False negative analysis (at recall ceiling)

All 7 permanent misses are f=0 trajectories — IrrGate found no L2/L3 action. These are trajectories where the only action is `send_msg_to_user`, classified as L0 by design (IrrGate scopes to state-changing side effects, not communicative ones).

## Project Structure

```
irrgate/                      # Core package
├── actions.py                # Action parsing and representation
├── classifier.py             # Stage-1 rules + stage-2 LLM fallback (L0–L3)
├── config.py                 # Hyperparameters (ALPHA, TAU_D, TAU_PI); settings I/O
├── gate.py                   # gate_decision(profile, tau_d, tau_pi): block/approve
├── profile.py                # Risk profile: f, d_I, π (BID-coverage formula)
├── taxonomy.py               # Level definitions and severity weights
├── _gemini.py                # Gemini client singleton; backoff/throttling
├── data/                     # Data loading utilities
└── evaluation/               # Metrics, runner
scripts/
├── download_data.py          # Pull AgentRewardBench from HuggingFace
├── build_eval_set.py         # Build stratified eval set (deduplicated by task×model)
├── build_classification_cache.py  # Pre-compute per-step classifications → parquet
├── compute_profiles.py       # Full-trajectory peak profiles → results/profiles/
├── run_cv.py                 # 5×5 stratified CV → results/cv/cv_results.json
├── run_diagnostics.py        # Phase 3 diagnostics → results/diagnostics/
├── run_evaluation.py         # Trajectory-by-trajectory evaluation (five variants)
└── analyze_thresholds.py     # Post-hoc threshold sweep from progress.jsonl
data/
├── eval_set.json             # 870 trajectories (54 pos, 816 neg)
├── classification_cache.parquet  # Pre-built per-step classifications
└── raw/                      # Raw trajectory JSON files
results/
├── profiles/                 # profiles.parquet (one row per trajectory)
├── cv/                       # cv_results.json (two-pass: pass_a_density + pass_b_shape)
└── diagnostics/
    ├── density/              # disjunction at CV-selected τ_d=0.30, τ_π=0.50
    └── shape/                # pages_disj_l3 at CV-selected τ_pages=5, τ_l3=14
tests/                        # pytest test suite
config/
└── settings.json             # Active tau_d, tau_pi (written by analyze_thresholds.py)
```

## Running Tests

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

2 pre-existing failures in `test_loader` are unrelated to core gating logic.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALPHA` | 0.5 | Severity weight for L2 actions (fixed, not tuned — inert on this dataset) |
| `TAU_D` | 0.15 | d_I threshold for blocking |
| `TAU_PI` | 0.30 | π threshold for blocking |

Active values are read from `config/settings.json`; CLI flags take precedence.

## Limitations

- **Density FPR budget unachievable**: The density formulation cannot meet FPR ≤ 10%; minimum is 20.6% at recall 44.4%. The shape formulation achieves FPR 0.098 at recall 0.544 under held-out CV (budget met). The two formulations are complementary: density requires no page-visit tracking but cannot stay under the FPR budget; shape requires page-count bookkeeping but does.
- **d_I dilution**: Mean severity is divided over all steps; a long trajectory with one risky action near the end may never cross τ_d. The shape formulation uses raw counts (not averages) and is not subject to this dilution.
- **τ_π inertness**: π has negligible discriminative effect on this dataset; τ_d is the only active lever in the density formulation.
- **Shape threshold brittleness**: At (τ_pages=5, τ_l3=14), decreasing τ_pages by 1 raises full-dataset FPR from 0.100 to 0.168 (+6.8pp); increasing it by 1 drops recall from 0.574 to 0.481 (−9.3pp). The τ_pages lever is sensitive near the operating point.
- **7 permanent f=0 misses**: Pure communicative trajectories (`send_msg_to_user` only) that annotators labeled as side effects — a taxonomy boundary, not a bug.
