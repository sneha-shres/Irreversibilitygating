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
compute risk profile (f, d_I, p_I)
        ↓
gate_decision(profile, τ_d, τ_π) → approve or block
```

**Gate policy (irr_gate):** block iff `f=1` and (`d_I ≥ τ_d` or `p_I ≥ τ_π`). No LLM at gate time.

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
- **d_I**: absolute cumulative severity sum across all steps — NOT a mean; monotone increasing, avoids the length-dilution problem of density
- **p_I**: count of distinct page URLs from step 0 up to and including the last L2/L3 step; captures how broadly the agent explored before its last risky action

### Gate decision

```python
def gate_decision(profile, tau_d, tau_pi) -> "approve" | "block":
    if profile.f == 0:
        return "approve"                              # no risky step seen
    if profile.d_I >= tau_d or profile.irr_pos >= tau_pi:
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

# Step 4: Diagnostics (reads CV-selected thresholds automatically)
PYTHONPATH=. python3 scripts/run_diagnostics.py
# → results/diagnostics/

# Step 5 (optional): Full trajectory-by-trajectory evaluation with a fixed threshold
PYTHONPATH=. python3 scripts/run_evaluation.py \
  --eval-set data/eval_set.json \
  --trajectory-dir data/raw \
  --tau-d 10.0 \
  --tau-pi 6 \
  --output-dir results/my_run
```

### Ablation variants

```bash
--ablation-variant f_only              # block iff f=1
--ablation-variant irr_density_only    # block iff f=1 AND d_I ≥ τ_d
--ablation-variant irr_positional_only # block iff f=1 AND irr_pos ≥ τ_π
--ablation-variant irr_gate            # block iff f=1 AND (d_I ≥ τ_d OR irr_pos ≥ τ_π) [primary]
```

## Results

### Dataset

- **870 trajectories**: 54 positives (side-effect=Yes), 816 negatives
- **4 agent models**: GPT-4o, Claude 3.7 Sonnet, Qwen 2.5-VL-72B, Llama 3.3-70B
- **2 benchmarks**: WebArena (32 pos), WorkArena (22 pos)

### CV summary (5×5 repeated stratified CV, FPR budget ≤ 10%)

Grid: τ_d ∈ {0.5–20.0}, τ_π ∈ {1–10}. Modal CV-selected: **τ_d = 10.0** (17/25 splits), **τ_π = 6** (19/25 splits). Pooled Wilson 95% CIs across all 25 held-out folds; `*` marks the primary variant.

| Variant | Recall | 95% CI | FPR | 95% CI |
|---------|--------|--------|-----|--------|
| f_only | 0.889 | [0.845, 0.922] | 0.387 | [0.372, 0.402] |
| irr_density_only | 0.170 | [0.130, 0.220] | 0.060 | [0.053, 0.068] |
| irr_positional_only | 0.341 | [0.287, 0.399] | 0.050 | [0.043, 0.057] |
| **irr_gate** * | **0.470** | [0.412, 0.530] | **0.097** | [0.088, 0.106] |

- **Recall ceiling: 88.9% (48/54)** — 6 permanent f=0 misses (pure L0/L1 trajectories; taxonomy boundary)
- **FPR budget met.** irr_gate strictly dominates both single-feature variants on both metrics
- irr_positional_only contributes the larger share of recall; the d_I term adds +12.9pp recall at +4.7pp FPR cost


## Project Structure

```
irrgate/                      # Core package
├── actions.py                # Action parsing and representation
├── classifier.py             # Stage-1 rules + stage-2 LLM fallback (L0–L3)
├── config.py                 # Hyperparameters (ALPHA, BETA, TAU_D, TAU_PI); settings I/O
├── gate.py                   # gate_decision(profile, tau_d, tau_pi): block/approve
├── profile.py                # Risk profile: f, d_I (cumulative), irr_pos (distinct pages)
├── taxonomy.py               # Level definitions and severity weights
├── _gemini.py                # Gemini client singleton; backoff/throttling
├── data/                     # Data loading utilities
└── evaluation/               # Metrics, runner
scripts/
├── download_data.py          # Pull AgentRewardBench from HuggingFace
├── build_eval_set.py         # Build stratified eval set (deduplicated by task×model)
├── build_classification_cache.py  # Pre-compute per-step classifications → parquet
├── compute_profiles.py       # Full-trajectory profiles → results/profiles/
├── run_cv.py                 # 5×5 stratified CV → results/cv/cv_results.json
├── run_diagnostics.py        # Diagnostics → results/diagnostics/
├── run_evaluation.py         # Trajectory-by-trajectory evaluation (four variants)
└── analyze_thresholds.py     # Post-hoc threshold sweep from progress.jsonl
data/
├── eval_set.json             # 870 trajectories (54 pos, 816 neg)
├── classification_cache.parquet  # Pre-built per-step classifications
└── raw/                      # Raw trajectory JSON files
results/
├── profiles/                 # profiles.parquet (one row per trajectory)
├── cv/                       # cv_results.json (irr_gate CV; Wilson CIs; tau selection)
└── diagnostics/              # FN list, FP sample, threshold sensitivity, stage contribution
config/
└── settings.json             # Active tau_d, tau_pi (written by analyze_thresholds.py)
```

