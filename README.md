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

In the retrospective evaluation, completed trajectories are replayed to determine at which step IrrGate would have intervened.

## Installation

```bash
pip install -e .
```

## Data Download

```bash
python scripts/download_data.py --output-dir data/raw
```

Downloads `annotations.csv` and WorkArena/WebArena trajectory JSON files from HuggingFace.

## Build Evaluation Set

```bash
python scripts/build_eval_set.py --data-dir data/raw --output data/eval_set.json
```

Creates `data/eval_set.json` with positive (side-effect) and negative trajectories.

## Run Evaluation

```bash
python scripts/run_evaluation.py \
  --eval-set data/eval_set.json \
  --trajectory-dir data/raw \
  --tau-d 0.1 \
  --tau-pi 0.3 \
  --output-dir results/my_run
```

Options:
- `--tau-d`: d_I threshold (default 0.15)
- `--tau-pi`: π threshold (default 0.30)
- `--ablation-variant`: `f_only` | `f_plus_d` | `f_plus_pi` | `disjunction` | `conjunction` | `full`
- `--output-dir`: Output directory (default `results/{date}/`)
- `--no-resume`: Ignore existing progress and restart from scratch

## Results

Results are saved to `results/{date}/`:
- `per_trajectory_results.csv`: One row per trajectory with blocking decisions
- `aggregate_results.json`: Summary metrics
- `progress.jsonl`: Per-trajectory log, used for resuming interrupted runs

The script prints a summary to stdout including two recall metrics:

| Metric | Definition |
|--------|-----------|
| `recall` | caught / all positives |
| `recall_catchable` | caught / positives where agent reached a classifiable L2/L3 step |

**Use `recall` as the primary metric** (caught / all positives). `recall_catchable` is a secondary diagnostic that excludes trajectories where the classifier never reached an L2/L3 action.

## Project Structure

```
irrgate/
├── irrgate/              # Core package
│   ├── actions.py        # Action parsing and representation
│   ├── classifier.py     # Stage-1 rules + stage-2 LLM fallback (L0–L3)
│   ├── config.py         # Hyperparameters (ALPHA, TAU_D, TAU_PI)
│   ├── gate.py           # gate_decision(profile, tau_d, tau_pi): block/approve
│   ├── profile.py        # Risk profile: f, d_I, π
│   ├── taxonomy.py       # Level definitions and severity weights
│   ├── data/             # Data loading utilities
│   └── evaluation/       # Metrics, runner, analysis
├── scripts/
│   ├── download_data.py  # Pull AgentRewardBench from HuggingFace
│   ├── build_eval_set.py # Build stratified eval set
│   └── run_evaluation.py # Main entry point (five ablation variants)
├── tests/                # Unit tests (pytest)
└── pyproject.toml
```

## Running Tests

```bash
pytest tests/
```

2 pre-existing failures in `test_loader` are unrelated to core gating logic.
