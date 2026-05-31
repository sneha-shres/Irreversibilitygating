"""Configuration and constants for the IrrGate project."""

import json
from dataclasses import dataclass
from pathlib import Path

# Load .env from repo root if present (requires python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

DATASET_REPO = "McGill-NLP/agent-reward-bench"
ANNOTATIONS_PATH = "data/annotations.csv"
TARGET_BENCHMARKS = {"webarena", "workarena"}
BETA      = 0.1   # severity weight for L1 (agent-reversible)
ALPHA     = 0.5   # severity weight for L2 (cost-reversible)
# L3 is always 1.0; L0 is always 0.0
TAU_D     = 5.0   # irreversibility density threshold (absolute cumulative severity units)
TAU_PI = 5     # irreversibility positional risk threshold (distinct pages)

_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.json"


def load_settings() -> dict:
    """Load `config/settings.json`. Return {} if missing."""
    if _SETTINGS_PATH.exists():
        return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    return {}


def save_settings(updates: dict) -> None:
    """Merge `updates` into settings file and write it."""
    settings = load_settings()
    settings.update(updates)
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass
class Config:
    tau_d: float = TAU_D
    tau_pi: int = TAU_PI
