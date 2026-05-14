"""Configuration and constants for the IrrGate evaluation project."""

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
ALPHA = 0.5
TAU_D = 0.15
TAU_PI = 0.30

_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.json"


def load_settings() -> dict:
    """Load config/settings.json. Returns empty dict if file is missing."""
    if _SETTINGS_PATH.exists():
        return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    return {}


def save_settings(updates: dict) -> None:
    """Merge updates into config/settings.json and write it back."""
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
    tau_pi: float = TAU_PI
    use_conjunction: bool = False  # if True, block iff f=1 AND d_I>=tau_d AND pi>=tau_pi
