"""Configuration and constants for the IrrGate evaluation project."""

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
RUBRIC_MODE = "stub"


@dataclass
class Config:
    tau_d: float = TAU_D
    tau_pi: float = TAU_PI
    rubric_mode: str = RUBRIC_MODE
