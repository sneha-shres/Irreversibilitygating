from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator

import pandas as pd


@dataclass
class Trajectory:
    benchmark: str
    task_id: str
    model: str
    goal: str
    steps: list[dict[str, Any]]
    side_effect_label: str


def load_annotations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    if "benchmark" not in df.columns and "benchmark_name" in df.columns:
        df = df.rename(columns={"benchmark_name": "benchmark"})

    if "task_id" not in df.columns and "tid" in df.columns:
        df = df.rename(columns={"tid": "task_id"})

    if "model" not in df.columns and "model_name" in df.columns:
        df = df.rename(columns={"model_name": "model"})

    if "trajectory_side_effect" not in df.columns and "side_effect_label" in df.columns:
        df = df.rename(columns={"side_effect_label": "trajectory_side_effect"})

    required_columns = {"benchmark", "task_id", "model", "trajectory_side_effect"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"annotations CSV is missing required columns: {sorted(missing)}")

    df = df.loc[:, ["benchmark", "task_id", "model", "trajectory_side_effect"]].copy()
    df["benchmark"] = df["benchmark"].astype(str).str.strip()
    df["task_id"] = df["task_id"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["trajectory_side_effect"] = df["trajectory_side_effect"].astype(str).str.strip()
    return df


def load_trajectory(path: str) -> Trajectory:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Trajectory JSON must contain an object at the top level")

    goal = data.get("goal", "")
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("Trajectory JSON 'steps' field must be a list")

    task_id = os.path.splitext(os.path.basename(path))[0]
    return Trajectory(
        benchmark="",
        task_id=task_id,
        model="",
        goal=str(goal),
        steps=[dict(step) for step in steps],
        side_effect_label=str(data.get("trajectory_side_effect", "")).strip(),
    )


def iter_trajectories(annotations_df: pd.DataFrame, trajectory_dir: str) -> Iterator[Trajectory]:
    # Build task_id -> path map once instead of walking the directory per row
    cleaned_dir = os.path.join(trajectory_dir, "cleaned")
    path_map: dict[str, str] = {}
    if os.path.exists(cleaned_dir):
        for root, _dirs, files in os.walk(cleaned_dir):
            for file in files:
                if file.endswith(".json"):
                    path_map[file[:-5]] = os.path.join(root, file)
    # Also index top-level JSON files in the trajectory directory itself
    if os.path.exists(trajectory_dir):
        for root, _dirs, files in os.walk(trajectory_dir):
            # skip the cleaned subdir which was already processed
            if os.path.abspath(root) == os.path.abspath(cleaned_dir):
                continue
            for file in files:
                if file.endswith(".json"):
                    path_map.setdefault(file[:-5], os.path.join(root, file))

    for _, row in annotations_df.iterrows():
        benchmark = str(row["benchmark"]).strip()
        if benchmark.lower() not in {"webarena", "workarena"}:
            continue

        task_id = str(row["task_id"]).strip()
        model = str(row["model"]).strip()

        candidate_path = path_map.get(task_id)
        if not candidate_path:
            raise FileNotFoundError(
                f"Trajectory file for task_id '{task_id}' not found in {trajectory_dir}"
            )

        trajectory = load_trajectory(candidate_path)
        yield Trajectory(
            benchmark=benchmark,
            task_id=task_id,
            model=model,
            goal=trajectory.goal,
            steps=trajectory.steps,
            side_effect_label=str(row["trajectory_side_effect"]).strip(),
        )
