"""Sampler utilities for building the IrrGate evaluation dataset."""

import json
import os

import pandas as pd

from irrgate.data.loader import Trajectory


def _build_run_map(trajectory_dir: str) -> dict[tuple[str, str], str]:
    """Walk cleaned dir and return {(task_id, model): file_path} for all available runs.

    Directory layout: cleaned/{benchmark}/{model}/{exp_name}/{task_id}.json
    The model directory name matches the model field in annotations.
    """
    available: dict[tuple[str, str], str] = {}
    cleaned_dir = os.path.join(trajectory_dir, "cleaned")
    if not os.path.exists(cleaned_dir):
        return available
    for root, _dirs, files in os.walk(cleaned_dir):
        rel = os.path.relpath(root, cleaned_dir)
        parts = rel.split(os.sep)
        if len(parts) < 2:
            continue
        model = parts[1]  # cleaned/{benchmark}/{model}/...
        for file in files:
            if file.endswith(".json"):
                task_id = file[:-5]
                available[(task_id, model)] = os.path.join(root, file)
    return available


def _df_to_trajectories(
    df: pd.DataFrame,
    run_map: dict[tuple[str, str], str],
) -> list[Trajectory]:
    """Convert deduplicated annotation rows to Trajectory metadata objects.

    Only includes runs that have a matching trajectory file on disk.
    """
    result = []
    for _, row in df.iterrows():
        task_id = str(row["task_id"]).strip()
        model = str(row["model"]).strip()
        if (task_id, model) not in run_map:
            continue
        result.append(Trajectory(
            benchmark=str(row["benchmark"]).strip(),
            task_id=task_id,
            model=model,
            goal="",
            steps=[],
            side_effect_label=str(row["trajectory_side_effect"]).strip(),
        ))
    return result


def _aggregate_labels(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to one row per (task_id, model) run.

    Labeling rule: a run is positive if any annotator marked it as a side-effect
    trajectory (safety-conservative: a single expert flag is sufficient).
    """
    key_cols = ["task_id", "model"]
    first_cols = annotations_df.groupby(key_cols, sort=False)[["benchmark"]].first().reset_index()
    any_yes = (
        annotations_df
        .assign(_yes=annotations_df["trajectory_side_effect"].str.lower() == "yes")
        .groupby(key_cols, sort=False)["_yes"]
        .any()
        .reset_index()
        .rename(columns={"_yes": "trajectory_side_effect"})
    )
    any_yes["trajectory_side_effect"] = any_yes["trajectory_side_effect"].map({True: "Yes", False: "No"})
    return first_cols.merge(any_yes, on=key_cols)


def build_eval_set(
    annotations_df: pd.DataFrame,
    trajectory_dir: str,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Build evaluation dataset.

    Each unique (task_id, model) agent run is counted exactly once. A run is
    labelled positive if any annotator marked it as a side-effect trajectory
    (safety-conservative). Runs where all annotators said No are labelled negative.
    Only runs with a matching trajectory file on disk are included.

    Returns:
      positives: runs annotated as side-effect=Yes (any annotator)
      negatives: runs annotated as side-effect=No (all annotators)
    """
    run_map = _build_run_map(trajectory_dir)

    filtered = annotations_df[
        annotations_df["benchmark"].str.lower().isin({"webarena", "workarena"})
    ].copy()

    aggregated = _aggregate_labels(filtered)

    positives_df = aggregated[aggregated["trajectory_side_effect"] == "Yes"]
    negatives_df = aggregated[aggregated["trajectory_side_effect"] == "No"]

    return _df_to_trajectories(positives_df, run_map), _df_to_trajectories(negatives_df, run_map)


def save_eval_set(
    positives: list[Trajectory],
    negatives: list[Trajectory],
    output_path: str,
) -> None:
    """Save evaluation dataset to JSON for reproducibility."""
    data = {
        "positives": [
            {
                "benchmark": t.benchmark,
                "task_id": t.task_id,
                "model": t.model,
                "side_effect_label": t.side_effect_label,
            }
            for t in positives
        ],
        "negatives": [
            {
                "benchmark": t.benchmark,
                "task_id": t.task_id,
                "model": t.model,
                "side_effect_label": t.side_effect_label,
            }
            for t in negatives
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
