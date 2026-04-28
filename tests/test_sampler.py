from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from irrgate.data.loader import Trajectory
from irrgate.data.sampler import build_eval_set, save_eval_set


def make_trajectory(task_id: str, side_effect: str, steps: list[dict] | None = None) -> Trajectory:
    if steps is None:
        steps = [{"action": "noop()", "axtree": "", "url": "", "reasoning": "", "bounding_boxes": []}]
    return Trajectory(
        benchmark="webarena",
        task_id=task_id,
        model="gpt-4",
        goal="Test",
        steps=steps,
        side_effect_label=side_effect,
    )


def test_build_eval_set_returns_correct_positives_and_negatives():
    annotations = pd.DataFrame([
        {"benchmark": "WebArena", "task_id": "pos1", "model": "gpt-4", "trajectory_side_effect": "Yes"},
        {"benchmark": "WebArena", "task_id": "pos2", "model": "gpt-4", "trajectory_side_effect": "Yes"},
        {"benchmark": "WebArena", "task_id": "neg1", "model": "gpt-4", "trajectory_side_effect": "No"},
        {"benchmark": "WebArena", "task_id": "neg2", "model": "gpt-4", "trajectory_side_effect": "No"},
        {"benchmark": "Other", "task_id": "other", "model": "gpt-4", "trajectory_side_effect": "Yes"},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trajectory files
        for row in annotations.itertuples():
            if row.benchmark.lower() in {"webarena", "workarena"}:
                path = Path(tmpdir) / f"{row.task_id}.json"
                trajectory = make_trajectory(row.task_id, row.trajectory_side_effect)
                with open(path, "w") as f:
                    json.dump({
                        "goal": trajectory.goal,
                        "steps": trajectory.steps,
                        "trajectory_side_effect": trajectory.side_effect_label,
                    }, f)

        positives, negatives = build_eval_set(annotations, tmpdir, n_negatives_per_pos=1, seed=42)

        assert len(positives) == 2
        assert all(p.side_effect_label == "Yes" for p in positives)
        assert len(negatives) == 2  # 2 positives * 1 negative per positive
        assert all(n.side_effect_label == "No" for n in negatives)


def test_build_eval_set_stratifies_negatives():
    # Create trajectories: some with irreversible actions, some without
    irreversible_steps = [{"action": "click('100')", "axtree": "[100] role='button' name='Submit'", "url": "", "reasoning": "", "bounding_boxes": []}]
    reversible_steps = [{"action": "noop()", "axtree": "", "url": "", "reasoning": "", "bounding_boxes": []}]

    annotations = pd.DataFrame([
        {"benchmark": "WebArena", "task_id": "pos1", "model": "gpt-4", "trajectory_side_effect": "Yes"},
        {"benchmark": "WebArena", "task_id": "neg_irrev", "model": "gpt-4", "trajectory_side_effect": "No"},
        {"benchmark": "WebArena", "task_id": "neg_rev", "model": "gpt-4", "trajectory_side_effect": "No"},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Positives
        path = Path(tmpdir) / "pos1.json"
        trajectory = make_trajectory("pos1", "Yes", reversible_steps)
        with open(path, "w") as f:
            json.dump({
                "goal": trajectory.goal,
                "steps": trajectory.steps,
                "trajectory_side_effect": trajectory.side_effect_label,
            }, f)

        # Irreversible negative
        path = Path(tmpdir) / "neg_irrev.json"
        trajectory = make_trajectory("neg_irrev", "No", irreversible_steps)
        with open(path, "w") as f:
            json.dump({
                "goal": trajectory.goal,
                "steps": trajectory.steps,
                "trajectory_side_effect": trajectory.side_effect_label,
            }, f)

        # Reversible negative
        path = Path(tmpdir) / "neg_rev.json"
        trajectory = make_trajectory("neg_rev", "No", reversible_steps)
        with open(path, "w") as f:
            json.dump({
                "goal": trajectory.goal,
                "steps": trajectory.steps,
                "trajectory_side_effect": trajectory.side_effect_label,
            }, f)

        positives, negatives = build_eval_set(annotations, tmpdir, n_negatives_per_pos=1, seed=42)

        assert len(positives) == 1
        assert len(negatives) == 1
        # Should prefer the irreversible negative
        assert negatives[0].task_id == "neg_irrev"


def test_build_eval_set_is_deterministic():
    annotations = pd.DataFrame([
        {"benchmark": "WebArena", "task_id": "pos1", "model": "gpt-4", "trajectory_side_effect": "Yes"},
        {"benchmark": "WebArena", "task_id": "neg1", "model": "gpt-4", "trajectory_side_effect": "No"},
        {"benchmark": "WebArena", "task_id": "neg2", "model": "gpt-4", "trajectory_side_effect": "No"},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        for row in annotations.itertuples():
            path = Path(tmpdir) / f"{row.task_id}.json"
            trajectory = make_trajectory(row.task_id, row.trajectory_side_effect)
            with open(path, "w") as f:
                json.dump({
                    "goal": trajectory.goal,
                    "steps": trajectory.steps,
                    "trajectory_side_effect": trajectory.side_effect_label,
                }, f)

        # Run twice with same seed
        pos1, neg1 = build_eval_set(annotations, tmpdir, seed=42)
        pos2, neg2 = build_eval_set(annotations, tmpdir, seed=42)

        assert len(pos1) == len(pos2)
        assert len(neg1) == len(neg2)
        assert [p.task_id for p in pos1] == [p.task_id for p in pos2]
        assert [n.task_id for n in neg1] == [n.task_id for n in neg2]


def test_save_eval_set_creates_json_file():
    positives = [make_trajectory("pos1", "Yes")]
    negatives = [make_trajectory("neg1", "No")]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "eval_set.json")
        save_eval_set(positives, negatives, output_path)

        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            data = json.load(f)

        assert "positives" in data
        assert "negatives" in data
        assert len(data["positives"]) == 1
        assert len(data["negatives"]) == 1
        assert data["positives"][0]["task_id"] == "pos1"
        assert data["negatives"][0]["task_id"] == "neg1"