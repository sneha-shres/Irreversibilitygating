from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from irrgate.data.loader import Trajectory
from irrgate.data.sampler import build_eval_set, save_eval_set

QWEN = "GenericAgent-Qwen_Qwen2.5-VL-72B-Instruct"
GPT4 = "GenericAgent-gpt-4o-2024-11-20"


def make_trajectory(task_id: str, side_effect: str, model: str = QWEN) -> Trajectory:
    return Trajectory(
        benchmark="webarena",
        task_id=task_id,
        model=model,
        goal="Test",
        steps=[{"action": "noop()", "axtree": "", "url": "", "reasoning": "", "bounding_boxes": []}],
        side_effect_label=side_effect,
    )


def _write_trajectory_file(tmpdir: str, benchmark: str, model: str, task_id: str, side_effect: str) -> None:
    """Write a trajectory file at the expected cleaned/{benchmark}/{model}/ path."""
    path = Path(tmpdir) / "cleaned" / benchmark / model / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "goal": "Test",
            "steps": [{"action": "noop()", "axtree": "", "url": "", "reasoning": "", "bounding_boxes": []}],
            "trajectory_side_effect": side_effect,
        }, f)


def test_build_eval_set_returns_correct_positives_and_negatives():
    annotations = pd.DataFrame([
        {"benchmark": "webarena", "task_id": "pos1", "model": QWEN, "trajectory_side_effect": "Yes"},
        {"benchmark": "webarena", "task_id": "pos2", "model": QWEN, "trajectory_side_effect": "Yes"},
        {"benchmark": "webarena", "task_id": "neg1", "model": QWEN, "trajectory_side_effect": "No"},
        {"benchmark": "webarena", "task_id": "neg2", "model": QWEN, "trajectory_side_effect": "No"},
        {"benchmark": "other",    "task_id": "other", "model": QWEN, "trajectory_side_effect": "Yes"},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        for _, row in annotations.iterrows():
            if row["benchmark"] in {"webarena", "workarena"}:
                _write_trajectory_file(tmpdir, row["benchmark"], row["model"], row["task_id"], row["trajectory_side_effect"])

        positives, negatives = build_eval_set(annotations, tmpdir)

        assert len(positives) == 2
        assert all(p.side_effect_label == "Yes" for p in positives)
        assert len(negatives) == 2
        assert all(n.side_effect_label == "No" for n in negatives)


def test_build_eval_set_deduplicates_multi_annotator_runs():
    """Same (task_id, model) annotated twice: should appear once, positive if any Yes."""
    annotations = pd.DataFrame([
        {"benchmark": "webarena", "task_id": "t1", "model": QWEN, "trajectory_side_effect": "Yes"},
        {"benchmark": "webarena", "task_id": "t1", "model": QWEN, "trajectory_side_effect": "No"},
        {"benchmark": "webarena", "task_id": "t2", "model": QWEN, "trajectory_side_effect": "No"},
        {"benchmark": "webarena", "task_id": "t2", "model": QWEN, "trajectory_side_effect": "No"},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_trajectory_file(tmpdir, "webarena", QWEN, "t1", "Yes")
        _write_trajectory_file(tmpdir, "webarena", QWEN, "t2", "No")

        positives, negatives = build_eval_set(annotations, tmpdir)

        assert len(positives) == 1
        assert positives[0].task_id == "t1"
        assert len(negatives) == 1
        assert negatives[0].task_id == "t2"


def test_build_eval_set_treats_same_task_different_model_as_distinct_runs():
    """Same task_id run by two models = two independent trajectories."""
    annotations = pd.DataFrame([
        {"benchmark": "webarena", "task_id": "t1", "model": QWEN,  "trajectory_side_effect": "Yes"},
        {"benchmark": "webarena", "task_id": "t1", "model": GPT4,  "trajectory_side_effect": "No"},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_trajectory_file(tmpdir, "webarena", QWEN, "t1", "Yes")
        _write_trajectory_file(tmpdir, "webarena", GPT4, "t1", "No")

        positives, negatives = build_eval_set(annotations, tmpdir)

        assert len(positives) == 1
        assert positives[0].model == QWEN
        assert len(negatives) == 1
        assert negatives[0].model == GPT4


def test_build_eval_set_excludes_runs_without_trajectory_file():
    annotations = pd.DataFrame([
        {"benchmark": "webarena", "task_id": "has_file",    "model": QWEN, "trajectory_side_effect": "Yes"},
        {"benchmark": "webarena", "task_id": "missing_file","model": QWEN, "trajectory_side_effect": "Yes"},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_trajectory_file(tmpdir, "webarena", QWEN, "has_file", "Yes")
        # no file for missing_file

        positives, negatives = build_eval_set(annotations, tmpdir)

        assert len(positives) == 1
        assert positives[0].task_id == "has_file"


def test_build_eval_set_is_deterministic():
    annotations = pd.DataFrame([
        {"benchmark": "webarena", "task_id": "pos1", "model": QWEN, "trajectory_side_effect": "Yes"},
        {"benchmark": "webarena", "task_id": "neg1", "model": QWEN, "trajectory_side_effect": "No"},
        {"benchmark": "webarena", "task_id": "neg2", "model": QWEN, "trajectory_side_effect": "No"},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        for _, row in annotations.iterrows():
            _write_trajectory_file(tmpdir, row["benchmark"], row["model"], row["task_id"], row["trajectory_side_effect"])

        pos1, neg1 = build_eval_set(annotations, tmpdir)
        pos2, neg2 = build_eval_set(annotations, tmpdir)

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
