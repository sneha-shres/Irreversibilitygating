from __future__ import annotations

import os
import pandas as pd

from irrgate.data.loader import Trajectory, iter_trajectories, load_annotations, load_trajectory


def test_load_annotations(tmp_path):
    csv_path = tmp_path / "annotations.csv"
    csv_path.write_text(
        "benchmark,task_id,model,trajectory_side_effect\n"
        "WebArena,webarena.690,gpt-4,Yes\n"
        "WorkArena,workarena.servicenow.foo-l2,gpt-4,No\n"
    )

    df = load_annotations(str(csv_path))

    assert df.shape[0] == 2
    assert df.loc[0, "benchmark"] == "WebArena"
    assert df.loc[1, "task_id"] == "workarena.servicenow.foo-l2"
    assert df.loc[1, "trajectory_side_effect"] == "No"


def test_load_trajectory_from_fixture():
    fixture_path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "webarena.690.json",
    )
    trajectory = load_trajectory(fixture_path)

    assert isinstance(trajectory, Trajectory)
    assert trajectory.task_id == "webarena.690"
    assert trajectory.goal == "Submit the contact form"
    assert trajectory.side_effect_label == "Yes"
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0]["action"] == "click('1976')"


def test_iter_trajectories_loads_matching_files():
    annotations = pd.DataFrame(
        [
            {"benchmark": "WebArena", "task_id": "webarena.690", "model": "gpt-4", "trajectory_side_effect": "Yes"},
            {"benchmark": "WorkArena", "task_id": "workarena.servicenow.foo-l2", "model": "gpt-4", "trajectory_side_effect": "No"},
            {"benchmark": "OtherBench", "task_id": "other.1", "model": "gpt-4", "trajectory_side_effect": "No"},
        ]
    )

    trajectory_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    loaded = list(iter_trajectories(annotations, trajectory_dir))

    assert len(loaded) == 2
    assert loaded[0].benchmark == "WebArena"
    assert loaded[1].benchmark == "WorkArena"
