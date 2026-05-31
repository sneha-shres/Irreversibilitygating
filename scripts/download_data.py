from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from irrgate.config import ANNOTATIONS_PATH, DATASET_REPO, TARGET_BENCHMARKS
from irrgate.data.loader import load_annotations


def download_repo_file(repo_path: str, output_dir: Path, max_attempts: int = 3) -> Path:
    output_path = output_dir / repo_path
    if output_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        try:
            downloaded = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=repo_path,
                repo_type="dataset",
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,
            )
            return Path(downloaded)
        except (HfHubHTTPError, OSError) as exc:
            logging.warning("Attempt %d/%d failed downloading %s: %s", attempt, max_attempts, repo_path, exc)
            if attempt == max_attempts:
                raise
            time.sleep(2**attempt)


def trajectory_candidates(benchmark: str, model: str, task_id: str) -> list[str]:
    key = benchmark.lower()
    candidates: list[str] = []

    if key == "webarena":
        candidates.extend([
            f"cleaned/webarena/{model}/{model}_on_webarena/{task_id}.json",
            f"cleaned/webarena/{model}/{model}_on_webarena/webarena.{task_id}.json",
        ])
        if task_id.startswith("webarena."):
            stripped = task_id.split(".", 1)[1]
            candidates.extend([
                f"cleaned/webarena/{model}/{model}_on_webarena/{stripped}.json",
                f"cleaned/webarena/{model}/{model}_on_webarena/webarena.{stripped}.json",
            ])
    elif key == "workarena":
        candidates.extend([
            f"cleaned/workarena/{model}/{model}_on_workarena.servicenow/{task_id}.json",
            f"cleaned/workarena/{model}/{model}_on_workarena.servicenow/workarena.servicenow.{task_id}.json",
        ])
        if task_id.startswith("workarena.servicenow."):
            stripped = task_id.split(".", 2)[-1]
            candidates.extend([
                f"cleaned/workarena/{model}/{model}_on_workarena.servicenow/{stripped}.json",
                f"cleaned/workarena/{model}/{model}_on_workarena.servicenow/workarena.servicenow.{stripped}.json",
            ])

    return candidates


def download_data(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = download_repo_file(ANNOTATIONS_PATH, output_dir)

    annotations = load_annotations(str(annotations_path))
    filtered = annotations[annotations["benchmark"].str.lower().isin(TARGET_BENCHMARKS)]

    logging.info("Preparing to download %d trajectories from %s", len(filtered), DATASET_REPO)

    for _, row in filtered.iterrows():
        benchmark = str(row["benchmark"]).strip()
        model = str(row["model"]).strip()
        task_id = str(row["task_id"]).strip()
        logging.info("Downloading trajectory for %s / %s", benchmark, task_id)

        for candidate in trajectory_candidates(benchmark, model, task_id):
            try:
                download_repo_file(candidate, output_dir)
                break
            except Exception:
                continue
        else:
            raise FileNotFoundError(
                f"Could not find trajectory file for task_id={task_id} model={model} benchmark={benchmark}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AgentRewardBench WebArena and WorkArena files.")
    parser.add_argument("--output-dir", default="data/raw", help="Destination directory for downloaded annotations and trajectories")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download_data(Path(args.output_dir))


if __name__ == "__main__":
    main()
