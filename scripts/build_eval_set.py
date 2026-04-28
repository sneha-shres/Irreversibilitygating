"""Build the evaluation dataset from downloaded AgentRewardBench files."""

import argparse
import os

from irrgate.data.loader import load_annotations
from irrgate.data.sampler import build_eval_set, save_eval_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation dataset from AgentRewardBench.")
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing annotations.csv and trajectory files",
    )
    parser.add_argument(
        "--output",
        default="data/eval_set.json",
        help="Output path for evaluation dataset JSON",
    )
    args = parser.parse_args()

    annotations_path = os.path.join(args.data_dir, "annotations.csv")
    trajectory_dir = args.data_dir

    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    annotations = load_annotations(annotations_path)
    positives, negatives = build_eval_set(annotations, trajectory_dir)

    print(f"Built evaluation set: {len(positives)} positives, {len(negatives)} negatives")
    save_eval_set(positives, negatives, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
