"""Report issue-tag precision against completed human annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.issue_audit import evaluate_annotations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "issue_tag_annotations.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    args = parser.parse_args()
    if not args.annotations.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {args.annotations}. "
            "Create and complete an audit sample in the Issue Tag Audit page first."
        )
    annotations = pd.read_csv(
        args.annotations,
        dtype={"human_tags": "string", "predicted_tags": "string"},
        keep_default_na=False,
    )
    metrics = evaluate_annotations(annotations)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
