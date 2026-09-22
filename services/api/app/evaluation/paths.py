"""Where the evaluation artefacts live.

All three directories are mounted from the repository into the container
(`docker-compose.yml`), because a dataset, a sweep report and a baseline are
things you commit and review - not container state. Resolving them here rather
than in each script means one place decides, and the environment variables are
documented once.
"""

from __future__ import annotations

import os
from pathlib import Path

DATASET_DIR = Path(os.environ.get("EVAL_DATASET_DIR", "evaluation/datasets/questions"))
EXPERIMENT_DIR = Path(os.environ.get("EVAL_EXPERIMENT_DIR", "evaluation/experiments"))
REPORT_DIR = Path(os.environ.get("EVAL_REPORT_DIR", "evaluation/reports"))

BASELINE = REPORT_DIR / "baseline.json"
DEFAULT_VERSION = "v1"


def dataset_path(version: str = DEFAULT_VERSION) -> Path:
    """The file for a dataset version.

    The version is used as a filename, so it is validated rather than trusted:
    it arrives from a request body, and a version of `../../etc/passwd` would
    otherwise be a path traversal into whatever the process can read.
    """
    if not version or not version.replace(".", "").replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Invalid dataset version: {version!r}")
    return DATASET_DIR / f"{version}.jsonl"
