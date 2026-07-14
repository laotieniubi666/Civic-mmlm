#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civic_mmlm.data.registry import DATASET_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display acquisition requirements for an EGD-BENCH resource"
    )
    parser.add_argument("--dataset", choices=["all", *DATASET_REGISTRY.keys()], default="all")
    args = parser.parse_args()
    keys = DATASET_REGISTRY.keys() if args.dataset == "all" else [args.dataset]
    records = {
        key: {
            "name": DATASET_REGISTRY[key].name,
            "task": DATASET_REGISTRY[key].task,
            "access": DATASET_REGISTRY[key].access,
            "notes": DATASET_REGISTRY[key].notes,
        }
        for key in keys
    }
    print(json.dumps(records, ensure_ascii=False, indent=2))
    print(
        "\nThis command intentionally does not auto-download license-gated or very large datasets. "
        "See docs/DATASETS.md for the manifest format and adapter contract."
    )


if __name__ == "__main__":
    main()
