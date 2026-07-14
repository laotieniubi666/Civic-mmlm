#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civic_mmlm import CIVICMMLM
from civic_mmlm.data.factory import build_synthetic_splits
from civic_mmlm.data.types import modalities_from_batch
from civic_mmlm.models.conformal import ConformalAbstentionCalibrator
from civic_mmlm.training.trainer import move_batch_to_device
from civic_mmlm.utils.io import load_checkpoint


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Run one auditable synthetic prediction")
    parser.add_argument("--checkpoint", default=str(ROOT / "outputs" / "demo" / "best_model.pt"))
    parser.add_argument("--calibrator", default=str(ROOT / "outputs" / "demo" / "calibrator.json"))
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    model = CIVICMMLM(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    calibrator = ConformalAbstentionCalibrator.load(args.calibrator)
    _, _, test_set = build_synthetic_splits(config)
    sample_index = args.index % len(test_set)
    loader = DataLoader(Subset(test_set, [sample_index]), batch_size=1)
    raw_batch = next(iter(loader))
    batch = move_batch_to_device(raw_batch, torch.device("cpu"))
    with torch.no_grad():
        output = model(
            modalities_from_batch(batch),
            batch["legality"],
            batch["budget"],
            batch["action_costs"],
            compute_certificate=True,
        )
        score = calibrator.score(
            output["decision"].entropy,
            output["barycenter"].disagreement,
            batch["identification"],
            output["decision"].constraint_slack,
        )
        prediction = int(output["decision"].probabilities.argmax(-1).item())
        response = {
            "sample_id": raw_batch["sample_id"][0],
            "prediction": prediction,
            "reference_label": int(batch["label"].item()),
            "probabilities": output["decision"].probabilities[0].tolist(),
            "accepted": bool(calibrator.accept(score)[0].item()),
            "nonconformity_score": float(score.item()),
            "threshold": float(calibrator.threshold),
            "certificate": model.certificate.extract(
                output["certificate"], 0, batch["modalities"]
            ),
        }
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
