#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civic_mmlm import CIVICMMLM
from civic_mmlm.data.factory import build_synthetic_splits
from civic_mmlm.data.types import modalities_from_batch
from civic_mmlm.models.conformal import ConformalAbstentionCalibrator
from civic_mmlm.training.trainer import move_batch_to_device
from civic_mmlm.utils.io import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit split-conformal abstention threshold")
    parser.add_argument("--checkpoint", default=str(ROOT / "outputs" / "demo" / "best_model.pt"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "demo" / "calibrator.json"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = CIVICMMLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    _, dev_set, _ = build_synthetic_splits(config)
    loader = DataLoader(dev_set, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    calibrator = ConformalAbstentionCalibrator(
        alpha=float(config["calibration"].get("alpha", 0.2)),
        score_weights=config["calibration"].get("score_weights"),
    )
    scores = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            output = model(
                modalities_from_batch(batch),
                batch["legality"],
                batch["budget"],
                batch["action_costs"],
                compute_certificate=False,
            )
            score = calibrator.score(
                output["decision"].entropy,
                output["barycenter"].disagreement,
                batch["identification"],
                output["decision"].constraint_slack,
            )
            scores.append(score.cpu())
    threshold = calibrator.fit(torch.cat(scores))
    calibrator.save(args.output)
    print(f"calibration_size={calibrator.calibration_size} threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
