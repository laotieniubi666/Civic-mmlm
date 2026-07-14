#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civic_mmlm import CIVICMMLM
from civic_mmlm.data.factory import build_synthetic_splits
from civic_mmlm.data.types import modalities_from_batch
from civic_mmlm.evaluation.metrics import classification_metrics, risk_coverage_curve
from civic_mmlm.evaluation.stress import apply_atom_dropout
from civic_mmlm.models.conformal import ConformalAbstentionCalibrator
from civic_mmlm.training.trainer import move_batch_to_device
from civic_mmlm.utils.io import ensure_dir, load_checkpoint, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the public synthetic demo")
    parser.add_argument("--checkpoint", default=str(ROOT / "outputs" / "demo" / "best_model.pt"))
    parser.add_argument("--calibrator", default=str(ROOT / "outputs" / "demo" / "calibrator.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "demo"))
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def infer(model, loader, device, calibrator, corruption_rate: float = 0.0):
    probabilities, labels, groups, needs, scores, accepted = [], [], [], [], [], []
    first_certificate = None
    correct_accepted = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            batch = move_batch_to_device(raw_batch, device)
            if corruption_rate > 0:
                batch["modalities"] = apply_atom_dropout(
                    batch["modalities"], corruption_rate, seed=1000 + batch_index
                )
            output = model(
                modalities_from_batch(batch),
                batch["legality"],
                batch["budget"],
                batch["action_costs"],
                compute_certificate=True,
            )
            probs = output["decision"].probabilities
            score = calibrator.score(
                output["decision"].entropy,
                output["barycenter"].disagreement,
                batch["identification"],
                output["decision"].constraint_slack,
            )
            accept = calibrator.accept(score)
            probabilities.append(probs.cpu().numpy())
            labels.append(batch["label"].cpu().numpy())
            groups.append(batch["group"].cpu().numpy())
            needs.append(batch["need_bin"].cpu().numpy())
            scores.append(score.cpu().numpy())
            accepted.append(accept.cpu().numpy())
            predictions = probs.argmax(-1)
            correct_accepted.extend(
                ((predictions == batch["label"]) & accept).cpu().tolist()
            )
            if first_certificate is None:
                target_action = int(predictions[0].item())
                first_certificate = {
                    "sample_id": raw_batch["sample_id"][0],
                    "predicted_action": target_action,
                    "accepted": bool(accept[0].item()),
                    "nonconformity_score": float(score[0].item()),
                    "threshold": float(calibrator.threshold),
                    "evidence": model.certificate.extract(
                        output["certificate"],
                        0,
                        batch["modalities"],
                        target_support=0.8,
                        max_atoms=6,
                    ),
                }
    return {
        "probabilities": np.concatenate(probabilities),
        "labels": np.concatenate(labels),
        "groups": np.concatenate(groups),
        "needs": np.concatenate(needs),
        "scores": np.concatenate(scores),
        "accepted": np.concatenate(accepted),
        "certificate": first_certificate,
    }


def main() -> None:
    torch.set_num_threads(1)
    args = parse_args()
    output_dir = ensure_dir(args.output)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = CIVICMMLM(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    calibrator = ConformalAbstentionCalibrator.load(args.calibrator)
    _, _, test_set = build_synthetic_splits(config)
    loader = DataLoader(test_set, batch_size=int(config["training"]["batch_size"]), shuffle=False)

    result = infer(model, loader, device, calibrator)
    metrics = classification_metrics(
        result["probabilities"], result["labels"], result["groups"], result["needs"]
    )
    accept = result["accepted"]
    predictions = result["probabilities"].argmax(axis=1)
    metrics["coverage"] = float(accept.mean())
    metrics["selective_risk"] = (
        float(1.0 - (predictions[accept] == result["labels"][accept]).mean())
        if accept.any()
        else float("nan")
    )
    curve = risk_coverage_curve(
        result["probabilities"], result["labels"], result["scores"]
    )

    stress_results = []
    for rate in (0.0, 0.2, 0.4, 0.6):
        stressed = infer(model, loader, device, calibrator, corruption_rate=rate)
        stressed_metrics = classification_metrics(
            stressed["probabilities"], stressed["labels"], stressed["groups"], stressed["needs"]
        )
        stress_results.append(
            {"missing_atom_rate": rate, "accuracy": stressed_metrics["accuracy"]}
        )

    payload = {
        "scope": "synthetic software-verification benchmark",
        "metrics": metrics,
        "risk_coverage": curve,
        "robustness": stress_results,
    }
    save_json(payload, output_dir / "metrics.json")
    save_json(result["certificate"], output_dir / "certificate_example.json")

    plt.figure(figsize=(6.0, 4.0))
    plt.plot([item["coverage"] for item in curve], [item["risk"] for item in curve], marker="o")
    plt.xlabel("Coverage")
    plt.ylabel("Selective risk")
    plt.title("Synthetic risk-coverage curve")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_coverage.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.0, 4.0))
    plt.plot(
        [item["missing_atom_rate"] for item in stress_results],
        [item["accuracy"] for item in stress_results],
        marker="o",
    )
    plt.xlabel("Missing evidence-atom rate")
    plt.ylabel("Accuracy")
    plt.title("Synthetic missing-evidence stress test")
    plt.tight_layout()
    plt.savefig(output_dir / "robustness.png", dpi=180)
    plt.close()

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
