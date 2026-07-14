#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from civic_mmlm import CIVICMMLM, load_config
from civic_mmlm.data.factory import build_synthetic_splits
from civic_mmlm.training.trainer import Trainer
from civic_mmlm.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the synthetic CIVIC-MMLM demo")
    parser.add_argument("--config", default=str(ROOT / "configs" / "demo.yaml"))
    parser.add_argument("--output", default=str(ROOT / "outputs" / "demo"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 7)), num_threads=int(config.get("runtime", {}).get("num_threads", 1)))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_set, dev_set, _ = build_synthetic_splits(config)
    batch_size = int(config["training"].get("batch_size", 32))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)
    model = CIVICMMLM(config)
    trainer = Trainer(model, config, device, args.output)
    print(f"device={device} train={len(train_set)} dev={len(dev_set)}")
    trainer.fit(train_loader, dev_loader)


if __name__ == "__main__":
    main()
