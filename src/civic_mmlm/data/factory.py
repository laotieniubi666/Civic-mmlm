from __future__ import annotations

import torch
from torch.utils.data import Subset

from .synthetic import SyntheticGovernanceDataset


def build_synthetic_splits(config: dict) -> tuple[Subset, Subset, Subset]:
    data_cfg = config["data"]
    model_cfg = config["model"]
    dataset = SyntheticGovernanceDataset(
        size=int(data_cfg.get("size", 768)),
        modalities=model_cfg["modalities"],
        raw_dim=int(model_cfg["raw_dim"]),
        atoms_per_modality=int(data_cfg.get("atoms_per_modality", 6)),
        num_actions=int(model_cfg["num_actions"]),
        num_groups=int(data_cfg.get("num_groups", 4)),
        missing_probability=float(data_cfg.get("missing_probability", 0.12)),
        contradiction_probability=float(
            data_cfg.get("contradiction_probability", 0.16)
        ),
        seed=int(data_cfg.get("seed", 13)),
    )
    train_fraction = float(data_cfg.get("train_fraction", 0.65))
    dev_fraction = float(data_cfg.get("dev_fraction", 0.175))
    total = len(dataset)
    train_size = int(total * train_fraction)
    dev_size = int(total * dev_fraction)
    test_size = total - train_size - dev_size
    generator = torch.Generator().manual_seed(int(data_cfg.get("split_seed", 31)))
    permutation = torch.randperm(total, generator=generator).tolist()
    train_indices = permutation[:train_size]
    dev_indices = permutation[train_size : train_size + dev_size]
    test_indices = permutation[train_size + dev_size : train_size + dev_size + test_size]
    return (
        Subset(dataset, train_indices),
        Subset(dataset, dev_indices),
        Subset(dataset, test_indices),
    )
