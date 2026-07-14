from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from civic_mmlm import CIVICMMLM, load_config
from civic_mmlm.data.factory import build_synthetic_splits
from civic_mmlm.data.types import modalities_from_batch
from civic_mmlm.training.losses import UnifiedObjective


def test_one_gradient_step() -> None:
    config = load_config("configs/demo.yaml")
    train_set, _, _ = build_synthetic_splits(config)
    batch = next(iter(DataLoader(Subset(train_set, range(4)), batch_size=4)))
    model = CIVICMMLM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def forward(key: str, certificate: bool):
        return model(
            modalities_from_batch(batch, key),
            batch["legality"],
            batch["budget"],
            batch["action_costs"],
            certificate_action=batch["label"] if certificate else None,
            compute_certificate=certificate,
        )

    output = forward("modalities", True)
    valid = forward("valid_modalities", False)
    material = forward("material_modalities", False)
    loss = UnifiedObjective(config)(output, valid, material, batch).total
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
