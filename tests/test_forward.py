from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset

from civic_mmlm import CIVICMMLM, load_config
from civic_mmlm.data.factory import build_synthetic_splits
from civic_mmlm.data.types import modalities_from_batch


def test_forward_shapes_and_finite() -> None:
    config = load_config("configs/demo.yaml")
    train_set, _, _ = build_synthetic_splits(config)
    batch = next(iter(DataLoader(Subset(train_set, range(4)), batch_size=4)))
    model = CIVICMMLM(config)
    output = model(
        modalities_from_batch(batch),
        batch["legality"],
        batch["budget"],
        batch["action_costs"],
        certificate_action=batch["label"],
        compute_certificate=True,
    )
    probs = output["decision"].probabilities
    assert probs.shape == (4, config["model"]["num_actions"])
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-5)
    assert output["barycenter"].slots.shape[1] == config["model"]["barycenter"]["num_slots"]
    assert torch.isfinite(output["barycenter"].modality_costs).all()
    assert output["certificate"].selected_gates.ndim == 2
