from __future__ import annotations

from torch.utils.data import DataLoader, Subset

from civic_mmlm import CIVICMMLM, load_config
from civic_mmlm.data.factory import build_synthetic_splits
from civic_mmlm.data.types import modalities_from_batch


def test_certificate_export() -> None:
    config = load_config("configs/demo.yaml")
    train_set, _, _ = build_synthetic_splits(config)
    batch = next(iter(DataLoader(Subset(train_set, [0]), batch_size=1)))
    model = CIVICMMLM(config)
    output = model(
        modalities_from_batch(batch),
        batch["legality"],
        batch["budget"],
        batch["action_costs"],
        certificate_action=batch["label"],
        compute_certificate=True,
    )
    certificate = model.certificate.extract(
        output["certificate"], 0, batch["modalities"], max_atoms=4
    )
    assert 1 <= len(certificate) <= 4
    assert {"modality", "atom_index", "support", "provenance"}.issubset(certificate[0])
