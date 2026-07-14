from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class EvidenceTensor:
    """A padded batch of evidence atoms for one modality."""

    atoms: torch.Tensor
    mask: torch.Tensor
    provenance: torch.Tensor
    reliability: torch.Tensor
    contradiction: torch.Tensor

    def to(self, device: torch.device | str) -> "EvidenceTensor":
        return EvidenceTensor(
            atoms=self.atoms.to(device),
            mask=self.mask.to(device),
            provenance=self.provenance.to(device),
            reliability=self.reliability.to(device),
            contradiction=self.contradiction.to(device),
        )


Modalities = Dict[str, EvidenceTensor]


def modalities_from_batch(batch: dict, key: str = "modalities") -> Modalities:
    result: Modalities = {}
    for name, value in batch[key].items():
        result[name] = EvidenceTensor(
            atoms=value["atoms"],
            mask=value["mask"].bool(),
            provenance=value["provenance"],
            reliability=value["reliability"],
            contradiction=value["contradiction"],
        )
    return result
