from __future__ import annotations

from copy import deepcopy

import torch


def apply_atom_dropout(modalities: dict, rate: float, seed: int) -> dict:
    output = deepcopy(modalities)
    generator = torch.Generator()
    generator.manual_seed(seed)
    for value in output.values():
        keep = torch.rand(value["mask"].shape, generator=generator) >= rate
        value["mask"] = value["mask"].bool() & keep
    return output
