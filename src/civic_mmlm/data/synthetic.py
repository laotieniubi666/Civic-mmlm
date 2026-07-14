from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticGovernanceDataset(Dataset):
    """Small, fully public proxy dataset used to test the complete code path.

    The data are deliberately synthetic. They are useful for software verification,
    unit tests, and ablations, but must not be reported as evidence for real public-
    service performance.
    """

    def __init__(
        self,
        size: int = 512,
        modalities: Iterable[str] = ("text", "image", "geo", "table"),
        raw_dim: int = 16,
        atoms_per_modality: int = 6,
        num_actions: int = 4,
        num_groups: int = 4,
        missing_probability: float = 0.12,
        contradiction_probability: float = 0.16,
        seed: int = 13,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.modalities = list(modalities)
        self.raw_dim = int(raw_dim)
        self.atoms_per_modality = int(atoms_per_modality)
        self.num_actions = int(num_actions)
        self.num_groups = int(num_groups)
        self.rng = np.random.default_rng(seed)

        prototypes = self.rng.normal(size=(num_actions, raw_dim)).astype(np.float32)
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
        self.prototypes = prototypes
        self.group_styles = self.rng.normal(
            scale=0.35, size=(num_groups, len(self.modalities), raw_dim)
        ).astype(np.float32)
        self.modality_offsets = self.rng.normal(
            scale=0.15, size=(len(self.modalities), raw_dim)
        ).astype(np.float32)
        self.action_costs = np.linspace(0.2, 1.0, num_actions, dtype=np.float32)

        self.samples = [
            self._make_sample(i, missing_probability, contradiction_probability)
            for i in range(self.size)
        ]

    def _make_modalities(
        self,
        label: int,
        group: int,
        alt_group: int,
        missing_probability: float,
        contradiction_probability: float,
    ) -> tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
        modalities: Dict[str, dict] = {}
        valid: Dict[str, dict] = {}
        material: Dict[str, dict] = {}
        material_label = (label + 1) % self.num_actions

        for m_idx, name in enumerate(self.modalities):
            n = self.atoms_per_modality
            semantic = self.prototypes[label] + self.modality_offsets[m_idx]
            semantic_material = self.prototypes[material_label] + self.modality_offsets[m_idx]
            noise = self.rng.normal(scale=0.30, size=(n, self.raw_dim)).astype(np.float32)
            atoms = semantic + self.group_styles[group, m_idx] + noise
            valid_atoms = semantic + self.group_styles[alt_group, m_idx] + noise
            material_atoms = semantic_material + self.group_styles[group, m_idx] + noise

            provenance = self.rng.beta(7.0, 2.0, size=n).astype(np.float32)
            reliability = self.rng.beta(6.0, 2.2, size=n).astype(np.float32)
            contradiction = np.zeros(n, dtype=np.float32)
            contradiction_mask = self.rng.random(n) < contradiction_probability
            if contradiction_mask.any():
                wrong = (label + 2) % self.num_actions
                atoms[contradiction_mask] = (
                    self.prototypes[wrong]
                    + self.modality_offsets[m_idx]
                    + self.rng.normal(
                        scale=0.25,
                        size=(int(contradiction_mask.sum()), self.raw_dim),
                    )
                )
                valid_atoms[contradiction_mask] = atoms[contradiction_mask]
                contradiction[contradiction_mask] = self.rng.uniform(
                    0.65, 1.0, size=int(contradiction_mask.sum())
                )
                reliability[contradiction_mask] *= 0.55

            mask = np.ones(n, dtype=bool)
            if self.rng.random() < missing_probability:
                mask[:] = False
            else:
                atom_dropout = self.rng.random(n) < 0.08
                mask[atom_dropout] = False
                if not mask.any():
                    mask[self.rng.integers(0, n)] = True

            common = {
                "mask": torch.from_numpy(mask),
                "provenance": torch.from_numpy(provenance),
                "reliability": torch.from_numpy(reliability),
                "contradiction": torch.from_numpy(contradiction),
            }
            modalities[name] = {"atoms": torch.from_numpy(atoms), **common}
            valid[name] = {"atoms": torch.from_numpy(valid_atoms), **common}
            material[name] = {"atoms": torch.from_numpy(material_atoms), **common}

        return modalities, valid, material

    def _make_sample(
        self,
        index: int,
        missing_probability: float,
        contradiction_probability: float,
    ) -> dict:
        label = int(self.rng.integers(0, self.num_actions))
        group = int(self.rng.integers(0, self.num_groups))
        alt_group = int((group + self.rng.integers(1, self.num_groups)) % self.num_groups)
        modalities, valid_modalities, material_modalities = self._make_modalities(
            label,
            group,
            alt_group,
            missing_probability,
            contradiction_probability,
        )

        legality = np.ones(self.num_actions, dtype=np.float32)
        for action in range(self.num_actions):
            if action != label and self.rng.random() < 0.18:
                legality[action] = 0.0
        legality[label] = 1.0
        budget = float(
            min(1.25, self.action_costs[label] + self.rng.uniform(0.05, 0.35))
        )

        outcomes = np.zeros((self.num_actions, 4), dtype=np.float32)
        for action in range(self.num_actions):
            distance = min(
                abs(action - label),
                self.num_actions - abs(action - label),
            )
            benefit = 1.25 - 0.38 * distance + self.rng.normal(scale=0.05)
            delay = 0.18 + 0.12 * action + self.rng.normal(scale=0.03)
            harm = 0.08 + 0.24 * distance + self.rng.normal(scale=0.03)
            cost = self.action_costs[action] + self.rng.normal(scale=0.02)
            outcomes[action] = [benefit, delay, harm, cost]

        propensity_logits = np.full(self.num_actions, -0.8, dtype=np.float32)
        propensity_logits[label] = 1.2
        propensity_logits -= propensity_logits.max()
        propensity = np.exp(propensity_logits)
        propensity /= propensity.sum()
        factual_action = int(self.rng.choice(self.num_actions, p=propensity))
        factual_outcome = outcomes[factual_action] + self.rng.normal(
            scale=0.04, size=4
        ).astype(np.float32)
        identification = int(self.rng.choice([0, 1, 2], p=[0.05, 0.25, 0.70]))

        return {
            "sample_id": f"synthetic-{index:06d}",
            "modalities": modalities,
            "valid_modalities": valid_modalities,
            "material_modalities": material_modalities,
            "label": torch.tensor(label, dtype=torch.long),
            "material_label": torch.tensor(
                (label + 1) % self.num_actions, dtype=torch.long
            ),
            "group": torch.tensor(group, dtype=torch.long),
            "need_bin": torch.tensor(label, dtype=torch.long),
            "legality": torch.from_numpy(legality),
            "budget": torch.tensor(budget, dtype=torch.float32),
            "action_costs": torch.from_numpy(self.action_costs.copy()),
            "outcomes": torch.from_numpy(outcomes),
            "factual_action": torch.tensor(factual_action, dtype=torch.long),
            "factual_outcome": torch.from_numpy(factual_outcome),
            "propensity": torch.from_numpy(propensity.astype(np.float32)),
            "identification": torch.tensor(identification, dtype=torch.long),
        }

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def corrupt_modalities(batch_modalities: Dict[str, dict], rate: float, seed: int) -> Dict[str, dict]:
    """Return a copy with an identical Bernoulli evidence mask for stress tests."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result = deepcopy(batch_modalities)
    for value in result.values():
        original = value["mask"].bool()
        random_keep = torch.rand(original.shape, generator=generator) >= rate
        value["mask"] = original & random_keep
    return result
