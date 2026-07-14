from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from torch import nn

from .evidence import ProcessedModality


@dataclass
class CertificateOutput:
    gates: torch.Tensor
    selected_gates: torch.Tensor
    certificate_state: torch.Tensor
    complement_state: torch.Tensor
    atom_features: torch.Tensor
    atom_mask: torch.Tensor
    atom_metadata: List[tuple[str, int]]


class MinimalEvidenceCertificate(nn.Module):
    """Differentiable atom gates and greedy audit certificate (Equations 18-21)."""

    def __init__(
        self,
        hidden_dim: int,
        num_actions: int,
        gate_temperature: float = 0.7,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.gate_temperature = float(gate_temperature)
        self.action_embeddings = nn.Parameter(torch.randn(num_actions, hidden_dim) * 0.03)
        self.atom_projection = nn.Linear(hidden_dim, hidden_dim)
        self.bias = nn.Parameter(torch.zeros(num_actions))

    @staticmethod
    def concatenate_atoms(
        processed: Dict[str, ProcessedModality], modality_names: Iterable[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[tuple[str, int]]]:
        features = []
        masks = []
        base_weights = []
        metadata: List[tuple[str, int]] = []
        for name in modality_names:
            item = processed[name]
            features.append(item.atoms)
            masks.append(item.mask)
            base_weights.append(item.weights)
            metadata.extend((name, index) for index in range(item.atoms.shape[1]))
        return (
            torch.cat(features, dim=1),
            torch.cat(masks, dim=1),
            torch.cat(base_weights, dim=1),
            metadata,
        )

    def forward(
        self,
        processed: Dict[str, ProcessedModality],
        modality_names: Iterable[str],
        action_index: torch.Tensor,
    ) -> CertificateOutput:
        atoms, mask, base_weights, metadata = self.concatenate_atoms(
            processed, modality_names
        )
        projected = self.atom_projection(atoms)
        logits = torch.einsum(
            "bnd,ad->bna", projected, self.action_embeddings
        ) + self.bias[None, None, :]
        logits = logits + torch.log(base_weights.clamp_min(1e-8))[:, :, None]
        gates = torch.sigmoid(logits / self.gate_temperature)
        gates = gates * mask[:, :, None].to(gates.dtype)
        selected = gates.gather(
            2,
            action_index[:, None, None].expand(-1, atoms.shape[1], 1),
        ).squeeze(-1)
        cert_weights = selected * base_weights * mask.to(selected.dtype)
        comp_weights = (1.0 - selected) * base_weights * mask.to(selected.dtype)
        cert_weights = cert_weights / cert_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        comp_weights = comp_weights / comp_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        certificate_state = torch.einsum("bn,bnd->bd", cert_weights, atoms)
        complement_state = torch.einsum("bn,bnd->bd", comp_weights, atoms)
        return CertificateOutput(
            gates=gates,
            selected_gates=selected,
            certificate_state=certificate_state,
            complement_state=complement_state,
            atom_features=atoms,
            atom_mask=mask,
            atom_metadata=metadata,
        )

    @torch.no_grad()
    def extract(
        self,
        certificate: CertificateOutput,
        batch_index: int,
        modality_inputs: dict,
        target_support: float = 0.8,
        max_atoms: int = 6,
    ) -> list[dict]:
        scores = certificate.selected_gates[batch_index].clone()
        scores = scores * certificate.atom_mask[batch_index].to(scores.dtype)
        total = scores.sum().clamp_min(1e-8)
        normalized = scores / total
        order = torch.argsort(normalized, descending=True)
        selected = []
        cumulative = 0.0
        for flat_index in order.tolist():
            if len(selected) >= max_atoms:
                break
            score = float(normalized[flat_index].item())
            if score <= 0.0:
                continue
            modality, atom_index = certificate.atom_metadata[flat_index]
            source = modality_inputs[modality]
            selected.append(
                {
                    "modality": modality,
                    "atom_index": int(atom_index),
                    "support": score,
                    "provenance": float(source["provenance"][batch_index, atom_index].item()),
                    "reliability": float(source["reliability"][batch_index, atom_index].item()),
                    "contradiction": float(source["contradiction"][batch_index, atom_index].item()),
                }
            )
            cumulative += score
            if cumulative >= target_support:
                break
        return selected
