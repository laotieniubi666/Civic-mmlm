from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
from torch import nn

from civic_mmlm.data.types import EvidenceTensor, Modalities


@dataclass
class ProcessedModality:
    atoms: torch.Tensor
    mask: torch.Tensor
    weights: torch.Tensor
    summary: torch.Tensor
    provenance_mean: torch.Tensor
    reliability_mean: torch.Tensor
    contradiction_mean: torch.Tensor
    available: torch.Tensor


class ProvenanceAwareEvidenceEncoder(nn.Module):
    """Equations (3)-(4): transport and reliability-weight evidence atoms."""

    def __init__(
        self,
        modality_names: Iterable[str],
        raw_dim: int,
        hidden_dim: int,
        temperature: float = 0.7,
        alpha_provenance: float = 0.65,
        alpha_reliability: float = 0.65,
        alpha_contradiction: float = 1.0,
    ) -> None:
        super().__init__()
        self.modality_names = list(modality_names)
        self.temperature = float(temperature)
        self.alpha_provenance = float(alpha_provenance)
        self.alpha_reliability = float(alpha_reliability)
        self.alpha_contradiction = float(alpha_contradiction)
        self.transports = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(raw_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for name in self.modality_names
            }
        )
        self.semantic_scores = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, 1) for name in self.modality_names}
        )

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
        masked = logits.masked_fill(~mask, -1e4)
        probs = torch.softmax(masked, dim=dim) * mask.to(logits.dtype)
        denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)
        return probs / denom

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(values.dtype)
        return (values * mask_f).sum(-1) / mask_f.sum(-1).clamp_min(1.0)

    def forward(self, modalities: Modalities) -> Dict[str, ProcessedModality]:
        outputs: Dict[str, ProcessedModality] = {}
        for name in self.modality_names:
            evidence: EvidenceTensor = modalities[name]
            atoms = self.transports[name](evidence.atoms)
            semantic = self.semantic_scores[name](atoms).squeeze(-1)
            score = (
                semantic
                + self.alpha_provenance * torch.log(evidence.provenance.clamp_min(1e-6))
                + self.alpha_reliability * torch.log(evidence.reliability.clamp_min(1e-6))
                - self.alpha_contradiction * evidence.contradiction
            )
            weights = self._masked_softmax(score / self.temperature, evidence.mask)
            summary = torch.einsum("bn,bnd->bd", weights, atoms)
            available = evidence.mask.any(dim=1)
            summary = summary * available[:, None].to(summary.dtype)
            outputs[name] = ProcessedModality(
                atoms=atoms,
                mask=evidence.mask,
                weights=weights,
                summary=summary,
                provenance_mean=self._masked_mean(evidence.provenance, evidence.mask),
                reliability_mean=self._masked_mean(evidence.reliability, evidence.mask),
                contradiction_mean=self._masked_mean(evidence.contradiction, evidence.mask),
                available=available,
            )
        return outputs


class ModalityGating(nn.Module):
    """Decision-conditioned modality weights inspired by Equation (7)."""

    def __init__(self, hidden_dim: int, num_actions: int, modality_names: Iterable[str]) -> None:
        super().__init__()
        self.modality_names = list(modality_names)
        self.num_actions = int(num_actions)
        feature_dim = hidden_dim + 3
        self.action_vectors = nn.Parameter(torch.empty(num_actions, feature_dim))
        self.global_vector = nn.Parameter(torch.empty(feature_dim))
        nn.init.normal_(self.action_vectors, std=0.02)
        nn.init.normal_(self.global_vector, std=0.02)

    def forward(
        self, processed: Dict[str, ProcessedModality]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = []
        available = []
        summaries = []
        for name in self.modality_names:
            item = processed[name]
            meta = torch.stack(
                [
                    torch.log(item.provenance_mean.clamp_min(1e-6)),
                    torch.log(item.reliability_mean.clamp_min(1e-6)),
                    -item.contradiction_mean,
                ],
                dim=-1,
            )
            features.append(torch.cat([item.summary, meta], dim=-1))
            summaries.append(item.summary)
            available.append(item.available)
        feature_tensor = torch.stack(features, dim=1)  # [B, M, F]
        summary_tensor = torch.stack(summaries, dim=1)  # [B, M, D]
        available_tensor = torch.stack(available, dim=1)  # [B, M]

        global_logits = torch.einsum("bmf,f->bm", feature_tensor, self.global_vector)
        global_logits = global_logits.masked_fill(~available_tensor, -1e4)
        global_weights = torch.softmax(global_logits, dim=1)
        all_missing = ~available_tensor.any(dim=1)
        if all_missing.any():
            global_weights = global_weights.clone()
            global_weights[all_missing] = 1.0 / len(self.modality_names)

        action_logits = torch.einsum("bmf,af->bam", feature_tensor, self.action_vectors)
        action_logits = action_logits.masked_fill(~available_tensor[:, None, :], -1e4)
        action_weights = torch.softmax(action_logits, dim=-1)
        if all_missing.any():
            action_weights = action_weights.clone()
            action_weights[all_missing] = 1.0 / len(self.modality_names)
        action_context = torch.einsum("bam,bmd->bad", action_weights, summary_tensor)
        return global_weights, action_weights, action_context
