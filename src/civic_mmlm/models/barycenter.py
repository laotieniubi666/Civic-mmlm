from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch
from torch import nn

from .evidence import ProcessedModality


@dataclass
class BarycenterOutput:
    slots: torch.Tensor
    weights: torch.Tensor
    summary: torch.Tensor
    modality_costs: torch.Tensor
    transported_mass: torch.Tensor
    disagreement: torch.Tensor


class UnbalancedEvidenceBarycenter(nn.Module):
    """Differentiable unbalanced entropic barycenter approximation.

    The implementation uses a compact set of barycenter support slots and generalized
    Sinkhorn scaling with KL-relaxed marginals. It follows the mechanism described by
    Equations (5)-(6), while remaining small enough for the public CPU demo.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_slots: int = 4,
        epsilon: float = 0.35,
        rho: float = 1.0,
        sinkhorn_iterations: int = 5,
        barycenter_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.epsilon = float(epsilon)
        self.rho = float(rho)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.barycenter_iterations = int(barycenter_iterations)
        self.slot_offsets = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.03)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)

    def _transport(
        self,
        slots: torch.Tensor,
        slot_weights: torch.Tensor,
        atoms: torch.Tensor,
        atom_weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        differences = slots[:, :, None, :] - atoms[:, None, :, :]
        cost = differences.pow(2).mean(dim=-1)
        exponent = (-cost / max(self.epsilon, 1e-4)).clamp(min=-30.0, max=0.0)
        kernel = torch.exp(exponent)
        kernel = kernel * mask[:, None, :].to(kernel.dtype)
        a = slot_weights.clamp_min(1e-8)
        b = atom_weights.clamp_min(1e-8) * mask.to(atom_weights.dtype)
        b = b / b.sum(-1, keepdim=True).clamp_min(1e-8)
        tau = self.rho / (self.rho + self.epsilon)
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        for _ in range(self.sinkhorn_iterations):
            kv = torch.einsum("bkn,bn->bk", kernel, v).clamp_min(1e-8)
            ratio_u = (a / kv).clamp(min=1e-8, max=1e8)
            u = ratio_u.pow(tau).clamp(max=1e4)
            ktu = torch.einsum("bkn,bk->bn", kernel, u).clamp_min(1e-8)
            ratio_v = (b / ktu).clamp(min=1e-8, max=1e8)
            v = ratio_v.pow(tau).clamp(max=1e4)
            v = v * mask.to(v.dtype)
        plan = u[:, :, None] * kernel * v[:, None, :]
        plan = torch.nan_to_num(plan, nan=0.0, posinf=1e4, neginf=0.0)
        plan = plan * mask[:, None, :].to(plan.dtype)
        return plan, cost

    def forward(
        self,
        processed: Dict[str, ProcessedModality],
        modality_names: Iterable[str],
        modality_weights: torch.Tensor,
    ) -> BarycenterOutput:
        names = list(modality_names)
        summaries = torch.stack([processed[name].summary for name in names], dim=1)
        global_summary = torch.einsum("bm,bmd->bd", modality_weights, summaries)
        slots = self.context_projection(global_summary)[:, None, :] + self.slot_offsets[None]
        batch = slots.shape[0]
        slot_weights = torch.full(
            (batch, self.num_slots),
            1.0 / self.num_slots,
            device=slots.device,
            dtype=slots.dtype,
        )

        modality_costs = []
        transported_mass = []
        for _ in range(self.barycenter_iterations):
            numerator = torch.zeros_like(slots)
            denominator = torch.zeros_like(slot_weights)
            current_costs = []
            current_mass = []
            for m_idx, name in enumerate(names):
                item = processed[name]
                plan, cost = self._transport(
                    slots, slot_weights, item.atoms, item.weights, item.mask
                )
                lambda_m = modality_weights[:, m_idx]
                weighted_plan = plan * lambda_m[:, None, None]
                numerator = numerator + torch.einsum(
                    "bkn,bnd->bkd", weighted_plan, item.atoms
                )
                denominator = denominator + weighted_plan.sum(-1)
                mass = plan.sum(dim=(1, 2))
                normalized_cost = (plan * cost).sum(dim=(1, 2)) / mass.clamp_min(1e-8)
                current_costs.append(normalized_cost)
                current_mass.append(mass)
            updated = numerator / denominator[:, :, None].clamp_min(1e-8)
            active = denominator > 1e-8
            slots = torch.where(active[:, :, None], updated, slots)
            proposed_weights = denominator / denominator.sum(-1, keepdim=True).clamp_min(1e-8)
            has_mass = denominator.sum(-1, keepdim=True) > 1e-8
            slot_weights = torch.where(has_mass, proposed_weights, slot_weights)
            modality_costs = current_costs
            transported_mass = current_mass

        modality_cost_tensor = torch.stack(modality_costs, dim=1)
        mass_tensor = torch.stack(transported_mass, dim=1)
        summary = torch.einsum("bk,bkd->bd", slot_weights, slots)

        normalized = torch.nn.functional.normalize(summaries, dim=-1)
        similarity = torch.einsum("bmd,bnd->bmn", normalized, normalized)
        pair_mask = torch.ones_like(similarity, dtype=torch.bool).triu(diagonal=1)
        disagreement_values = (1.0 - similarity).masked_select(pair_mask)
        if disagreement_values.numel() == 0:
            disagreement = torch.zeros(batch, device=slots.device, dtype=slots.dtype)
        else:
            disagreement = (1.0 - similarity).triu(diagonal=1).sum((1, 2))
            pairs = len(names) * (len(names) - 1) / 2
            disagreement = disagreement / max(pairs, 1.0)
        contradiction = torch.stack(
            [processed[name].contradiction_mean for name in names], dim=1
        )
        disagreement = disagreement + torch.einsum(
            "bm,bm->b", modality_weights, contradiction
        )
        return BarycenterOutput(
            slots=slots,
            weights=slot_weights,
            summary=summary,
            modality_costs=modality_cost_tensor,
            transported_mass=mass_tensor,
            disagreement=disagreement,
        )
