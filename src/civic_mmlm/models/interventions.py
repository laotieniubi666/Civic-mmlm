from __future__ import annotations

import torch


def jensen_shannon_divergence(
    p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    midpoint = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p) - torch.log(midpoint))).sum(-1)
    kl_qm = (q * (torch.log(q) - torch.log(midpoint))).sum(-1)
    return 0.5 * (kl_pm + kl_qm)


def cultural_invariance_loss(
    original_probabilities: torch.Tensor,
    intervened_probabilities: torch.Tensor,
) -> torch.Tensor:
    return jensen_shannon_divergence(
        original_probabilities, intervened_probabilities
    ).mean()


def material_sensitivity_loss(
    original_probabilities: torch.Tensor,
    material_probabilities: torch.Tensor,
    margin: float = 0.35,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(
        original_probabilities - material_probabilities, ord=1, dim=-1
    )
    return torch.relu(margin - distance).mean()
