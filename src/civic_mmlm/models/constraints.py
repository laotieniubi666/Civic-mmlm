from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DecisionOutput:
    raw_logits: torch.Tensor
    adjusted_logits: torch.Tensor
    probabilities: torch.Tensor
    entropy: torch.Tensor
    constraint_slack: torch.Tensor
    risk: torch.Tensor


class ConstrainedSelectiveDecisionField(nn.Module):
    """Risk-sensitive constrained action decoder (Equations 12-17)."""

    def __init__(
        self,
        hidden_dim: int,
        num_actions: int,
        utility_scale: float = 0.7,
        risk_scale: float = 0.35,
        entropy_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.utility_scale = float(utility_scale)
        self.risk_scale = float(risk_scale)
        self.entropy_temperature = float(entropy_temperature)
        self.action_embeddings = nn.Parameter(torch.randn(num_actions, hidden_dim) * 0.03)
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_state(
        self, state: torch.Tensor, action_context: torch.Tensor | None = None
    ) -> torch.Tensor:
        projected = self.state_projection(state)
        batch = state.shape[0]
        state_expanded = projected[:, None, :].expand(-1, self.num_actions, -1)
        action_embed = self.action_embeddings[None].expand(batch, -1, -1)
        if action_context is None:
            action_context = torch.zeros_like(state_expanded)
        features = torch.cat([state_expanded, action_embed, action_context], dim=-1)
        return self.action_scorer(features).squeeze(-1)

    def apply_constraints(
        self,
        raw_logits: torch.Tensor,
        utility: torch.Tensor,
        uncertainty: torch.Tensor,
        outcomes: torch.Tensor,
        legality: torch.Tensor,
        budget: torch.Tensor,
        action_costs: torch.Tensor,
    ) -> DecisionOutput:
        risk = outcomes[..., 2].relu() + 0.25 * outcomes[..., 1].relu() + uncertainty
        adjusted = raw_logits + self.utility_scale * utility - self.risk_scale * risk
        legal = legality > 0.5
        if action_costs.ndim == 1:
            action_costs = action_costs[None].expand(raw_logits.shape[0], -1)
        capacity_ok = action_costs <= budget[:, None]
        feasible = legal & capacity_ok
        no_feasible = ~feasible.any(dim=1)
        if no_feasible.any():
            feasible = feasible.clone()
            least_cost = action_costs.argmin(dim=1)
            feasible[no_feasible, least_cost[no_feasible]] = True
        adjusted = adjusted.masked_fill(~feasible, -1e4)
        probabilities = torch.softmax(adjusted / self.entropy_temperature, dim=-1)
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-8))
        ).sum(-1)
        illegal_slack = (probabilities * (~legal).to(probabilities.dtype)).sum(-1)
        capacity_slack = (
            probabilities * (action_costs - budget[:, None]).relu()
        ).sum(-1)
        return DecisionOutput(
            raw_logits=raw_logits,
            adjusted_logits=adjusted,
            probabilities=probabilities,
            entropy=entropy,
            constraint_slack=illegal_slack + capacity_slack,
            risk=risk,
        )

    def forward(
        self,
        state: torch.Tensor,
        action_context: torch.Tensor,
        utility: torch.Tensor,
        uncertainty: torch.Tensor,
        outcomes: torch.Tensor,
        legality: torch.Tensor,
        budget: torch.Tensor,
        action_costs: torch.Tensor,
    ) -> DecisionOutput:
        raw_logits = self.score_state(state, action_context)
        return self.apply_constraints(
            raw_logits,
            utility,
            uncertainty,
            outcomes,
            legality,
            budget,
            action_costs,
        )


def conditional_fairness_penalty(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    need_bins: torch.Tensor,
) -> torch.Tensor:
    """Differentiable equal-opportunity proxy conditioned on admissible need."""

    true_prob = probabilities.gather(1, labels[:, None]).squeeze(1)
    penalties = []
    for need in need_bins.unique():
        need_mask = need_bins == need
        group_means = []
        for group in groups[need_mask].unique():
            mask = need_mask & (groups == group)
            if mask.sum() >= 2:
                group_means.append(true_prob[mask].mean())
        if len(group_means) >= 2:
            stacked = torch.stack(group_means)
            penalties.append(stacked.max() - stacked.min())
    if not penalties:
        return probabilities.new_zeros(())
    return torch.stack(penalties).mean()


class DualVariables(nn.Module):
    """Non-trainable projected-ascent multipliers for aggregate constraints."""

    def __init__(self, names: tuple[str, ...] = ("constraint", "fairness")) -> None:
        super().__init__()
        for name in names:
            self.register_buffer(name, torch.tensor(0.0))
        self.names = names

    @torch.no_grad()
    def projected_ascent(self, violations: dict[str, torch.Tensor], lr: float) -> None:
        for name, violation in violations.items():
            value = getattr(self, name)
            value.add_(lr * violation.detach()).clamp_(min=0.0, max=100.0)

    def penalty(self, violations: dict[str, torch.Tensor]) -> torch.Tensor:
        terms = [getattr(self, name) * violations[name] for name in violations]
        return torch.stack(terms).sum() if terms else torch.tensor(0.0)
