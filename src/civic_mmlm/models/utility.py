from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class UtilityOutput:
    outcomes: torch.Tensor
    utility: torch.Tensor
    uncertainty: torch.Tensor
    propensity_logits: torch.Tensor


class CausalUtilityEstimator(nn.Module):
    """Action-specific consequence and propensity heads (Equations 10-11)."""

    def __init__(
        self,
        hidden_dim: int,
        num_actions: int,
        utility_weights: tuple[float, float, float, float] = (1.0, -0.25, -1.0, -0.1),
    ) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions * 4),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_actions),
            nn.Softplus(),
        )
        self.propensity_head = nn.Linear(hidden_dim, num_actions)
        self.register_buffer(
            "utility_weights", torch.tensor(utility_weights, dtype=torch.float32)
        )

    def forward(self, state: torch.Tensor) -> UtilityOutput:
        outcomes = self.outcome_head(state).view(-1, self.num_actions, 4)
        uncertainty = self.uncertainty_head(state)
        utility = torch.einsum("bao,o->ba", outcomes, self.utility_weights)
        return UtilityOutput(
            outcomes=outcomes,
            utility=utility,
            uncertainty=uncertainty,
            propensity_logits=self.propensity_head(state),
        )


def doubly_robust_pseudo_outcomes(
    predicted_outcomes: torch.Tensor,
    factual_action: torch.Tensor,
    factual_outcome: torch.Tensor,
    propensity: torch.Tensor,
    clip: float = 0.05,
) -> torch.Tensor:
    """Cross-fitting is external; this function implements the DR correction itself."""

    batch, num_actions, _ = predicted_outcomes.shape
    action_one_hot = torch.nn.functional.one_hot(
        factual_action, num_classes=num_actions
    ).to(predicted_outcomes.dtype)
    factual_prediction = predicted_outcomes[
        torch.arange(batch, device=predicted_outcomes.device), factual_action
    ]
    residual = factual_outcome - factual_prediction
    clipped = propensity.clamp(min=clip, max=1.0)
    correction = action_one_hot[:, :, None] * residual[:, None, :] / clipped[:, :, None]
    return predicted_outcomes + correction
