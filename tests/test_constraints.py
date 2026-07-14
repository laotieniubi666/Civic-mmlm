from __future__ import annotations

import torch

from civic_mmlm.models.constraints import ConstrainedSelectiveDecisionField


def test_infeasible_actions_receive_zero_probability() -> None:
    decoder = ConstrainedSelectiveDecisionField(hidden_dim=8, num_actions=3)
    state = torch.randn(2, 8)
    action_context = torch.randn(2, 3, 8)
    utility = torch.zeros(2, 3)
    uncertainty = torch.zeros(2, 3)
    outcomes = torch.zeros(2, 3, 4)
    legality = torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.float32)
    budget = torch.tensor([0.5, 0.25])
    costs = torch.tensor([[0.2, 0.4, 0.8], [0.2, 0.4, 0.8]])
    output = decoder(
        state, action_context, utility, uncertainty, outcomes, legality, budget, costs
    )
    assert output.probabilities[0, 1].item() < 1e-6
    assert output.probabilities[0, 2].item() < 1e-6
    assert output.probabilities[1, 1].item() < 1e-6
    assert output.probabilities[1, 2].item() < 1e-6
