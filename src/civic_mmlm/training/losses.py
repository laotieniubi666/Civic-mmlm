from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from civic_mmlm.models.constraints import conditional_fairness_penalty
from civic_mmlm.models.interventions import (
    cultural_invariance_loss,
    jensen_shannon_divergence,
    material_sensitivity_loss,
)


@dataclass
class LossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]
    violations: dict[str, torch.Tensor]


class UnifiedObjective:
    """Practical realization of Equations (22)-(23)."""

    def __init__(self, config: dict) -> None:
        loss_cfg = config["training"].get("loss_weights", {})
        self.weights = {
            "task": loss_cfg.get("task", 1.0),
            "barycenter": loss_cfg.get("barycenter", 0.04),
            "cultural": loss_cfg.get("cultural", 0.25),
            "sensitivity": loss_cfg.get("sensitivity", 0.15),
            "dr": loss_cfg.get("dr", 0.25),
            "selective": loss_cfg.get("selective", 0.10),
            "certificate": loss_cfg.get("certificate", 0.18),
            "fairness": loss_cfg.get("fairness", 0.08),
            "constraint": loss_cfg.get("constraint", 0.25),
        }
        self.certificate_sparsity = loss_cfg.get("certificate_sparsity", 0.02)
        self.certificate_necessity_margin = loss_cfg.get(
            "certificate_necessity_margin", 0.15
        )
        self.material_margin = loss_cfg.get("material_margin", 0.35)

    def __call__(
        self,
        output: dict,
        valid_output: dict,
        material_output: dict,
        batch: dict,
        dual_penalty: torch.Tensor | None = None,
    ) -> LossOutput:
        decision = output["decision"]
        labels = batch["label"]
        task = F.cross_entropy(decision.adjusted_logits, labels)

        barycenter = output["barycenter"]
        bary = barycenter.modality_costs.mean() + 0.2 * barycenter.disagreement.mean()
        cultural = cultural_invariance_loss(
            decision.probabilities, valid_output["decision"].probabilities
        )
        sensitivity = material_sensitivity_loss(
            decision.probabilities,
            material_output["decision"].probabilities,
            margin=self.material_margin,
        )

        utility = output["utility"]
        factual_pred = utility.outcomes[
            torch.arange(labels.shape[0], device=labels.device), batch["factual_action"]
        ]
        outcome_loss = F.mse_loss(factual_pred, batch["factual_outcome"])
        propensity_loss = F.cross_entropy(
            utility.propensity_logits, batch["factual_action"]
        )
        dr = outcome_loss + 0.25 * propensity_loss

        one_hot = F.one_hot(labels, num_classes=decision.probabilities.shape[1]).to(
            decision.probabilities.dtype
        )
        selective = ((decision.probabilities - one_hot) ** 2).sum(-1).mean()

        certificate = output["certificate"]
        cert_probs = output["certificate_decision"].probabilities
        comp_probs = output["complement_decision"].probabilities
        sufficiency = jensen_shannon_divergence(
            decision.probabilities, cert_probs
        ).mean()
        predicted_support = cert_probs.gather(1, labels[:, None]).squeeze(1)
        complement_support = comp_probs.gather(1, labels[:, None]).squeeze(1)
        necessity = torch.relu(
            self.certificate_necessity_margin
            - (predicted_support - complement_support)
        ).mean()
        mask_count = certificate.atom_mask.to(certificate.selected_gates.dtype).sum(-1)
        sparsity = (
            certificate.selected_gates.sum(-1) / mask_count.clamp_min(1.0)
        ).mean()
        certificate_loss = (
            sufficiency + necessity + self.certificate_sparsity * sparsity
        )

        fairness = conditional_fairness_penalty(
            decision.probabilities,
            labels,
            batch["group"],
            batch["need_bin"],
        )
        constraint = decision.constraint_slack.mean()
        violations = {"constraint": constraint, "fairness": fairness}

        components = {
            "task": task,
            "barycenter": bary,
            "cultural": cultural,
            "sensitivity": sensitivity,
            "dr": dr,
            "selective": selective,
            "certificate": certificate_loss,
            "fairness": fairness,
            "constraint": constraint,
        }
        total = sum(self.weights[name] * value for name, value in components.items())
        if dual_penalty is not None:
            total = total + dual_penalty
        return LossOutput(total=total, components=components, violations=violations)
