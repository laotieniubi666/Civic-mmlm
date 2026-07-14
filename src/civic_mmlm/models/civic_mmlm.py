from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from civic_mmlm.data.types import Modalities

from .barycenter import UnbalancedEvidenceBarycenter
from .certificates import MinimalEvidenceCertificate
from .constraints import ConstrainedSelectiveDecisionField
from .evidence import ModalityGating, ProvenanceAwareEvidenceEncoder
from .utility import CausalUtilityEstimator


class CIVICMMLM(nn.Module):
    """Executable public reconstruction of the paper's coupled decision system."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.config = config
        self.modality_names = list(model_cfg["modalities"])
        self.raw_dim = int(model_cfg["raw_dim"])
        self.hidden_dim = int(model_cfg["hidden_dim"])
        self.num_actions = int(model_cfg["num_actions"])

        evidence_cfg = model_cfg.get("evidence", {})
        self.evidence_encoder = ProvenanceAwareEvidenceEncoder(
            self.modality_names,
            self.raw_dim,
            self.hidden_dim,
            temperature=evidence_cfg.get("temperature", 0.7),
            alpha_provenance=evidence_cfg.get("alpha_provenance", 0.65),
            alpha_reliability=evidence_cfg.get("alpha_reliability", 0.65),
            alpha_contradiction=evidence_cfg.get("alpha_contradiction", 1.0),
        )
        self.modality_gating = ModalityGating(
            self.hidden_dim, self.num_actions, self.modality_names
        )
        bary_cfg = model_cfg.get("barycenter", {})
        self.barycenter = UnbalancedEvidenceBarycenter(
            self.hidden_dim,
            num_slots=bary_cfg.get("num_slots", 4),
            epsilon=bary_cfg.get("epsilon", 0.35),
            rho=bary_cfg.get("rho", 1.0),
            sinkhorn_iterations=bary_cfg.get("sinkhorn_iterations", 5),
            barycenter_iterations=bary_cfg.get("barycenter_iterations", 3),
        )
        utility_cfg = model_cfg.get("utility", {})
        self.utility_estimator = CausalUtilityEstimator(
            self.hidden_dim,
            self.num_actions,
            utility_weights=tuple(
                utility_cfg.get("weights", [1.0, -0.25, -1.0, -0.1])
            ),
        )
        decision_cfg = model_cfg.get("decision", {})
        self.decision_field = ConstrainedSelectiveDecisionField(
            self.hidden_dim,
            self.num_actions,
            utility_scale=decision_cfg.get("utility_scale", 0.7),
            risk_scale=decision_cfg.get("risk_scale", 0.35),
            entropy_temperature=decision_cfg.get("temperature", 1.0),
        )
        certificate_cfg = model_cfg.get("certificate", {})
        self.certificate = MinimalEvidenceCertificate(
            self.hidden_dim,
            self.num_actions,
            gate_temperature=certificate_cfg.get("gate_temperature", 0.7),
        )
        self.state_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        modalities: Modalities,
        legality: torch.Tensor,
        budget: torch.Tensor,
        action_costs: torch.Tensor,
        certificate_action: torch.Tensor | None = None,
        compute_certificate: bool = True,
    ) -> Dict[str, object]:
        processed = self.evidence_encoder(modalities)
        global_weights, action_weights, action_context = self.modality_gating(processed)
        barycenter = self.barycenter(
            processed, self.modality_names, global_weights
        )
        state = self.state_norm(barycenter.summary)
        utility = self.utility_estimator(state)
        decision = self.decision_field(
            state,
            action_context,
            utility.utility,
            utility.uncertainty,
            utility.outcomes,
            legality,
            budget,
            action_costs,
        )
        output: Dict[str, object] = {
            "processed": processed,
            "global_modality_weights": global_weights,
            "action_modality_weights": action_weights,
            "action_context": action_context,
            "barycenter": barycenter,
            "state": state,
            "utility": utility,
            "decision": decision,
        }
        if compute_certificate:
            if certificate_action is None:
                certificate_action = decision.probabilities.argmax(dim=-1)
            cert = self.certificate(
                processed, self.modality_names, certificate_action
            )
            cert_raw = self.decision_field.score_state(cert.certificate_state)
            comp_raw = self.decision_field.score_state(cert.complement_state)
            cert_decision = self.decision_field.apply_constraints(
                cert_raw,
                utility.utility,
                utility.uncertainty,
                utility.outcomes,
                legality,
                budget,
                action_costs,
            )
            comp_decision = self.decision_field.apply_constraints(
                comp_raw,
                utility.utility,
                utility.uncertainty,
                utility.outcomes,
                legality,
                budget,
                action_costs,
            )
            output.update(
                {
                    "certificate": cert,
                    "certificate_decision": cert_decision,
                    "complement_decision": comp_decision,
                }
            )
        return output
