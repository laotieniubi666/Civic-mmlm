# Mapping from the manuscript to the public code

This document records which implementation choice corresponds to each mathematical object. It also makes approximation boundaries explicit.

| Equation | Manuscript object | Implementation | Notes |
|---|---|---|---|
| (1)-(2) | multimodal case, context, actions, consequences | nested modality dictionaries plus legality, budget, action costs, and four-outcome head | Context variables are supplied by dataset adapters; the synthetic proxy uses group and need bins. |
| (3) | empirical evidence measure | `ProvenanceAwareEvidenceEncoder` | Atom weights are normalized over valid atoms. |
| (4) | reliability score | semantic linear score + log provenance + log reliability - contradiction | The coefficients are configurable. |
| (5)-(6) | unbalanced entropic barycenter and contradiction control | `UnbalancedEvidenceBarycenter` | Uses learned compact support slots and KL-relaxed generalized Sinkhorn scaling. This is a tractable approximation, not a claim of byte-identical author code. |
| (7) | action-conditioned modality weights | `ModalityGating` | Produces both global weights for the barycenter and per-action weights for decoding. |
| (8) | culturally valid intervention invariance | `cultural_invariance_loss` | Jensen-Shannon divergence between original and valid-intervention posteriors. |
| (9) | material sensitivity | `material_sensitivity_loss` | L1 posterior-distance margin. |
| (10) | doubly robust pseudo-outcome | `doubly_robust_pseudo_outcomes` | Cross-fitting must be done by an external fold pipeline on real data. The demo trains factual outcome and propensity heads. |
| (11) | scalar public utility | `CausalUtilityEstimator` | Configurable weighted combination of benefit, delay, harm, and cost. |
| (12) | utility-risk objective | `ConstrainedSelectiveDecisionField` | Utility is added and risk proxy is subtracted from action logits. |
| (13) | legality/procedure constraints | feasibility mask and slack | General differentiable legal rules can be inserted before `apply_constraints`. |
| (14) | fairness conditioned on need | `conditional_fairness_penalty` | Uses a differentiable true-class probability gap within need bins. |
| (15) | capacity constraint | action costs vs. sample budget | Infeasible actions are masked; expected violation is reported as slack. |
| (16) | projected dual ascent | `DualVariables.projected_ascent` | Two aggregate multipliers are included in the public trainer. |
| (17) | nonconformity score | `ConformalAbstentionCalibrator.score` | Entropy, barycenter disagreement, identification weakness, and constraint slack. |
| (18)-(21) | minimal certificate | `MinimalEvidenceCertificate` | Sigmoid gates are trained with sufficiency, necessity, and sparsity losses. Greedy export returns atom-level provenance metadata. A true entailment model is task-dependent and must be supplied by a real-data adapter. |
| (22)-(23) | unified objective | `UnifiedObjective` | All included loss terms are optimized jointly. |
| (24) | deletion stability | stress and certificate deletion interfaces | The package measures empirical drift; it does not present a new proof of the proposition. |
| (25) | selective-risk control | split-conformal calibration | Threshold uses the finite-sample corrected empirical quantile. |
| (26) | intervention generalization | valid-intervention stress evaluation | The code evaluates the behavior; the theoretical bound still depends on the assumptions stated in the paper. |

## Approximation choices

The manuscript calls the central object a distributional barycenter. The public implementation represents that distribution with a small learned support of `K` slots. This keeps the algorithm differentiable and executable on CPU while preserving the key behavior: evidence mass can be relaxed, missing modalities do not require dummy tokens, and transported mass/cost can be inspected.

The manuscript also describes hard-concrete certificate gates. The public implementation uses temperature-controlled sigmoid gates to avoid adding a specialized stochastic dependency. Replacing them with hard-concrete gates is localized to `models/certificates.py`.
