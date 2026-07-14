# Model card: CIVIC-MMLM public reference implementation

## Model details

- Type: multimodal evidence-fusion and constrained selective decision prototype.
- Release: independent public reconstruction, version 0.1.0.
- Training data included: synthetic only.
- License: MIT for code; external data and model licenses remain separate.

## Intended use

- research on provenance-aware multimodal fusion;
- software verification of unbalanced evidence transport;
- study of intervention invariance and material sensitivity;
- calibration and selective-prediction experiments;
- evidence-certificate auditing methods.

## Out-of-scope use

Do not use this model to autonomously determine or recommend high-impact outcomes involving identifiable people, including legal status, welfare eligibility, policing, migration, healthcare access, education access, employment, credit, housing, or essential services.

## Limitations

- Synthetic data do not represent real cultural, linguistic, legal, or administrative complexity.
- The barycenter and certificate modules are faithful computational approximations, not the authors' unpublished implementation.
- Fairness metrics depend on legally and normatively justified group and need definitions.
- Predicted utility is not causal merely because a doubly robust estimator is available.
- Conformal guarantees require exchangeability within the declared operating domain.
- A certificate can reveal model dependence but cannot establish factual, legal, or moral correctness.

## Required safeguards for any field study

- documented human authority and the ability to override;
- accessible appeal and correction channels;
- jurisdiction-specific legal and impact review;
- community and domain-expert participation;
- data minimization and access control;
- subgroup performance and calibration audits;
- incident logging and rollback plan;
- explicit monitoring of abstention load on human reviewers.
