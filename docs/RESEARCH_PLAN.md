# Research plan: When Not to Learn

## Working title

**Selective Federated Learning Under Client Heterogeneity and Distribution Shift**

## Core question

Under what conditions can a federation decide that a client's update is useful, harmful, or too uncertain to accept—without making individual participants less reliable or exposing sensitive local information?

## Motivation

Federated learning coordinates model training across data holders without requiring them to pool raw training records. However, client data distributions and objectives can differ. A global update may improve average performance while degrading the worst-served client. Changes in equipment, populations, software, or behaviour can further invalidate assumptions learned earlier.

## Questions to investigate

1. **Heterogeneity:** How do client distribution differences affect average and worst-client outcomes under local training, FedAvg, and selective participation?
2. **Drift:** When a client's data distribution changes during training, how quickly can the federation detect harm and recover?
3. **Decision quality:** What is the trade-off between rejecting harmful updates and rejecting updates that would have been useful?
4. **Personalisation:** When is a local or personalised model preferable to a shared model?
5. **Privacy and practicality:** Can useful selection signals be obtained without central access to private validation examples or sensitive per-client metrics?

## Initial hypotheses (to test, not assume)

- H1: Under some forms of strong client heterogeneity, optimising only mean performance can hide substantial worst-client degradation.
- H2: Selective participation may help in some regimes but can underperform when validation estimates are noisy, shifts are abrupt, or useful clients are rejected.
- H3: Selection thresholds produce a measurable trade-off between protection from harmful updates, convergence speed, and communication overhead.
- H4: Any claimed benefit must be tested against personalised and robust-FL baselines, not only local training and FedAvg.

## Current experimental protocol

- **Data:** synthetic binary classification generated independently for multiple clients.
- **Model:** logistic regression trained by full-batch gradient descent.
- **Methods:** local-only, FedAvg, and a validation-guided selective baseline.
- **Evaluation:** mean accuracy, worst-client accuracy, mean log-loss, and personalised mean accuracy on held-out shifted test distributions.
- **Reproducibility:** explicit random seeds; preserve all seeds and configurations, not only best-performing runs.
- **Outputs:** JSON and CSV for further statistical analysis.

## Critical limitations of the current version

The original `experiment.py` shifted test set is a static proxy; the v0.3 `drift.py` benchmark adds a genuine mid-training drift event. The selection mechanism uses an oracle-style evaluator with access to every site's validation set. It is neither a secure protocol nor a formal privacy guarantee. The update screening rule is a baseline and should not be presented as a novel algorithm without a literature review.

## Next milestones

### M1 — Establish reproducibility ✅ (20 seeds, bootstrap CIs)

Run at least 10–30 seeds. Report mean, standard deviation, and per-seed outcomes. Check that conclusions do not depend on a single seed.

### M2 — Introduce temporal drift ✅ (v0.3; recovery metrics are weakly informative at default severity)

Specify a drift round, alter a subset of clients' data-generating process after that round, and evaluate performance before and after the change. Measure detection delay and recovery time.

### M3 — Strengthen comparisons 🟡 (fine-tuned FedAvg and coordinate median added; FedProx/Ditto outstanding)

Add a personalised federated-learning method and a robust aggregation method. Document implementation choices and tuning budgets fairly across methods.

### M4 — Ablation and failure analysis 🟡 (heterogeneity, data size, severity done; tolerance, client count, validation size outstanding)

Vary client count, heterogeneity severity, drift magnitude, selection tolerance, and validation sample size. Record when selection rejects helpful updates or accepts harmful ones.

### M5 — Reliability and privacy

Add calibration/uncertainty measures and explicitly designed privacy threat models. Do not describe the system as privacy-preserving until the protocol and evidence support that claim.

### M6 — Public benchmark

Choose a suitable public dataset and document preprocessing, partitioning, licences, and limitations. Keep synthetic and real-data results distinct.

## Reporting checklist

- [ ] State the exact question and hypothesis before running the comparison.
- [ ] Publish configurations, seeds, and all planned metrics.
- [ ] Include all baselines and ablations, including negative results.
- [ ] Separate validation, model-selection, and final test data.
- [ ] Report mean and dispersion across seeds; avoid cherry-picking.
- [ ] Record compute time and communication assumptions.
- [ ] State privacy assumptions and attack surface explicitly.
- [ ] Avoid claims of novelty until supported by a careful literature review.
