# When Not to Learn: Site-Level Abstention and Update Selection in Federated Learning under Heterogeneity and Drift

**Abhinav Annareddy** — technical report, v0.4 (draft; not peer reviewed)

> **Reader's note.** This is a working technical report that accompanies the code in this repository. All experiments use small linear models; the synthetic benchmark and the partitions of public datasets are designed to isolate one question, not to represent production federations. The literature review is a good-faith orientation from memory and **should be verified against the primary sources before any submission**. No novelty claim is made.

## Abstract

Federated learning (FL) assumes that sites benefit from contributing to, and learning from, a shared model. When client distributions differ, or change during training, that assumption can fail for some or all sites. We study the decision of **whether a site (or an update) should participate**, rather than how to aggregate more cleverly. In a controlled simulator with mid-training drift, and on label- and covariate-skewed partitions of two public datasets, we compare local training, FedAvg, FedProx, Ditto, fine-tuned FedAvg, median and trimmed-mean aggregation, a validation-guided update-selection rule, and a per-site abstention gate. Across 20 seeds with paired bootstrap intervals we find that (i) under moderate-to-strong heterogeneity, local or personalised training beats every global-model method, by up to 14 pp (synthetic) and 20 pp (real, α = 0.1 digits); (ii) collaboration helps only when sites are near-identical (+2–3 pp), a pattern a per-site validation gate recovers by abstaining ~96 % of the time under strong skew and federating ~54 % of the time when skew is mild; (iii) a global mean/worst-site selection guardrail reduces average accuracy relative to FedAvg in most settings but protects the worst-served site increasingly as drift severity (up to +12 pp) and federation size (up to +14 pp) grow. We document limitations, including oracle validation, linear models, and untuned baselines, and release all code, seeds and outputs.

## 1. Introduction

FedAvg [1] and its descendants optimise a single global model from many sites' updates. With non-IID data, such a model can serve the average site poorly and individual sites worse [2, 3, 4]. Two families of remedies exist: *personalisation* [5, 6] and *robust aggregation* [7]. A third, less studied lever is **selection**: deciding, per round and per site, whether collaboration is worthwhile at all. This report asks:

* **RQ1** — How do local training, shared-model FL, personalised FL and selection compare as heterogeneity and local data size vary?
* **RQ2** — How does each respond to drift that arises *during* training?
* **RQ3** — Does a simple global guardrail on mean and worst-site validation loss help? Does a per-site abstention rule?
* **RQ4** — Do the synthetic conclusions survive on real data partitions?

## 2. Related work (orientation)

*Non-IID FL.* Heterogeneity is a central challenge in FL [2]; empirical studies catalogue partitioning schemes, including the Dirichlet label-skew we use [3, 4]. *Proximal and personalised FL.* FedProx adds a proximal term to limit local drift [5]; Ditto jointly learns a global model and per-site models regularised toward it [6]; surveys cover broader personalisation [8]. Clustered approaches group sites with similar objectives [9, 10]. *Fairness and worst-case objectives.* Agnostic FL optimises for the worst mixture of client distributions [11]; q-FFL reweights toward higher-loss clients [12]. *Robust aggregation.* Coordinate-wise median and trimmed mean come from Byzantine-robust learning [7]. *Concept drift.* Adaptation to drift is a long-standing topic in streaming learning [13]; FL-specific treatments of distributed concept drift exist [14]. *Privacy.* Secure aggregation [15] protects individual updates but not necessarily metrics; this matters for our oracle validation (Section 7).

**Positioning.** Our selection rule (accept an update if it does not worsen mean or worst-site validation loss beyond a tolerance) is deliberately simple and is **not** claimed to be novel. The contribution is empirical: a controlled comparison that includes the abstention option, strong personalised baselines, and paired statistics, with negative results reported.

## 3. Setup

Full definitions are in [`METHODOLOGY.md`](METHODOLOGY.md).

**Synthetic benchmark.** Six sites share a base logistic decision boundary perturbed by a heterogeneity scale *h*; a third of sites drift at round 15 of 30 (covariate + concept shift, severity *s*). Three schedules are supported: abrupt, gradual (10-round ramp), recurring (5-round period). **Real-data partitions.** *Digits* (binary: digit ≥ 5; Dirichlet(α) label skew over 6 sites) and *breast cancer* (sites are blocks of a noisy first-principal-component ranking), each standardised with local training statistics. **Methods.** Local only; FedAvg; FedAvg + 10 local steps; FedProx (μ = 0.1); Ditto (μ = 0.1); coordinate median; trimmed mean (20 %); selective federation; selective federation + per-site gate (a site uses the federated model only if it has lower validation loss than its own local model). **Metrics.** Final mean and worst-site test accuracy (last five rounds), drop at drift. **Statistics.** 20 paired seeds, 95 % percentile bootstrap CIs on mean paired differences.

## 4. Results

Complete tables, figures and CSVs: [`RESULTS.md`](RESULTS.md).

**RQ1 / RQ3 — local vs. shared, and the gate.** In the default drift scenario local training reaches 92.9 % mean / 87.6 % worst-site accuracy against 84.3 % / 67.5 % for FedAvg. The gain map (heterogeneity × data size) shows federation winning only for near-identical sites (heterogeneity 0.1; about +1 to +3 pp); losses reach −14 pp. The site gate matches local-only in the default scenario (federated in 3.5 % of decisions) and stays between −1.5 and +2.5 pp of local-only in every grid cell.

**RQ2 — drift.** Shared models drop 3–4 pp mean (up to ~19 pp worst-site) at an abrupt drift; Ditto, fine-tuned FedAvg and local models drop ≤ 0.4 pp. Gradual and recurring schedules give the same ranking.

**RQ3 — the selection guardrail.** Versus FedAvg the rule changes mean accuracy by −1.8 pp (CI −2.6, −0.9) and worst-site accuracy by +2.9 pp (CI −0.4, +7.0) at default severity. The worst-site benefit grows with severity (−1.4, −0.7, +6.3, +12.0, +11.7 pp at severity 0–4) and with federation size (+14 pp at 12 sites, +9 pp at 24). Its tolerance hyper-parameter matters little until it accepts everything; using as few as 5 validation samples per site does not degrade it.

**RQ4 — real data.** On digits with α = 5, collaboration helps: FedAvg +2.8 pp (CI +2.1, +3.6), Ditto +3.0 pp over local, and the gate federates 53.5 % of the time. At α = 0.1 FedAvg is 20 pp worse than local (CI −23.3, −17.2) while Ditto is within 2 pp and the gate abstains 96 % of the time. The breast-cancer partition behaves like strong skew. Selective federation is below FedAvg in all four real scenarios.

## 5. Discussion

*What is supported.* (1) Heterogeneity decides whether to collaborate, and the decision is cheaply learnable per site. (2) Strong personalised baselines are mandatory in evaluations. (3) A global guardrail has a defensible niche (worst-case protection under severe drift or many sites) but is not an accuracy method.

*What is not.* The experiments do not show that selection improves mean accuracy, that it is private, or that results carry over to deep models, other modalities, or realistic cross-silo deployments. FedProx and trimmed mean were not tuned; their lack of benefit should not be read as evidence about those methods in general.

## 6. Threats to validity

* **Oracle validation.** Selection evaluates every update on every site's validation set — a trusted central evaluator that real FL generally lacks. A validation-size ablation suggests small samples suffice but does not address trust or leakage.
* **Linear models** and ≥ 60 samples per site favour local training.
* **Untuned baselines** (μ, trim fraction, tolerance, steps) at fixed defaults for all methods.
* **Simulated drift** is a single parametric family; **partitions** of public datasets are synthetic constructs.
* **Multiple comparisons.** Intervals are reported per comparison without correction; p-values are exploratory.
* **No privacy analysis.** Nothing here is a privacy guarantee.

## 7. Future work

Trusted-execution or secure-aggregation realisations of the selection signal; deep models and a real cross-silo benchmark (e.g. multi-site medical data with documented licences); tuned FedProx / SCAFFOLD / clustered FL; detection delay and recovery under harsher recurring drift; a learned (instead of validation-threshold) abstention rule with calibrated uncertainty; leakage analysis of the metrics exchanged.

## Reproducibility

Code, seeds, configurations and outputs are in this repository. Commands: see [`RESULTS.md`](RESULTS.md). Seeds 1–20 (main, severity, real data) and 1–10 (grids, ablations) are fixed.

## References

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS, 2017.
[2] P. Kairouz et al. *Advances and Open Problems in Federated Learning.* Foundations and Trends in Machine Learning, 2021.
[3] T.-M. H. Hsu, H. Qi, M. Brown. *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification.* arXiv:1909.06335, 2019.
[4] Q. Li, Y. Diao, Q. Chen, B. He. *Federated Learning on Non-IID Data Silos: An Experimental Study.* ICDE, 2022.
[5] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith. *Federated Optimization in Heterogeneous Networks.* MLSys, 2020.
[6] T. Li, S. Hu, A. Beirami, V. Smith. *Ditto: Fair and Robust Federated Learning Through Personalization.* ICML, 2021.
[7] D. Yin, Y. Chen, K. Ramchandran, P. Bartlett. *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.* ICML, 2018.
[8] A. Z. Tan, H. Yu, L. Cui, Q. Yang. *Towards Personalized Federated Learning.* IEEE TNNLS, 2022.
[9] A. Ghosh, J. Chung, D. Yin, K. Ramchandran. *An Efficient Framework for Clustered Federated Learning.* NeurIPS, 2020.
[10] F. Sattler, K.-R. Müller, W. Samek. *Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization under Privacy Constraints.* IEEE TNNLS, 2021.
[11] M. Mohri, G. Sivek, A. T. Suresh. *Agnostic Federated Learning.* ICML, 2019.
[12] T. Li, M. Sanjabi, A. Beirami, V. Smith. *Fair Resource Allocation in Federated Learning.* ICLR, 2020.
[13] J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, A. Bouchachia. *A Survey on Concept Drift Adaptation.* ACM Computing Surveys, 2014.
[14] E. Jothimurugesan, K. Hsieh, J. Wang, G. Joshi, P. B. Gibbons. *Federated Learning under Distributed Concept Drift.* AISTATS, 2023.
[15] K. Bonawitz et al. *Practical Secure Aggregation for Privacy-Preserving Machine Learning.* ACM CCS, 2017.
