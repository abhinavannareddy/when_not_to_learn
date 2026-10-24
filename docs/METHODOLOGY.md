# Methodology

This document defines exactly what the drift benchmark (`drift.py`, `benchmark.py`) measures, so results can be critiqued and reproduced.

## Data-generating process

Each site `i` draws binary-classification data from a logistic-style model:

```
x ~ N(offset_i, I)          # 6 features
y = 1[ x · slope_i + 0.25·sum(offset_i) + ε > 0 ],   ε ~ N(0, 0.8²)
```

* `offset_i ~ N(0, (0.9·h)²)` and `slope_i = base_slope + N(0, (0.45·h)²)`, where **`h` is the heterogeneity scale** (default 1.0). `h → 0` gives near-identical sites.
* Every site has `n` samples per regime, split 60 / 20 / 20 into train / validation / test.
* **Drift.** A fraction `drift_fraction` (default ≈ ⅓ of sites) switches at `drift_round` (default 15 of 30): `offset' = offset + severity·N(0, I)` and `slope' = slope + severity·N(0, 0.5²·I)`. Drifted sites therefore suffer both covariate and concept shift. The remaining sites are stationary but are re-sampled.
* **Schedules.** *Abrupt*: from round `drift_round + 1` every site uses its second data draw. *Gradual*: a fraction `(r - drift_round)/ramp_rounds` of each site's data comes from the second draw (blended at the sample level). *Recurring*: sites alternate between the two draws every `period` rounds. Non-drifted sites have the same distribution in both draws, so blending is harmless for them.
* All methods train on, and are evaluated against, the regime currently in force.

## Strategies

| Key | Strategy | Notes |
|---|---|---|
| `local_only` | Each site keeps training its own model on its own current data. | No communication. |
| `fedavg` | Sample-weighted mean of all clients' local updates each round. | |
| `fedavg_ft` | FedAvg model, then 10 local gradient steps per site before evaluation. | Simple personalisation baseline. |
| `median` | Coordinate-wise median of local updates. | Robust-aggregation baseline. |
| `trimmed_mean` | Coordinate-wise mean after dropping the lowest and highest 20 % of updates. | Robust-aggregation baseline. |
| `fedprox` | FedAvg whose local objective adds `(mu/2)||w - w_global||^2`, `mu = 0.1`. | Not tuned. |
| `ditto` | Global model trained as FedAvg; each site also keeps a personal model trained with a proximal pull (`mu = 0.1`) toward the global model. Evaluated with the personal models. | Not tuned. |
| `selective` | Accept update *u* only if the **mean** and **worst-site** validation log-loss of *u* (evaluated on every site's validation set) do not exceed the current global model's by more than `tolerance` (0.015). Average the accepted updates; if none pass, use the least harmful one. | Oracle-style: the evaluator sees every site's validation data. |
| `site_gate` | Each site chooses, by its **own validation loss**, between its local model and the `selective` global model. | A per-site decision to abstain from collaboration. |

All strategies share the same optimiser (full-batch gradient descent, 30 local steps, lr 0.12), initialisation (zeros), data, and seed. No method was tuned individually; the tolerance and step counts are fixed defaults, which may disadvantage some methods.

## Real-data partitions

* **Digits** (1,797 samples, 64 features). Task: digit >= 5. For each digit class, a Dirichlet(alpha) draw over the 6 sites decides which share of that class each site receives; small alpha = strong label skew.
* **Breast cancer** (569 samples, 30 features). Task: malignant vs benign. Samples are ranked by the first principal component plus Gaussian noise (0.5 sd) and cut into 6 contiguous blocks, giving strong covariate and label-prior skew.
* Each site splits its samples 60/20/20 and standardises features **with its own training statistics** (this removes some covariate shift and so favours local models). There is no temporal drift in these experiments.

## Metrics

* **Final mean / worst-site accuracy**: mean (or minimum) over sites of test accuracy on the *current* regime, averaged over the **last five rounds** to reduce round-to-round noise.
* **Post-drift AUC**: mean mean-accuracy over all post-drift rounds.
* **Drop**: mean accuracy at round `drift_round` minus round `drift_round + 1`.
* **Recovery rounds**: rounds after the drift until mean accuracy first reaches 98 % of its final plateau. In the default scenario drift costs little after the first round, so this metric is near zero for most methods and is mostly informative at high severity.

## Statistics

* Seeds are *paired*: all methods see identical clients within a seed, so comparisons use per-seed differences.
* **Confidence intervals** are 95 % percentile bootstrap intervals for the mean (10 000 resamples).
* **p-values** come from a two-sided sign-flip permutation test on paired differences (10 000 permutations). They are reported for completeness; with several comparisons and metrics, **treat them as exploratory and do not read them as multiplicity-corrected**.
* **Win rate** is the fraction of seeds where method A strictly beats method B; **d_z** is the paired standardised effect size.

## Known threats to validity

* Synthetic, low-dimensional, linear data; behaviour on deep or real data may differ.
* Local-only is a strong baseline here because each site has enough data for a 7-parameter model; the data-scarcity grid explores where that stops being true.
* The selection rule uses oracle validation. A deployable version would have noisier signals, which would probably *reduce* its benefit.
* The tolerance, step count, learning rate, FedProx/Ditto `mu`, and trim fraction were not tuned per method; this may disadvantage FedProx and trimmed mean in particular.
* Ablation and grid cells use 10 seeds; the main comparison, severity sweep, and real-data runs use 20.
* A single drift event with a fixed subset of drifted sites is only one of many possible drift scenarios.
