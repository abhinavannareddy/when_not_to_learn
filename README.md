# When Not to Learn

### Selective collaboration for more reliable federated learning

**Research question:** *When should a participant's update be included in collaborative training—and when should a model stay local, personalise, or abstain?*

`when_not_to_learn` is an open research prototype for studying selective participation in federated learning under client heterogeneity and distribution shift. It compares local-only training, conventional Federated Averaging (FedAvg), and a transparent validation-guided selection baseline on synthetic multi-site data.

> **Research status:** early experimental baseline. The code is intended to make assumptions testable—not to claim novelty, guarantee improvements, or provide production-grade privacy.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-orange">
  <img alt="Privacy" src="https://img.shields.io/badge/privacy-not%20formally%20guaranteed-lightgrey">
</p>

## Why this problem matters

Federated learning lets organisations train collaboratively without pooling their raw training records. But their data can differ: hospitals serve different populations, telecom operators cover different geographies, and factories run different equipment. A shared update may help one participant while hurting another. Those relationships can also change over time.

This project asks whether a federation can use evidence about local and worst-site performance to decide when collaboration is useful, instead of assuming every update should always be aggregated.

## What is implemented today

- **Synthetic non-IID clients** representing independent sites.
- **Local-only baseline**: each site trains without cross-site updates.
- **FedAvg baseline**: all clients contribute in each communication round.
- **Selective-federation baseline**: candidate updates are screened using simulated validation metrics across all sites, with mean- and worst-site-loss guardrails.
- **Shifted held-out test distributions** to probe generalisation beyond each site's training distribution.
- **Evaluation metrics**: mean accuracy, worst-site accuracy, mean log-loss, and personalised mean accuracy.
- **Reproducible runs** using explicit random seeds; optional multi-seed aggregation reports means and standard deviations.
- **Artifacts**: JSON summary, per-seed CSV, and an optional participation plot.
- **Automated tests and CI** to catch basic regressions.

## Quick start

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[experiments,test]"
python -m pytest
```

Run one reproducible experiment:

```powershell
python -m when_not_to_learn.experiment --seed 7 --rounds 25 --out results
```

Run a small multi-seed sweep (recommended over interpreting one run):

```powershell
python -m when_not_to_learn.experiment --seeds 1,2,3,4,5,6,7,8,9,10 --rounds 25 --out results_10_seeds
```

If PowerShell blocks virtual-environment activation, use `\.venv\Scripts\python.exe` in place of `python` in the commands above.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[experiments,test]'
python -m pytest
python -m when_not_to_learn.experiment --seeds 1,2,3,4,5 --rounds 25 --out results
```

## Outputs

| File | Contents |
|---|---|
| `summary.json` | Configuration, individual run outputs, and aggregate mean / standard deviation across seeds. |
| `metrics.csv` | One row per seed and method for analysis in Python, R, Excel, or a notebook. |
| `participation.png` | Per-round participation diagnostic for the first seed in the run. |

Generated `results/` directories are ignored by Git by default. Commit result artifacts only when the experiment is reproducible and you want to preserve them as a research record.

## Research design and limitations

This is a deliberately small baseline, not a state-of-the-art benchmark. Please read [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) before interpreting the outputs.

1. **Synthetic data only.** The current results cannot establish effectiveness in hospitals, telecom, or industrial settings.
2. **The selection rule is not claimed as novel.** Validation-guided selection is a starting point for comparison and critique.
3. **Oracle-style validation.** Candidate updates are evaluated against every simulated site's validation set. This central evaluator has access to site-level validation data/metrics; it is not a deployable privacy-preserving protocol.
4. **No formal privacy guarantee.** There is no differential privacy, secure aggregation, cryptographic protection, or formal leakage analysis in this version. Federated learning alone does not guarantee privacy.
5. **Shift is currently a held-out distribution proxy.** The current code does not yet trigger a genuine time-dependent drift event mid-training; that is a priority for the next milestone.
6. **Guardrails are not guarantees.** Individually screened candidates can interact badly when aggregated, and the final aggregate is not guaranteed to satisfy the same thresholds.
7. **No result is guaranteed to win.** If selective federation underperforms, report it and investigate why. Do not select only favourable seeds or metrics.

## Roadmap

- [x] Establish local-only and FedAvg baselines.
- [x] Implement an explicit selective-update baseline.
- [x] Add repeatable seed-based evaluation and machine-readable outputs.
- [x] Add automated tests and continuous integration.
- [ ] Add true temporal drift during training and recovery-time measurements.
- [ ] Add a stronger personalised-FL baseline and a robust-aggregation baseline.
- [ ] Add confidence intervals and paired statistical comparisons across seeds.
- [ ] Add ablations for client heterogeneity, selection tolerance, number of clients, and drift severity.
- [ ] Investigate privacy leakage and realistic privacy-preserving evaluation protocols.
- [ ] Evaluate on an appropriate public benchmark before making applied claims.

## Repository map

```text
when_not_to_learn/
├── .github/workflows/tests.yml    # Continuous integration
├── docs/RESEARCH_PLAN.md          # Research questions, hypotheses, milestones
├── src/when_not_to_learn/
│   ├── core.py                    # Synthetic clients, training, baselines, metrics
│   └── experiment.py              # CLI, multi-seed runs, output files
├── tests/                         # Automated smoke tests
├── CITATION.cff                   # Citation metadata
├── LICENSE
├── pyproject.toml
└── README.md
```

## How to contribute or challenge the idea

Useful contributions include a reproducible counterexample, a stronger baseline, a drift scenario, a metric critique, or a well-motivated change to the selection rule. Please include the command, seed(s), configuration, and output needed to reproduce a reported result. See the research plan before opening a feature request.

## Citation

If you reference this prototype, use the metadata in [`CITATION.cff`](CITATION.cff). The repository is an evolving research artifact, not a peer-reviewed publication.

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
