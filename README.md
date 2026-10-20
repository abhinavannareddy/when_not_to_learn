# Selective Federated Intelligence

A small, reproducible research prototype exploring a simple question:

> Can a federation improve collaboration by declining client updates that appear harmful to local validation performance?

This repository is an **experimental baseline**, not a novel or production-ready federated-learning system. It uses synthetic data and logistic regression to make the experiment easy to reproduce before expanding to realistic datasets and stronger methods.

## Research question

Under client heterogeneity and distribution shift, can validation-guided selective participation improve average and worst-client performance compared with local-only training and standard FedAvg, without excessive participation or communication costs?

## What is implemented

- Synthetic non-IID clients representing independent sites.
- Local-only baseline.
- Standard FedAvg baseline.
- A transparent selective-participation baseline: a candidate client update is eligible when its hypothetical global model does not increase mean or worst-site validation log-loss beyond a configurable absolute tolerance (simulated metric-oracle baseline).
- Shifted held-out test data to approximate changed operating conditions.
- Metrics for mean accuracy, worst-site accuracy, mean log-loss, and personalized accuracy.
- Round-by-round accepted-client counts, JSON/CSV output, and an optional participation plot.
- Small automated tests.

## Important limitations

1. The data is synthetic. Results do not establish real-world effectiveness.
2. The selective rule is deliberately simple and should be treated as a baseline to improve, not as a claimed research novelty.
3. The current implementation does not provide differential privacy, secure aggregation, cryptographic protections, or a formal privacy guarantee. Model updates and validation metrics can leak information.
4. Candidate updates are screened individually using every site's validation metrics in a simulated oracle. This is not a deployable privacy-preserving protocol, and the final aggregated update is not guaranteed to satisfy the same guardrails.
5. The shifted test set is a static proxy for a later operating condition, not a full temporal drift simulation. The `drift_after_round` argument is currently descriptive and does not trigger a live shift during training.
6. The evaluation should be repeated across many random seeds and compared with established personalized/robust FL methods before drawing conclusions.

## Requirements

- Python 3.10+
- NumPy
- Optional: Matplotlib for a plot; pytest for tests

## Quick start (Windows PowerShell)

```powershell
cd selective-federated-intelligence
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[experiments,test]"
pytest
python -m sfi.experiment --seed 7 --rounds 25 --out results
```

If PowerShell blocks activation, run the commands below instead (no activation required):

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[experiments,test]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m sfi.experiment --seed 7 --rounds 25 --out results
```

## Quick start (macOS/Linux)

```bash
cd selective-federated-intelligence
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[experiments,test]'
pytest
python -m sfi.experiment --seed 7 --rounds 25 --out results
```

## Output files

The experiment writes the following into `results/`:

- `summary.json` — configuration, metrics, and per-round participation.
- `metrics.csv` — compact comparison table.
- `participation.png` — client participation by round (if Matplotlib is installed).

The results directory is ignored by Git by default. For a research repository, commit selected results only after you have decided they are useful and reproducible; never commit private or sensitive datasets.

## Suggested research workflow

1. Run the baseline unchanged and save the output.
2. Repeat over at least 10 seeds; report mean and standard deviation, not just the best run.
3. Add a true time-dependent drift event at a specified round.
4. Compare with a stronger personalized-FL baseline and at least one robust aggregation method.
5. Add ablations for validation tolerance, client count, non-IID severity, and drift magnitude.
6. Report mean performance, worst-client performance, calibration/false-confidence, bytes communicated, and recovery time after drift.
7. Only investigate privacy attacks after the basic experiment is reproducible; do not describe this prototype as privacy-preserving.

## Repository layout

```text
selective-federated-intelligence/
├── src/sfi/core.py          # data generation, training, baselines, evaluation
├── src/sfi/experiment.py    # CLI and results output
├── tests/test_core.py       # smoke tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Research integrity note

This project is a starting point for a research discussion, not evidence that the proposed idea is novel. A literature review is required to position the contribution against selective participation, personalized federated learning, client drift detection, and robust aggregation work.
