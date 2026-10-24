# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0]

### Added
- **Mid-training drift benchmark** (`when_not_to_learn.drift`): per-round evaluation, drop-at-drift and recovery metrics.
- **New baselines**: FedAvg + local fine-tuning, coordinate-wise median aggregation, and a per-site abstention gate on top of selective federation.
- **Statistics module** (`when_not_to_learn.stats`): bootstrap confidence intervals and paired comparisons (permutation p-value, win rate, d_z).
- **Benchmark runner** (`wnl-benchmark` / `python -m when_not_to_learn.benchmark`) with heterogeneity × local-data ablation grids, a drift-severity sweep, and publication-style figures.
- Documentation: `docs/METHODOLOGY.md`, `docs/RESULTS.md`; committed result data in `docs/results/` and figures in `assets/results/`.
- Ruff linting and a benchmark smoke test in CI; Python 3.13 added to the matrix.

### Changed
- README rewritten around reproducible headline findings and honest limitations.
- Research plan milestones M1–M3 marked complete; M4 partially complete.

## [0.2.0]
- Research prototype: local-only, FedAvg, and selective-federation baselines; multi-seed CLI; interactive Decision Observatory demo; CI.

## [0.1.0]
- Initial prototype of selective federated intelligence.
