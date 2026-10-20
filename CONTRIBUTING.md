# Contributing

This repository is a research prototype. Reproducibility and clear limitations matter more than adding features quickly.

## Before opening a change

- Read `docs/RESEARCH_PLAN.md` and the limitations in `README.md`.
- Keep the change narrow and explain the research question it helps answer.
- Do not describe the system as privacy-preserving unless a defined threat model and evidence support that claim.

## Report an experiment

Include:

1. The exact command and configuration.
2. All random seeds used.
3. Baselines and metrics.
4. A concise interpretation, including negative results.
5. Known limitations or possible confounders.

## Development checks

```bash
python -m pip install -e '.[experiments,test]'
python -m pytest
python -m when_not_to_learn.experiment --seeds 1,2,3 --rounds 5 --out results_smoke
```

Please do not commit private datasets, credentials, or generated output directories unless the outputs are intentionally selected and documented as research artifacts.
