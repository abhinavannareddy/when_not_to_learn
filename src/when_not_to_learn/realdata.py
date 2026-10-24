"""Real-data experiments: the same methods on public tabular datasets with non-IID partitions.

Datasets ship inside scikit-learn (no download required):

* ``digits``  - 8x8 handwritten digits (1,797 samples, 64 features). Binary task: digit >= 5.
  Sites receive **label-skewed** shares of the ten digit classes drawn from a Dirichlet(alpha)
  prior; small ``alpha`` means strongly different class mixes per site.
* ``cancer``  - Wisconsin diagnostic breast cancer (569 samples, 30 features). Binary task:
  malignant vs benign. Sites are **covariate-skewed**: samples are ranked by a noisy version of
  the first principal component and cut into contiguous blocks.

There is no temporal drift here (the "pre" and "post" regimes are identical), so this tests the
heterogeneity question on real data. Requires ``pip install scikit-learn``.

    python -m when_not_to_learn.realdata --seeds 20 --out results/realdata
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np

from .drift import METHOD_LABELS, DriftClient, DriftConfig, run_drift_experiment
from .stats import bootstrap_ci, paired_comparison

REAL_METHODS = ("local_only", "fedavg", "fedavg_ft", "fedprox", "ditto", "median",
                "trimmed_mean", "selective", "site_gate")


def _load(name: str):
    try:
        from sklearn import datasets
    except ImportError as exc:  # pragma: no cover - exercised only without scikit-learn
        raise SystemExit("scikit-learn is required: pip install 'when-not-to-learn[realdata]'") from exc
    if name == "digits":
        d = datasets.load_digits()
        return d.data.astype(float), (d.target >= 5).astype(float), d.target
    if name == "cancer":
        d = datasets.load_breast_cancer()
        return d.data.astype(float), (1 - d.target).astype(float), d.target
    raise ValueError(f"unknown dataset {name!r}")


def partition(name: str, seed: int, n_clients: int = 6, alpha: float = 0.5) -> List[np.ndarray]:
    """Return index arrays, one per site."""
    X, _, classes = _load(name)
    rng = np.random.default_rng(seed)
    if name == "digits":
        shares = rng.dirichlet(alpha * np.ones(n_clients), size=10)   # class -> site shares
        sites: List[list] = [[] for _ in range(n_clients)]
        for c in range(10):
            idx = rng.permutation(np.nonzero(classes == c)[0])
            cuts = (np.cumsum(shares[c]) * len(idx)).astype(int)[:-1]
            for s, part in enumerate(np.split(idx, cuts)):
                sites[s].extend(part.tolist())
        return [np.array(s) for s in sites]
    Xc = X - X.mean(0)
    pc1 = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][0]
    score = pc1 + rng.normal(0, 0.5 * pc1.std(), size=len(pc1))
    return [np.array(b) for b in np.array_split(np.argsort(score), n_clients)]


def make_real_clients(name: str, seed: int, n_clients: int = 6, alpha: float = 0.5) -> List[DriftClient]:
    X, y, _ = _load(name)
    rng = np.random.default_rng(seed + 1)
    clients = []
    for i, idx in enumerate(partition(name, seed, n_clients, alpha)):
        idx = rng.permutation(idx)
        a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
        tr, va, te = idx[:a], idx[a:b], idx[b:]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6      # standardise with *local* training stats
        part = {"X_train": (X[tr] - mu) / sd, "y_train": y[tr], "X_val": (X[va] - mu) / sd,
                "y_val": y[va], "X_test": (X[te] - mu) / sd, "y_test": y[te]}
        clients.append(DriftClient(f"site_{i + 1}", False, part, part))
    return clients


SCENARIOS = {
    "digits_alpha0.1": ("digits", 0.1),
    "digits_alpha0.5": ("digits", 0.5),
    "digits_alpha5": ("digits", 5.0),
    "cancer_covariate": ("cancer", 0.0),
}


def run_scenario(tag: str, seeds: List[int], methods=REAL_METHODS) -> List[dict]:
    name, alpha = SCENARIOS[tag]
    cfg = replace(DriftConfig(), drift_round=15, rounds=30, lr=0.1)
    return [run_drift_experiment(cfg, s, methods, clients=make_real_clients(name, s, 6, alpha))
            for s in seeds]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the methods on real, non-IID partitions.")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/realdata")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(1, args.seeds + 1))
    summary = {}
    for tag in SCENARIOS:
        print(f"scenario {tag}", flush=True)
        runs = run_scenario(tag, seeds)
        def col(m, k, runs=runs):
            return [r["scalars"][m][k] for r in runs]

        res = {}
        for m in REAL_METHODS:
            lo, hi = bootstrap_ci(col(m, "final_accuracy"))
            res[m] = {"mean_accuracy": float(np.mean(col(m, "final_accuracy"))), "ci_low": lo, "ci_high": hi,
                      "worst_site_accuracy": float(np.mean(col(m, "final_worst_site_accuracy")))}
        paired = {f"{a}_vs_{b}": paired_comparison(col(a, "final_accuracy"), col(b, "final_accuracy"))
                  for a, b in (("selective", "fedavg"), ("selective", "local_only"), ("fedavg", "local_only"),
                               ("site_gate", "local_only"), ("ditto", "local_only"))}
        summary[tag] = {"methods": res, "paired": paired,
                        "site_gate_federated_share": float(np.mean([np.mean(r["gate_federated_share"]) for r in runs]))}
    (out / "summary.json").write_text(json.dumps({"seeds": seeds, "scenarios": summary}, indent=2),
                                      encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    from .benchmark import COLORS, _style
    _style(plt)
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(14, 3.8), sharey=True)
    for ax, (tag, s) in zip(axes, summary.items()):
        ms = list(REAL_METHODS)
        ys = [s["methods"][m]["mean_accuracy"] for m in ms]
        err = [[s["methods"][m]["mean_accuracy"] - s["methods"][m]["ci_low"] for m in ms],
               [s["methods"][m]["ci_high"] - s["methods"][m]["mean_accuracy"] for m in ms]]
        ax.bar(range(len(ms)), ys, yerr=err, color=[COLORS[m] for m in ms], capsize=2)
        ax.set_xticks(range(len(ms)), [METHOD_LABELS[m] for m in ms], rotation=60, ha="right", fontsize=7)
        ax.set_title(tag.replace("_", " "), fontsize=10)
        ax.set_ylim(min(0.5, min(ys) - 0.05), 1.0)
    axes[0].set_ylabel("Mean site test accuracy (95% CI)")
    fig.tight_layout()
    fig.savefig(out / "realdata.png", bbox_inches="tight")
    print(f"Wrote results to {out.resolve()}")


if __name__ == "__main__":
    main()
