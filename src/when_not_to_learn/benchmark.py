"""One-command drift benchmark: main comparison, paired statistics, ablation sweeps, figures.

    python -m when_not_to_learn.benchmark --seeds 20 --sweep-seeds 10 --out results/benchmark
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from .drift import METHOD_LABELS, METHODS, DriftConfig, run_seeds
from .stats import bootstrap_ci, paired_comparison

SCALARS = ("final_accuracy", "final_worst_site_accuracy", "final_log_loss", "post_auc",
           "drop", "recovery_rounds")
# Okabe-Ito colour-blind-safe palette
COLORS = {"local_only": "#0072B2", "fedavg": "#D55E00", "fedavg_ft": "#CC79A7",
          "median": "#999999", "selective": "#009E73", "site_gate": "#E69F00"}


def _column(runs: List[dict], method: str, key: str) -> List[float]:
    return [r["scalars"][method][key] for r in runs]


def aggregate(runs: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for m in METHODS:
        out[m] = {}
        for k in SCALARS:
            v = _column(runs, m, k)
            lo, hi = bootstrap_ci(v)
            out[m][k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                         "ci_low": lo, "ci_high": hi}
    return out


def paired_table(runs: List[dict]) -> Dict[str, dict]:
    pairs = [("selective", "fedavg"), ("selective", "median"), ("selective", "local_only"),
             ("site_gate", "selective"), ("site_gate", "local_only"), ("fedavg_ft", "fedavg")]
    return {f"{a}_vs_{b}": {k: paired_comparison(_column(runs, a, k), _column(runs, b, k))
                            for k in ("final_accuracy", "final_worst_site_accuracy", "post_auc")}
            for a, b in pairs}


def sweep(base: DriftConfig, field: str, values: List[float], seeds: List[int]) -> List[dict]:
    rows = []
    for v in values:
        runs = run_seeds(replace(base, **{field: v}), seeds)
        for m in METHODS:
            acc = _column(runs, m, "final_accuracy")
            lo, hi = bootstrap_ci(acc)
            rows.append({field: v, "method": m, "mean_final_accuracy": float(np.mean(acc)),
                         "ci_low": lo, "ci_high": hi,
                         "mean_worst_site_accuracy": float(np.mean(_column(runs, m, "final_worst_site_accuracy")))})
        print(f"  {field}={v}: done", flush=True)
    return rows


def grid(base: DriftConfig, hets: List[float], sizes: List[int], seeds: List[int]) -> List[dict]:
    """Final accuracy for every method over heterogeneity x local-sample-size."""
    rows = []
    for n in sizes:
        for h in hets:
            runs = run_seeds(replace(base, heterogeneity=h, samples_per_client=n), seeds)
            row = {"heterogeneity": h, "samples_per_client": n}
            for m in METHODS:
                row[m] = float(np.mean(_column(runs, m, "final_accuracy")))
            rows.append(row)
            print(f"  heterogeneity={h} n={n}: done", flush=True)
    return rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --- figures ----------------------------------------------------------------------------

def _style(plt):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
                         "axes.titleweight": "bold", "axes.titlesize": 11})


def plot_trajectories(runs: List[dict], cfg: DriftConfig, path: Path, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, key, title in ((axes[0], "mean_accuracy", "Mean site accuracy"),
                           (axes[1], "worst_site_accuracy", "Worst-site accuracy")):
        for m in METHODS:
            arr = np.array([[t[key] for t in r["trajectories"][m]] for r in runs])
            mu = arr.mean(0)
            se = arr.std(0, ddof=1) / np.sqrt(len(arr))
            x = np.arange(1, len(mu) + 1)
            ax.plot(x, mu, color=COLORS[m], label=METHOD_LABELS[m], lw=2 if m in ("selective", "site_gate") else 1.4)
            ax.fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=COLORS[m], alpha=0.12, lw=0)
        ax.axvline(cfg.drift_round + 0.5, color="black", ls="--", lw=1)
        ax.set(xlabel="Communication round", title=title)
    axes[0].text(cfg.drift_round + 0.8, axes[0].get_ylim()[0] + 0.01, "drift", fontsize=9)
    axes[0].set_ylabel("Accuracy on current regime")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"Accuracy through a mid-training drift ({len(runs)} seeds, 95% CI band)", y=1.0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_sweep(rows: List[dict], field: str, xlabel: str, title: str, path: Path, plt) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m in METHODS:
        pts = sorted((r[field], r["mean_final_accuracy"], r["ci_low"], r["ci_high"])
                     for r in rows if r["method"] == m)
        x, y, lo, hi = map(np.array, zip(*pts))
        ax.plot(x, y, "o-", color=COLORS[m], label=METHOD_LABELS[m], lw=1.8, ms=4)
        ax.fill_between(x, lo, hi, color=COLORS[m], alpha=0.10, lw=0)
    ax.set(xlabel=xlabel, ylabel="Final mean accuracy", title=title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_gain_map(rows: List[dict], path: Path, plt, scenario: str) -> None:
    """Heatmaps of (method - local only) accuracy: positive means collaboration helped."""
    hets = sorted({r["heterogeneity"] for r in rows})
    sizes = sorted({r["samples_per_client"] for r in rows})
    panels = (("fedavg", "FedAvg"), ("selective", "Selective federation"), ("site_gate", "Selective + site gate"))
    lim = 100 * max(abs(r[m] - r["local_only"]) for r in rows for m, _ in panels)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, (m, label) in zip(axes, panels):
        z = 100 * np.array([[next(r[m] - r["local_only"] for r in rows
                            if r["heterogeneity"] == h and r["samples_per_client"] == n)
                       for h in hets] for n in sizes])
        im = ax.imshow(z, origin="lower", cmap="PuOr", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(hets)), [str(h) for h in hets])
        ax.set_yticks(range(len(sizes)), [str(n) for n in sizes])
        for i in range(len(sizes)):
            for j in range(len(hets)):
                ax.text(j, i, f"{z[i, j]:+.1f}", ha="center", va="center", fontsize=8)
        ax.set(xlabel="Client heterogeneity", title=label)
        ax.grid(False)
    axes[0].set_ylabel("Training samples per site")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Accuracy vs local-only (pp)")
    fig.suptitle(f"When should a site learn collaboratively? Scenario: {scenario}\n"
                 "(purple = collaboration helps, orange = it hurts)", y=1.08)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_forest(paired: Dict[str, dict], path: Path, plt) -> None:
    rows = [(k, v["final_worst_site_accuracy"]) for k, v in paired.items()]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, (_name, r) in enumerate(rows):
        c = "#009E73" if r["ci_low"] > 0 else "#D55E00" if r["ci_high"] < 0 else "#666666"
        ax.errorbar(r["mean_diff"], i, xerr=[[r["mean_diff"] - r["ci_low"]], [r["ci_high"] - r["mean_diff"]]],
                    fmt="o", color=c, capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n.replace("_vs_", "  vs  ") for n, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="Paired difference in final worst-site accuracy (95% bootstrap CI)",
           title="Which comparisons are distinguishable from zero?")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Drift benchmark with statistics and ablations.")
    ap.add_argument("--seeds", type=int, default=20, help="Seeds for the main comparison.")
    ap.add_argument("--sweep-seeds", type=int, default=10, help="Seeds per ablation setting.")
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--no-sweeps", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = DriftConfig()
    seeds = list(range(1, args.seeds + 1))
    print(f"Main comparison: {len(seeds)} seeds", flush=True)
    runs = run_seeds(cfg, seeds)
    agg, paired = aggregate(runs), paired_table(runs)
    gate_share = float(np.mean([np.mean(r["gate_federated_share"]) for r in runs]))
    payload = {"config": cfg.__dict__, "seeds": seeds, "aggregate": agg, "paired": paired,
               "site_gate_federated_share": gate_share,
               "note": "Synthetic benchmark. Not evidence of real-world effectiveness."}
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(out / "scalars.csv", [{"seed": r["seed"], "method": m, **r["scalars"][m]}
                                     for r in runs for m in METHODS])

    sweeps, grid_rows = {}, None
    if not args.no_sweeps:
        sw = list(range(1, args.sweep_seeds + 1))
        print("Grid: heterogeneity x local data", flush=True)
        grid_rows = {}
        for tag, c in (("drift", cfg), ("stationary", replace(cfg, drift_fraction=0.0))):
            print(f"  scenario: {tag}", flush=True)
            grid_rows[tag] = grid(c, [0.1, 0.5, 1.0, 1.5], [60, 150, 400, 900], sw)
            _write_csv(out / f"grid_{tag}.csv", grid_rows[tag])
        print("Sweep: severity", flush=True)
        sweeps["severity"] = sweep(cfg, "severity", [0.0, 1.0, 2.0, 3.0, 4.0], sw)
        for name, rows in sweeps.items():
            _write_csv(out / f"sweep_{name}.csv", rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    _style(plt)
    plot_trajectories(runs, cfg, out / "trajectories.png", plt)
    plot_forest(paired, out / "paired_differences.png", plt)
    if sweeps:
        for tag, rows in grid_rows.items():
            plot_gain_map(rows, out / f"gain_map_{tag}.png", plt,
                          "stationary data (no drift)" if tag == "stationary" else "a third of sites drift mid-training")
        plot_sweep(sweeps["severity"], "severity", "Drift severity (scale)",
                   "Robustness to drift severity", out / "sweep_severity.png", plt)
    print(f"Wrote results to {out.resolve()}")


if __name__ == "__main__":
    main()
