"""Run reproducible single-seed or multi-seed research experiments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .core import run_experiment

METHODS = ("local_only", "fedavg", "selective_federation")
METRICS = ("mean_accuracy", "worst_site_accuracy", "mean_log_loss", "personalized_mean_accuracy")


def _summarise_runs(runs: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """Return mean and sample standard deviation across independent random seeds."""
    summary: dict[str, Any] = {"seeds": seeds, "n_runs": len(runs), "methods": {}}
    for method in METHODS:
        summary["methods"][method] = {}
        for metric in METRICS:
            values = [float(run[method][metric]) for run in runs]
            summary["methods"][method][metric] = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark local training, FedAvg, and selective federation."
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed for a single run (default: 7).")
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds for repeated runs, e.g. 1,2,3,4,5. Overrides --seed.",
    )
    parser.add_argument("--rounds", type=int, default=25, help="Federated communication rounds.")
    parser.add_argument("--out", default="results", help="Output directory (default: results).")
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()] if args.seeds else [args.seed]
    if not seeds:
        parser.error("--seeds must contain at least one integer seed")
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [run_experiment(seed=seed, rounds=args.rounds) for seed in seeds]

    payload: dict[str, Any] = {
        "project": "when_not_to_learn",
        "description": "Synthetic benchmark; not evidence of real-world effectiveness or formal privacy.",
        "config": {"seeds": seeds, "rounds": args.rounds, "n_runs": len(seeds)},
        "runs": runs,
        "aggregate": _summarise_runs(runs, seeds),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "method", *METRICS])
        for seed, run in zip(seeds, runs):
            for method in METHODS:
                writer.writerow([seed, method, *[run[method][metric] for metric in METRICS]])

    try:
        import matplotlib.pyplot as plt
        # Participation histories are available for each run; plot the first run to keep
        # this diagnostic readable. Numerical results for every seed are saved above.
        first = runs[0]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, label in (("fedavg_history", "FedAvg"), ("selective_history", "Selective federation")):
            history = first[key]
            ax.plot([item["round"] for item in history], [item["accepted_clients"] for item in history], label=label)
        ax.set(xlabel="Communication round", ylabel="Clients contributing", title=f"Participation by round (seed={seeds[0]})")
        ax.set_ylim(0, first["config"]["clients"] + 0.5)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "participation.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping participation plot")

    print(json.dumps(payload["aggregate"], indent=2))
    print(f"\nWrote results to: {out.resolve()}")


if __name__ == "__main__":
    main()
