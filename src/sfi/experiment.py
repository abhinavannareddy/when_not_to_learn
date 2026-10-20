"""Run the benchmark and write JSON/CSV plus a participation plot."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .core import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Selective Federated Intelligence benchmark")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = run_experiment(seed=args.seed, rounds=args.rounds)
    (out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "mean_accuracy", "worst_site_accuracy", "mean_log_loss", "personalized_mean_accuracy"])
        for method in ("local_only", "fedavg", "selective_federation"):
            m = results[method]
            writer.writerow([method, m["mean_accuracy"], m["worst_site_accuracy"], m["mean_log_loss"], m["personalized_mean_accuracy"]])
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, label in (("fedavg_history", "FedAvg"), ("selective_history", "Selective federation")):
            hist = results[key]
            ax.plot([x["round"] for x in hist], [x["accepted_clients"] for x in hist], label=label)
        ax.set(xlabel="Communication round", ylabel="Clients contributing", title="Participation over training")
        ax.set_ylim(0, results["config"]["clients"] + 0.5)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "participation.png", dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipping participation plot")
    print(json.dumps({k: results[k] for k in ("local_only", "fedavg", "selective_federation")}, indent=2))
    print(f"\nWrote results to: {out.resolve()}")

if __name__ == "__main__":
    main()
