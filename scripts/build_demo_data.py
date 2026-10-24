"""Embed measured benchmark results into demo/index.html.

The demo is a single self-contained HTML file (it must work from ``file://``), so the
numbers are embedded as JSON between the ``MEASURED-DATA`` markers. Run this after
regenerating results:

    python scripts/build_demo_data.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
DEMO = ROOT / "demo" / "index.html"

LABELS = {
    "local_only": "Local only", "fedavg": "FedAvg", "fedavg_ft": "FedAvg + fine-tune",
    "fedprox": "FedProx", "ditto": "Ditto", "median": "Coordinate median",
    "trimmed_mean": "Trimmed mean", "selective": "Selective federation",
    "site_gate": "Selective + site gate",
}


def _methods(block: dict, mean_key: str, lo_key: str, hi_key: str, worst_key: str) -> list[dict]:
    return [{"key": k, "label": LABELS[k], "mean": v[mean_key], "lo": v[lo_key], "hi": v[hi_key],
             "worst": v[worst_key]} for k, v in block.items()]


def build() -> dict:
    main = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    agg = main["aggregate"]
    data = {
        "main": {
            "seeds": len(main["seeds"]),
            "gate_share": main["site_gate_federated_share"],
            "methods": [{"key": k, "label": LABELS[k], "mean": v["final_accuracy"]["mean"],
                         "lo": v["final_accuracy"]["ci_low"], "hi": v["final_accuracy"]["ci_high"],
                         "worst": v["final_worst_site_accuracy"]["mean"]} for k, v in agg.items()],
        },
        "grids": {},
        "real": {},
    }
    for tag in ("stationary", "drift"):
        with (RES / f"grid_{tag}.csv").open(encoding="utf-8") as f:
            data["grids"][tag] = [{k: (float(v) if k != "samples_per_client" else int(v)) for k, v in row.items()}
                                  for row in csv.DictReader(f)]
    real_path = RES / "realdata_summary.json"
    if real_path.exists():
        real = json.loads(real_path.read_text(encoding="utf-8"))
        data["real"] = {
            "seeds": len(real["seeds"]),
            "scenarios": {tag: {"gate_share": s["site_gate_federated_share"],
                                "methods": _methods(s["methods"], "mean_accuracy", "ci_low", "ci_high",
                                                    "worst_site_accuracy")}
                          for tag, s in real["scenarios"].items()},
        }
    return data


def main() -> None:
    payload = json.dumps(build(), separators=(",", ":"))
    html = DEMO.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!--MEASURED-DATA-->)(.*?)(<!--/MEASURED-DATA-->)", re.S)
    if not pattern.search(html):
        raise SystemExit("MEASURED-DATA markers not found in demo/index.html")
    block = f'<script id="measured-data" type="application/json">{payload}</script>'
    DEMO.write_text(pattern.sub(lambda m: m.group(1) + block + m.group(3), html), encoding="utf-8")
    print(f"Embedded {len(payload):,} bytes of measured results into {DEMO.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
