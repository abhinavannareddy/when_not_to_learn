"""Temporal-drift benchmark: clients whose data changes *during* federated training.

Unlike :mod:`when_not_to_learn.core`, where the shift only appears in the held-out test
set, here a subset of sites changes regime during training. Every method is evaluated
after every round against the regime that is currently in force, so we can measure damage
at the drift event and the time needed to recover.

Three drift schedules are supported (``DriftConfig.drift_kind``):

* ``abrupt``    - drifted sites switch to the new regime at ``drift_round``.
* ``gradual``   - the new regime is blended in linearly over ``ramp_rounds`` rounds.
* ``recurring`` - drifted sites alternate between regimes every ``period`` rounds.

This is a synthetic benchmark. Nothing here is evidence of real-world effectiveness.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .core import Array, accuracy, log_loss, predict_proba, train_logistic

ALL_METHODS = ("local_only", "fedavg", "fedavg_ft", "fedprox", "ditto", "median",
               "trimmed_mean", "selective", "site_gate")
METHODS = ALL_METHODS

METHOD_LABELS = {
    "local_only": "Local only",
    "fedavg": "FedAvg",
    "fedavg_ft": "FedAvg + fine-tune",
    "fedprox": "FedProx",
    "ditto": "Ditto",
    "median": "Coordinate median",
    "trimmed_mean": "Trimmed mean",
    "selective": "Selective federation",
    "site_gate": "Selective + site gate",
}


@dataclass(frozen=True)
class DriftConfig:
    """Scenario definition. All fields are explicit so a run can be reproduced exactly."""

    n_clients: int = 6
    n_features: int = 6
    samples_per_client: int = 900
    rounds: int = 30
    drift_round: int = 15
    severity: float = 1.5
    drift_fraction: float = 0.34
    heterogeneity: float = 1.0
    tolerance: float = 0.015
    local_steps: int = 30
    lr: float = 0.12
    drift_kind: str = "abrupt"      # abrupt | gradual | recurring
    ramp_rounds: int = 10           # gradual drift: rounds to blend the new regime in
    period: int = 5                 # recurring drift: rounds per regime
    val_size: int | None = None     # cap on validation samples seen by the selection rule
    prox_mu: float = 0.1            # FedProx / Ditto proximal strength
    trim_fraction: float = 0.2      # trimmed mean: fraction trimmed from each tail


@dataclass
class DriftClient:
    name: str
    drifted: bool
    pre: Dict[str, Array] = field(default_factory=dict)
    post: Dict[str, Array] = field(default_factory=dict)


def _sample(rng: np.random.Generator, n: int, offset: Array, slope: Array, noise: float = 0.8):
    X = rng.normal(size=(n, len(offset))) + offset
    logits = X @ slope + 0.25 * offset.sum() + rng.normal(0, noise, size=n)
    return X, (logits > 0).astype(float)


def _split(rng: np.random.Generator, n: int, offset: Array, slope: Array) -> Dict[str, Array]:
    X, y = _sample(rng, n, offset, slope)
    a, b = int(0.6 * n), int(0.8 * n)
    return {"X_train": X[:a], "y_train": y[:a], "X_val": X[a:b], "y_val": y[a:b],
            "X_test": X[b:], "y_test": y[b:]}


def make_drift_clients(cfg: DriftConfig, seed: int) -> List[DriftClient]:
    """Build clients with a pre-drift and a post-drift regime.

    Drifted sites receive both a covariate shift (feature offset) and a concept shift
    (changed decision boundary), each scaled by ``cfg.severity``. Non-drifted sites keep
    the same distribution, drawn afresh, so their post-drift data is stationary.
    """
    rng = np.random.default_rng(seed)
    base_slope = rng.normal(0, 0.8, size=cfg.n_features)
    n_drift = max(1, round(cfg.drift_fraction * cfg.n_clients)) if cfg.drift_fraction > 0 else 0
    drifted_ids = set(rng.choice(cfg.n_clients, size=n_drift, replace=False).tolist())
    clients = []
    for i in range(cfg.n_clients):
        offset = rng.normal(0, 0.9 * cfg.heterogeneity, size=cfg.n_features)
        slope = base_slope + rng.normal(0, 0.45 * cfg.heterogeneity, size=cfg.n_features)
        client = DriftClient(f"site_{i + 1}", i in drifted_ids)
        client.pre = _split(rng, cfg.samples_per_client, offset, slope)
        if client.drifted:
            offset2 = offset + cfg.severity * rng.normal(0, 1.0, size=cfg.n_features)
            slope2 = slope + cfg.severity * rng.normal(0, 0.5, size=cfg.n_features)
        else:
            offset2, slope2 = offset, slope
        client.post = _split(rng, cfg.samples_per_client, offset2, slope2)
        clients.append(client)
    return clients


def new_regime_weight(cfg: DriftConfig, r: int) -> float:
    """Fraction (0-1) of the *new* regime in force at round ``r`` for a drifted site."""
    if cfg.drift_kind == "abrupt":
        return 1.0 if r > cfg.drift_round else 0.0
    if cfg.drift_kind == "gradual":
        return float(np.clip((r - cfg.drift_round) / max(cfg.ramp_rounds, 1), 0.0, 1.0))
    if cfg.drift_kind == "recurring":
        return 1.0 if r > cfg.drift_round and ((r - cfg.drift_round - 1) // cfg.period) % 2 == 0 else 0.0
    raise ValueError(f"unknown drift_kind: {cfg.drift_kind!r}")


def regime_data(client: DriftClient, w: float) -> Dict[str, Array]:
    """Data for a client when a fraction ``w`` of its data comes from the ``post`` draw.

    Non-drifted sites share one distribution across both draws, so blending is harmless there.
    """
    if w <= 0.0:
        return client.pre
    if w >= 1.0:
        return client.post
    out = {}
    for key in client.pre:
        a, b = client.pre[key], client.post[key]
        k = int(round(w * len(a)))
        out[key] = np.concatenate([a[: len(a) - k], b[:k]])
    return out


# --- local training variants -------------------------------------------------------------

def train_prox(X: Array, y: Array, initial: Array, anchor: Array, mu: float,
               steps: int, lr: float, l2: float = 1e-3) -> Array:
    """Logistic regression with a proximal pull toward ``anchor`` (FedProx / Ditto local step)."""
    w = initial.copy()
    for _ in range(steps):
        p = predict_proba(X, w)
        grad = np.zeros_like(w)
        grad[:-1] = X.T @ (p - y) / len(y) + l2 * w[:-1]
        grad[-1] = np.mean(p - y)
        grad += mu * (w - anchor)
        w -= lr * grad
    return w


# --- aggregation strategies --------------------------------------------------------------

def _weighted_mean(updates: List[Array], sizes: List[int]) -> Array:
    return np.average(np.stack(updates), axis=0, weights=np.asarray(sizes, dtype=float))


def _trimmed_mean(updates: List[Array], frac: float) -> Array:
    stack = np.sort(np.stack(updates), axis=0)
    k = min(int(np.floor(frac * len(updates))), (len(updates) - 1) // 2)
    return stack[k: len(updates) - k].mean(axis=0)


def _selective_aggregate(global_w: Array, updates: List[Array], sizes: List[int],
                         val: List[tuple], tolerance: float) -> tuple[Array, int]:
    """Accept updates that do not worsen mean or worst-site validation loss beyond tolerance."""
    base = [log_loss(X, y, global_w) for X, y in val]
    base_mean, base_worst = float(np.mean(base)), float(np.max(base))
    scores = []
    for u in updates:
        losses = [log_loss(X, y, u) for X, y in val]
        scores.append((float(np.mean(losses)), float(np.max(losses))))
    keep = [i for i, (m, w) in enumerate(scores)
            if m <= base_mean + tolerance and w <= base_worst + tolerance]
    if not keep:  # least harmful candidate keeps the run deterministic and progressing
        keep = [min(range(len(scores)), key=lambda i: sum(scores[i]))]
    return _weighted_mean([updates[i] for i in keep], [sizes[i] for i in keep]), len(keep)


def _evaluate(models: List[Array], data: List[Dict[str, Array]]) -> Dict[str, float]:
    acc = [accuracy(d["X_test"], d["y_test"], w) for w, d in zip(models, data)]
    loss = [log_loss(d["X_test"], d["y_test"], w) for w, d in zip(models, data)]
    return {"mean_accuracy": float(np.mean(acc)), "worst_site_accuracy": float(np.min(acc)),
            "mean_log_loss": float(np.mean(loss))}


# --- experiment --------------------------------------------------------------------------

def run_drift_experiment(cfg: DriftConfig, seed: int, methods: Iterable[str] = ALL_METHODS,
                         clients: List[DriftClient] | None = None) -> Dict[str, object]:
    """Run the selected methods through the same drifting scenario.

    ``methods`` may be any subset of :data:`ALL_METHODS`. Methods are independent except
    that ``site_gate`` needs ``selective`` and ``local_only``; they are added automatically.
    ``clients`` lets callers supply their own data (e.g. a real dataset).
    """
    methods = list(methods)
    if "site_gate" in methods:
        methods += [m for m in ("selective", "local_only") if m not in methods]
    active = [m for m in ALL_METHODS if m in methods]
    run = set(active)

    clients = clients if clients is not None else make_drift_clients(cfg, seed)
    dim = clients[0].pre["X_train"].shape[1] + 1
    state = {m: np.zeros(dim) for m in ALL_METHODS}
    local_models = [np.zeros(dim) for _ in clients]
    ditto_models = [np.zeros(dim) for _ in clients]
    trajectories: Dict[str, List[Dict[str, float]]] = {m: [] for m in active}
    accepted: List[int] = []
    gate_share: List[float] = []
    rng = np.random.default_rng(seed + 10_000)  # only used for validation subsampling

    for r in range(1, cfg.rounds + 1):
        w_new = new_regime_weight(cfg, r)
        data = [regime_data(c, w_new) for c in clients]
        sizes = [len(d["y_train"]) for d in data]
        val = [(d["X_val"], d["y_val"]) for d in data]
        sel_val = val
        if cfg.val_size is not None:
            sel_val = []
            for X, y in val:
                idx = rng.choice(len(y), size=min(cfg.val_size, len(y)), replace=False)
                sel_val.append((X[idx], y[idx]))

        def local_updates(start: Array, data=data) -> List[Array]:
            return [train_logistic(d["X_train"], d["y_train"], start, cfg.local_steps, cfg.lr)
                    for d in data]

        if run & {"local_only", "site_gate"}:
            local_models = [train_logistic(d["X_train"], d["y_train"], w, cfg.local_steps, cfg.lr)
                            for w, d in zip(local_models, data)]
        if run & {"fedavg", "fedavg_ft"}:
            state["fedavg"] = _weighted_mean(local_updates(state["fedavg"]), sizes)
        if "median" in run:
            state["median"] = np.median(np.stack(local_updates(state["median"])), axis=0)
        if "trimmed_mean" in run:
            state["trimmed_mean"] = _trimmed_mean(local_updates(state["trimmed_mean"]), cfg.trim_fraction)
        if "fedprox" in run:
            anchor = state["fedprox"]
            ups = [train_prox(d["X_train"], d["y_train"], anchor, anchor, cfg.prox_mu,
                              cfg.local_steps, cfg.lr) for d in data]
            state["fedprox"] = _weighted_mean(ups, sizes)
        if "ditto" in run:
            g = state["ditto"]
            ups = [train_logistic(d["X_train"], d["y_train"], g, cfg.local_steps, cfg.lr) for d in data]
            ditto_models = [train_prox(d["X_train"], d["y_train"], p, g, cfg.prox_mu,
                                       cfg.local_steps, cfg.lr) for p, d in zip(ditto_models, data)]
            state["ditto"] = _weighted_mean(ups, sizes)
        if "selective" in run:
            state["selective"], n_ok = _selective_aggregate(
                state["selective"], local_updates(state["selective"]), sizes, sel_val, cfg.tolerance)
            accepted.append(n_ok)

        per_site: Dict[str, List[Array]] = {}
        if "local_only" in run:
            per_site["local_only"] = local_models
        if "fedavg_ft" in run:  # personalised baseline: a few local steps on the FedAvg model
            per_site["fedavg_ft"] = [train_logistic(d["X_train"], d["y_train"], state["fedavg"], 10, cfg.lr)
                                     for d in data]
        if "ditto" in run:
            per_site["ditto"] = ditto_models
        if "site_gate" in run:
            # Per-site abstention: each site keeps whichever of {its local model, the selective
            # global model} has lower validation loss, i.e. it decides *not* to learn from the
            # federation when the federation does not help it.
            g = state["selective"]
            use_g = [log_loss(*v, g) <= log_loss(*v, loc) for loc, v in zip(local_models, val)]
            per_site["site_gate"] = [g if u else loc for u, loc in zip(use_g, local_models)]
            gate_share.append(float(np.mean(use_g)))
        for m in active:
            models = per_site.get(m) or [state[m]] * len(clients)
            trajectories[m].append({"round": r, **_evaluate(models, data)})

    return {"seed": seed, "config": cfg.__dict__.copy(), "trajectories": trajectories,
            "selective_accepted": accepted, "gate_federated_share": gate_share,
            "drifted_sites": [c.name for c in clients if c.drifted],
            "scalars": {m: summarise_trajectory(trajectories[m], cfg) for m in active}}


def summarise_trajectory(traj: List[Dict[str, float]], cfg: DriftConfig) -> Dict[str, float]:
    """Scalar summaries of one method's trajectory.

    * ``drop``: fall in mean accuracy at the first post-drift round.
    * ``final_*``: mean over the last five rounds (less noisy than the last round).
    * ``post_auc``: mean accuracy over all post-drift rounds.
    * ``recovery_rounds``: rounds after the drift until mean accuracy first reaches 98 % of
      its final plateau (``cfg.rounds - cfg.drift_round`` if it never does).
    """
    acc = np.array([t["mean_accuracy"] for t in traj])
    worst = np.array([t["worst_site_accuracy"] for t in traj])
    loss = np.array([t["mean_log_loss"] for t in traj])
    d = cfg.drift_round
    plateau = acc[-5:].mean()
    post = acc[d:]
    hit = np.nonzero(post >= 0.98 * plateau)[0]
    recovery = float(hit[0]) if len(hit) else float(len(post))
    return {"pre_drift_accuracy": float(acc[d - 1]), "drop": float(acc[d - 1] - acc[d]),
            "final_accuracy": float(plateau), "final_worst_site_accuracy": float(worst[-5:].mean()),
            "final_log_loss": float(loss[-5:].mean()), "post_auc": float(post.mean()),
            "recovery_rounds": recovery}


def run_seeds(cfg: DriftConfig, seeds: List[int], methods: Iterable[str] = ALL_METHODS,
              progress: Callable[[int], None] | None = None) -> List[Dict[str, object]]:
    methods = list(methods)
    runs = []
    for s in seeds:
        runs.append(run_drift_experiment(cfg, s, methods))
        if progress:
            progress(s)
    return runs
