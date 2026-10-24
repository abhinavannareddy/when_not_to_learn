"""Small, reproducible federated-learning simulator for research prototyping.

This is a synthetic research baseline, not a production-ready or formally private FL system.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Array = np.ndarray


def sigmoid(z: Array) -> Array:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X: Array, weights: Array) -> Array:
    return sigmoid(X @ weights[:-1] + weights[-1])


def log_loss(X: Array, y: Array, weights: Array) -> float:
    p = np.clip(predict_proba(X, weights), 1e-8, 1.0 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def accuracy(X: Array, y: Array, weights: Array) -> float:
    return float(np.mean((predict_proba(X, weights) >= 0.5) == y))


def train_logistic(X: Array, y: Array, initial: Array | None = None,
                   steps: int = 80, lr: float = 0.15, l2: float = 1e-3) -> Array:
    """Full-batch gradient descent for a logistic regression model."""
    w = np.zeros(X.shape[1] + 1, dtype=float) if initial is None else initial.copy()
    for _ in range(steps):
        p = predict_proba(X, w)
        grad = np.zeros_like(w)
        grad[:-1] = X.T @ (p - y) / len(y) + l2 * w[:-1]
        grad[-1] = np.mean(p - y)
        w -= lr * grad
    return w


@dataclass
class Client:
    name: str
    X_train: Array
    y_train: Array
    X_val: Array
    y_val: Array
    X_test: Array
    y_test: Array


def _sample_client(rng: np.random.Generator, name: str, n: int,
                   offset: Array, slope: Array, noise: float = 0.8) -> Tuple[Array, Array]:
    X = rng.normal(size=(n, len(offset))) + offset
    logits = X @ slope + 0.25 * offset.sum() + rng.normal(0, noise, size=n)
    y = (logits > 0).astype(float)
    return X, y


def make_clients(seed: int = 7, n_clients: int = 5, n_features: int = 6,
                 samples_per_client: int = 900, drift_after_round: int = 12) -> List[Client]:
    """Create non-IID clients; test data shifts after a simulated equipment change.

    The `drift_after_round` parameter is retained for experiment configuration/documentation.
    Data is generated once, with each client's post-change test distribution represented
    by an additional offset so evaluation can compare pre/post-change performance.
    """
    rng = np.random.default_rng(seed)
    base_slope = rng.normal(0, 0.8, size=n_features)
    clients: List[Client] = []
    for i in range(n_clients):
        offset = rng.normal(0, 0.9, size=n_features)
        slope = base_slope + rng.normal(0, 0.45, size=n_features)
        X, y = _sample_client(rng, f"site_{i+1}", samples_per_client, offset, slope)
        order = rng.permutation(samples_per_client)
        X, y = X[order], y[order]
        a, b = int(0.60 * samples_per_client), int(0.80 * samples_per_client)
        X_train, y_train = X[:a], y[:a]
        X_val, y_val = X[a:b], y[a:b]
        # The held-out test set reflects a later operating condition / distribution shift.
        X_test, y_test = _sample_client(rng, f"site_{i+1}_post_shift",
                                         samples_per_client - b, offset + rng.normal(0, 1.0, n_features), slope)
        clients.append(Client(f"site_{i+1}", X_train, y_train, X_val, y_val, X_test, y_test))
    return clients


def fedavg(clients: List[Client], rounds: int = 25, local_steps: int = 30,
           lr: float = 0.12) -> Tuple[Array, List[Dict[str, float]]]:
    """FedAvg baseline: every client contributes each round, weighted by sample count."""
    dim = clients[0].X_train.shape[1]
    global_w = np.zeros(dim + 1)
    history: List[Dict[str, float]] = []
    for r in range(rounds):
        updates, weights = [], []
        for c in clients:
            local_w = train_logistic(c.X_train, c.y_train, global_w, local_steps, lr)
            updates.append(local_w)
            weights.append(len(c.y_train))
        global_w = np.average(np.stack(updates), axis=0, weights=np.array(weights))
        history.append({"round": float(r + 1), "accepted_clients": float(len(clients))})
    return global_w, history


def selective_federation(clients: List[Client], rounds: int = 25, local_steps: int = 30,
                         lr: float = 0.12, tolerance: float = 0.015,
                         min_clients: int = 1) -> Tuple[Array, List[Dict[str, float]]]:
    """Select client updates using mean and worst-site validation guardrails.

    In this simulator, each candidate update is tested as a hypothetical global model
    against every client's validation *metric*. An update is eligible only when it
    does not increase mean validation log-loss or worst-site validation log-loss beyond
    `tolerance`. This is intentionally an oracle-style research baseline: real deployments
    would need a protocol for obtaining trustworthy aggregate metrics without exposing
    private validation data. If no update passes, the least harmful candidate is used.
    """
    dim = clients[0].X_train.shape[1]
    global_w = np.zeros(dim + 1)
    history: List[Dict[str, float]] = []
    for r in range(rounds):
        baseline_losses = [log_loss(c.X_val, c.y_val, global_w) for c in clients]
        baseline_mean = float(np.mean(baseline_losses))
        baseline_worst = float(np.max(baseline_losses))
        candidates, scores = [], []
        for c in clients:
            candidate = train_logistic(c.X_train, c.y_train, global_w, local_steps, lr)
            candidate_losses = [log_loss(other.X_val, other.y_val, candidate) for other in clients]
            mean_loss = float(np.mean(candidate_losses))
            worst_loss = float(np.max(candidate_losses))
            candidates.append((candidate, len(c.y_train)))
            scores.append((mean_loss, worst_loss))
        accepted = [candidates[i] for i, (mean_loss, worst_loss) in enumerate(scores)
                    if mean_loss <= baseline_mean + tolerance
                    and worst_loss <= baseline_worst + tolerance]
        if len(accepted) < min_clients:
            # Keep progress deterministic even if every candidate fails the guardrails.
            best = min(range(len(scores)), key=lambda i: scores[i][0] + scores[i][1])
            accepted = [candidates[best]]
        global_w = np.average(np.stack([item[0] for item in accepted]), axis=0,
                              weights=np.array([item[1] for item in accepted]))
        history.append({"round": float(r + 1), "accepted_clients": float(len(accepted))})
    return global_w, history


def evaluate(model: Array, clients: List[Client], personalized_steps: int = 20) -> Dict[str, float]:
    """Report average and worst-site performance on each client's shifted test set."""
    accuracies, losses, personalized_accuracies = [], [], []
    for c in clients:
        accuracies.append(accuracy(c.X_test, c.y_test, model))
        losses.append(log_loss(c.X_test, c.y_test, model))
        local = train_logistic(c.X_train, c.y_train, model, steps=personalized_steps, lr=0.10)
        personalized_accuracies.append(accuracy(c.X_test, c.y_test, local))
    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "worst_site_accuracy": float(np.min(accuracies)),
        "mean_log_loss": float(np.mean(losses)),
        "personalized_mean_accuracy": float(np.mean(personalized_accuracies)),
    }


def run_experiment(seed: int = 7, rounds: int = 25) -> Dict[str, object]:
    clients = make_clients(seed=seed)
    # Independent local models: each site never receives information from other sites.
    local_models = [train_logistic(c.X_train, c.y_train) for c in clients]
    local_acc = [accuracy(c.X_test, c.y_test, w) for c, w in zip(clients, local_models)]
    fed_model, fed_history = fedavg(clients, rounds=rounds)
    selective_model, selective_history = selective_federation(clients, rounds=rounds)
    return {
        "local_only": {"mean_accuracy": float(np.mean(local_acc)),
                       "worst_site_accuracy": float(np.min(local_acc)),
                       "mean_log_loss": float(np.mean([log_loss(c.X_test, c.y_test, w) for c, w in zip(clients, local_models)])),
                       "personalized_mean_accuracy": float(np.mean(local_acc))},
        "fedavg": evaluate(fed_model, clients),
        "selective_federation": evaluate(selective_model, clients),
        "fedavg_history": fed_history,
        "selective_history": selective_history,
        "config": {"seed": seed, "rounds": rounds, "clients": len(clients), "samples_per_client": 900},
    }
