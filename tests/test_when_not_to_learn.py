import numpy as np
from when_not_to_learn.core import accuracy, log_loss, make_clients, run_experiment, train_logistic


def test_generated_clients_have_valid_shapes():
    clients = make_clients(seed=3, n_clients=3, n_features=4, samples_per_client=100)
    assert len(clients) == 3
    assert clients[0].X_train.shape[1] == 4
    assert set(np.unique(clients[0].y_train)).issubset({0.0, 1.0})


def test_training_produces_finite_metrics():
    clients = make_clients(seed=4, n_clients=2, n_features=3, samples_per_client=120)
    c = clients[0]
    model = train_logistic(c.X_train, c.y_train, steps=10)
    assert np.isfinite(log_loss(c.X_test, c.y_test, model))
    assert 0.0 <= accuracy(c.X_test, c.y_test, model) <= 1.0


def test_experiment_returns_all_methods():
    result = run_experiment(seed=5, rounds=2)
    assert {"local_only", "fedavg", "selective_federation"}.issubset(result)
    assert len(result["selective_history"]) == 2
