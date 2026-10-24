import numpy as np
import pytest

pytest.importorskip("sklearn")

from when_not_to_learn.realdata import SCENARIOS, make_real_clients, partition  # noqa: E402


@pytest.mark.parametrize("name", ["digits", "cancer"])
def test_partition_covers_every_sample_once(name):
    parts = partition(name, seed=1, n_clients=6, alpha=0.5)
    flat = np.concatenate(parts)
    assert len(flat) == len(set(flat.tolist()))
    assert all(len(p) > 0 for p in parts)


def test_real_clients_are_standardised_and_binary():
    for name, alpha in SCENARIOS.values():
        clients = make_real_clients(name, seed=2, alpha=alpha)
        assert len(clients) == 6
        assert set(np.unique(clients[0].pre["y_train"])).issubset({0.0, 1.0})
        assert abs(clients[0].pre["X_train"].mean()) < 1e-6


def test_skew_increases_as_alpha_shrinks():
    def spread(alpha):
        cl = make_real_clients("digits", seed=3, alpha=alpha)
        return np.std([c.pre["y_train"].mean() for c in cl])
    assert spread(0.1) > spread(50.0)
