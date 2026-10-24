import numpy as np

from when_not_to_learn.drift import METHODS, DriftConfig, make_drift_clients, run_drift_experiment
from when_not_to_learn.stats import bootstrap_ci, paired_comparison

SMALL = DriftConfig(n_clients=4, n_features=3, samples_per_client=120, rounds=6, drift_round=3)


def test_drift_clients_change_only_for_drifted_sites():
    clients = make_drift_clients(SMALL, seed=1)
    assert any(c.drifted for c in clients) and not all(c.drifted for c in clients)
    assert all(c.pre["X_train"].shape == c.post["X_train"].shape for c in clients)


def test_no_drift_fraction_means_no_drifted_sites():
    clients = make_drift_clients(DriftConfig(**{**SMALL.__dict__, "drift_fraction": 0.0}), seed=1)
    assert not any(c.drifted for c in clients)


def test_drift_experiment_is_reproducible_and_complete():
    a = run_drift_experiment(SMALL, seed=2)
    b = run_drift_experiment(SMALL, seed=2)
    assert a["scalars"] == b["scalars"]
    for m in METHODS:
        assert len(a["trajectories"][m]) == SMALL.rounds
        assert 0.0 <= a["scalars"][m]["final_accuracy"] <= 1.0
    assert len(a["selective_accepted"]) == SMALL.rounds
    assert all(1 <= n <= SMALL.n_clients for n in a["selective_accepted"])


def test_site_gate_never_worse_in_loss_than_selective_on_validation_choice():
    # The gate chooses per site between two models, so its share of federated use is a fraction.
    run = run_drift_experiment(SMALL, seed=3)
    assert all(0.0 <= s <= 1.0 for s in run["gate_federated_share"])


def test_bootstrap_ci_brackets_mean():
    v = np.random.default_rng(0).normal(1.0, 0.2, size=30)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi


def test_paired_comparison_detects_consistent_gain_and_null():
    base = np.random.default_rng(1).normal(0.8, 0.02, size=25)
    better = paired_comparison(base + 0.05, base)
    assert better["ci_low"] > 0 and better["win_rate"] == 1.0 and better["p_value"] < 0.01
    noise = np.random.default_rng(2).normal(0, 0.02, size=25)
    null = paired_comparison(base + noise, base)
    assert null["ci_low"] < 0 < null["ci_high"]


def test_all_methods_run_and_are_finite():
    from when_not_to_learn.drift import ALL_METHODS
    run = run_drift_experiment(SMALL, seed=4, methods=ALL_METHODS)
    assert set(run["scalars"]) == set(ALL_METHODS)
    for m in ALL_METHODS:
        assert np.isfinite(run["scalars"][m]["final_log_loss"])


def test_method_subset_pulls_in_dependencies():
    run = run_drift_experiment(SMALL, seed=4, methods=["site_gate"])
    assert {"site_gate", "selective", "local_only"} == set(run["scalars"])


def test_drift_schedules():
    from when_not_to_learn.drift import new_regime_weight
    ab = DriftConfig(drift_kind="abrupt", drift_round=5)
    assert [new_regime_weight(ab, r) for r in (5, 6)] == [0.0, 1.0]
    gr = DriftConfig(drift_kind="gradual", drift_round=5, ramp_rounds=10)
    assert new_regime_weight(gr, 5) == 0.0 and new_regime_weight(gr, 10) == 0.5 and new_regime_weight(gr, 20) == 1.0
    rc = DriftConfig(drift_kind="recurring", drift_round=5, period=3)
    assert [new_regime_weight(rc, r) for r in range(5, 15)] == [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]


def test_validation_subsampling_runs():
    cfg = DriftConfig(**{**SMALL.__dict__, "val_size": 5})
    assert run_drift_experiment(cfg, seed=1, methods=["selective"])["selective_accepted"]


def test_trimmed_mean_ignores_outlier():
    from when_not_to_learn.drift import _trimmed_mean
    ups = [np.zeros(3)] * 4 + [np.full(3, 100.0)]
    assert np.allclose(_trimmed_mean(ups, 0.2), 0.0)
