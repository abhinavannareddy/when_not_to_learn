"""Small, dependency-light statistics for comparing methods across random seeds."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], level: float = 0.95, n_boot: int = 10_000,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float(v.mean())
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def paired_comparison(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000,
                      seed: int = 0) -> Dict[str, float]:
    """Compare method ``a`` against ``b`` over the *same* seeds (paired design).

    Returns the mean difference ``a - b`` with a bootstrap 95 % CI, the fraction of seeds
    where ``a`` is strictly better (``win_rate``), a two-sided sign-flip permutation
    p-value, and the paired effect size (Cohen's d_z). Positive means ``a`` is larger, so
    interpret the sign according to whether the metric is higher-is-better.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    lo, hi = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed + 1)
    flips = rng.choice([-1.0, 1.0], size=(n_boot, n))
    null = np.abs((flips * diff).mean(axis=1))
    p = float((np.sum(null >= abs(diff.mean()) - 1e-15) + 1) / (n_boot + 1))
    sd = diff.std(ddof=1) if n > 1 else 0.0
    return {"mean_diff": float(diff.mean()), "ci_low": lo, "ci_high": hi,
            "win_rate": float(np.mean(diff > 0)), "p_value": p,
            "effect_size_dz": float(diff.mean() / sd) if sd > 0 else 0.0, "n": n}
