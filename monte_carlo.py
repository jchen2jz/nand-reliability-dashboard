"""
monte_carlo.py
==============
Monte Carlo layer for the NAND reliability model.

Two things live here:

1. simulate_population()
   Draw a large population of individual cells, each with its own Vt (plus a
   quenched per-cell process-variation offset that creates "weak" and "strong"
   cells), read every cell against the references, and count actual bit errors
   using Gray coding. This is the empirical counterpart to the closed-form
   analytical_rber() and is used to validate it.

2. crossing_prob_naive() / crossing_prob_is()
   Estimate a single ultra-rare tail probability (one state crossing its
   reference). Naive Monte Carlo needs ~1/p samples and collapses below ~1e-7.
   Importance sampling shifts the sampling distribution onto the boundary and
   reweights, reaching ~1e-12 with a modest sample count. This is how UBER-class
   probabilities (~1e-15) are estimated in practice.
"""

import numpy as np
from scipy.stats import norm

from nand_model import (degraded_distributions, midpoint_refs, optimal_refs)


def _popcount(arr: np.ndarray) -> np.ndarray:
    """Vectorized bit count for small non-negative integer arrays."""
    arr = arr.astype(np.int64).copy()
    c = np.zeros_like(arr)
    while arr.any():
        c += arr & 1
        arr >>= 1
    return c


def _gray(x: np.ndarray) -> np.ndarray:
    """Binary -> Gray code. Adjacent levels differ by exactly one bit."""
    x = x.astype(np.int64)
    return x ^ (x >> 1)


def simulate_population(cfg, cycles, ret_time_h, temp_C, reads,
                        n_cells=2_000_000, proc_var=0.05,
                        use_optimal_ref=False, seed=0):
    """Empirical RBER from a simulated cell population, with 95% CI."""
    rng = np.random.default_rng(seed)
    means, sig = degraded_distributions(cfg, cycles, ret_time_h, temp_C, reads)
    refs = optimal_refs(means, sig) if use_optimal_ref else midpoint_refs(means)

    levels = rng.integers(0, cfg.L, size=n_cells)          # true stored level
    cell_offset = rng.normal(0.0, proc_var, size=n_cells)  # quenched variation
    vt = rng.normal(means[levels], sig[levels]) + cell_offset

    read_level = np.digitize(vt, refs)                     # nearest level
    bit_err = _popcount(_gray(levels) ^ _gray(read_level)).sum()

    total_bits = n_cells * cfg.bits_per_cell
    rber = bit_err / total_bits
    se = np.sqrt(max(rber, 1e-12) * (1 - rber) / total_bits)
    return rber, (rber - 1.96 * se, rber + 1.96 * se)


def crossing_prob_naive(m, s, R, n, seed=0):
    """P(N(m,s) > R) by direct sampling. Fails when the true prob << 1/n."""
    rng = np.random.default_rng(seed)
    x = rng.normal(m, s, n)
    p = np.mean(x > R)
    return p


def crossing_prob_is(m, s, R, n, seed=0):
    """P(N(m,s) > R) via importance sampling: propose from N(R, s) (shifted
    onto the boundary) and reweight by the likelihood ratio f/g."""
    rng = np.random.default_rng(seed)
    x = rng.normal(R, s, n)                 # proposal g = N(R, s)
    w = norm.pdf(x, m, s) / norm.pdf(x, R, s)  # likelihood ratio f/g
    contrib = (x > R) * w
    est = contrib.mean()
    se = contrib.std(ddof=1) / np.sqrt(n)
    return est, se
