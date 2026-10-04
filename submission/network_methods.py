"""Correlation-MST hub stability; no market data are embedded in this module."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def correlation(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 2:
        raise ValueError("Need a complete observations-by-assets matrix")
    if not np.isfinite(x).all():
        raise ValueError("Missing values require an explicit panel policy")
    z = x - x.mean(axis=0)
    norm = np.sqrt(np.sum(z * z, axis=0))
    if np.any(norm <= 0):
        raise ValueError("Constant series cannot enter a correlation network")
    r = (z.T @ z) / np.outer(norm, norm)
    r = np.clip((r + r.T) / 2, -1, 1)
    np.fill_diagonal(r, 1)
    return r


def transform_returns(x: np.ndarray, kind: str, factors: np.ndarray | None = None) -> np.ndarray:
    if kind == "raw":
        return np.array(x, copy=True)
    if kind == "demeaned":
        return x - x.mean(axis=1, keepdims=True)
    if kind == "factor_residual":
        if factors is None:
            raise ValueError("Factor residuals require factors observed over this window")
        f = np.asarray(factors)
        if f.ndim == 1:
            f = f[:, None]
        if f.shape[0] != x.shape[0] or not np.isfinite(f).all():
            raise ValueError("Factors must share the return window")
        if f.ndim == 3:
            if f.shape[1] != x.shape[1]:
                raise ValueError("Asset-specific factor tensors need the same asset dimension")
            residual = np.empty_like(x)
            groups = {}
            for asset in range(x.shape[1]):
                key = np.ascontiguousarray(f[:,asset,:]).tobytes()
                groups.setdefault(key, []).append(asset)
            for assets in groups.values():
                design = np.column_stack([np.ones(len(x)), f[:,assets[0],:]])
                if np.linalg.matrix_rank(design) != design.shape[1]:
                    raise ValueError("Asset-specific factor design is rank deficient")
                coef = np.linalg.lstsq(design,x[:,assets],rcond=None)[0]
                residual[:,assets] = x[:,assets]-design@coef
            return residual
        if f.ndim != 2:
            raise ValueError("Factors must be a matrix or an asset-specific tensor")
        design = np.column_stack([np.ones(x.shape[0]), f])
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError("Factor design is rank deficient")
        coef = np.linalg.lstsq(design, x, rcond=None)[0]
        return x - design @ coef
    raise ValueError(f"Unknown return transform: {kind}")


def mst(r: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """MST from an order-equivalent cost; random relabeling avoids fixed-label ties.

    Cost 3-rho is strictly decreasing in rho, as is sqrt(2*(1-rho)).
    Kruskal ordering, hence the set of admissible MSTs, is identical. Positive
    off-diagonal costs avoid SciPy's treating exact zero distance as no edge.
    Random relabeling is a tie convention, not uniform sampling of tied trees.
    """
    n = r.shape[0]
    if r.shape != (n, n) or not np.allclose(r, r.T):
        raise ValueError("Correlation input must be square and symmetric")
    perm = np.arange(n) if rng is None else rng.permutation(n)
    cost = 3 - r[np.ix_(perm, perm)]
    np.fill_diagonal(cost, 0)
    tree = minimum_spanning_tree(cost)
    row, col = tree.nonzero()
    edges = np.sort(np.column_stack([perm[row], perm[col]]), axis=1)
    if len(edges) != n - 1:
        raise AssertionError("A spanning tree must contain N-1 edges")
    degree = np.bincount(edges.ravel(), minlength=n)
    if degree.sum() != 2 * (n - 1):
        raise AssertionError("MST degree sum is incorrect")
    return edges, degree


def fractional_top_k(degree: np.ndarray, k: int) -> np.ndarray:
    """Assign total membership k, sharing boundary ties fractionally."""
    if not 1 <= k < len(degree):
        raise ValueError("Need 1 <= k < N")
    threshold = np.sort(degree)[-k]
    above = degree > threshold
    tied = degree == threshold
    out = above.astype(float)
    out[tied] = (k - above.sum()) / tied.sum()
    return out


def circular_block_indices(t: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if not 1 <= block <= t:
        raise ValueError("Block length must be within the sample window")
    starts = rng.integers(t, size=int(np.ceil(t / block)))
    return ((starts[:, None] + np.arange(block)) % t).ravel()[:t]


@dataclass
class HubResult:
    edges: np.ndarray
    degree: np.ndarray
    selection: np.ndarray
    edge_frequency: np.ndarray
    degree_concentration: float
    reliability_share: float
    k: int


def hub_stability(x: np.ndarray, factors: np.ndarray | None, kind: str,
                  repetitions: int, block: int, seed: int,
                  fraction: float = .10) -> HubResult:
    if repetitions < 2:
        raise ValueError("Need at least two bootstrap replications")
    rng = np.random.default_rng(seed)
    n = x.shape[1]
    k = int(np.ceil(fraction * n))
    edges, degree = mst(correlation(transform_returns(x, kind, factors)), rng)
    p = np.zeros(n)
    edge_count = np.zeros((n, n))
    for _ in range(repetitions):
        idx = circular_block_indices(len(x), block, rng)
        # One day/block draw for all assets AND factor observations; refit OLS.
        fb = None if factors is None else factors[idx]
        rb = correlation(transform_returns(x[idx], kind, fb))
        eb, db = mst(rb, rng)
        p += fractional_top_k(db, k)
        edge_count[eb[:, 0], eb[:, 1]] += 1
    p /= repetitions
    if not np.isclose(p.sum(), k):
        raise AssertionError("Selection membership must sum to k")
    denom = 2 * (n - 1)
    return HubResult(edges, degree, p, edge_count / repetitions,
                     float(fractional_top_k(degree, k) @ degree / denom),
                     float(p @ degree / denom), k)


def diversification_loss(simple_returns: np.ndarray, weights: np.ndarray) -> float:
    """Inverse squared diversification ratio for fixed nonnegative weights."""
    if simple_returns.ndim != 2 or len(weights) != simple_returns.shape[1]:
        raise ValueError("Invalid future return panel or weights")
    if not np.isfinite(simple_returns).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
        raise ValueError("Need complete returns and long-only normalized weights")
    sigma = simple_returns.std(axis=0, ddof=1)
    denom = float(weights @ sigma) ** 2
    if denom <= 0:
        raise ValueError("Zero volatility denominator")
    value = float(np.var(simple_returns @ weights, ddof=1) / denom)
    if value > 1 + 1e-10 or value < -1e-10:
        raise AssertionError("PSD covariance and long-only weights imply 0 <= Y <= 1")
    return float(np.clip(value, 0, 1))


def matured_training_rows(origin_positions: np.ndarray, forecast_origin: int, horizon: int) -> np.ndarray:
    """A label ending at forecast_origin is known after that session closes."""
    return np.flatnonzero(origin_positions + horizon <= forecast_origin)


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    sigma = np.sqrt(np.diag(cov))
    if np.any(sigma <= 0):
        raise ValueError("Population variances must be positive")
    return np.clip(cov / np.outer(sigma, sigma), -1, 1)


def demean_population_covariance(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    projection = np.eye(n) - np.ones((n, n)) / n
    return projection @ cov @ projection
