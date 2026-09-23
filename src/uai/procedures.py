"""Interval rules for an unsampled area, all centred at a separately trained predictor.

Scores on calibration areas: V_c = (survey estimate) - (centre), with design variance D_c.
Target on a new area: W = theta - centre.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
_FEASIBLE_X = 0.3696          # above this, P(|e| <= t) < .9: data contradict the noise level


class ShrinkTable:
    """Conservative lookup of r_alpha(x) from results/r_table.json.

    r is nonincreasing in x on the table range (checked on the grid, not proved), so the
    value at the grid point at or below x is >= r(x) there. Below the first grid point the
    first table value is returned, not 1: for asymmetric log-concave W, r can exceed 1 at
    small x (explicit example in tests/test_core.py). All table values are numerical search
    results, not certified upper bounds; see docs/FINDINGS.md C10, C25.
    """

    def __init__(self, alpha='0.1', path=ROOT / 'results' / 'r_table.json'):
        tab = json.load(open(path))[alpha]
        self.x = np.array(sorted(float(k) for k in tab))
        self.r = np.array([tab[str(k)] for k in self.x])

    def __call__(self, x):
        if x >= _FEASIBLE_X:        # alpha = .10 only; the data contradict the noise level
            return 1.0
        if x < self.x[0]:
            return float(self.r[0])
        return float(self.r[np.searchsorted(self.x, x, side='right') - 1])


def conformal_threshold(scores, k):
    """k-th smallest |score| (1-based)."""
    return np.sort(np.abs(scores))[k - 1]


def noise_lower_bound(D_hat, df_total, eta=0.01):
    """Pooled chi-square lower confidence bound for a common noise variance."""
    return np.mean(D_hat) * df_total / stats.chi2.ppf(1 - eta, df_total)


def ldc_halfwidth(scores, D_low, k, table):
    t = conformal_threshold(scores, k)
    return t * table(D_low / t**2)


def fay_herriot_reml(V, D):
    """Intercept-only Fay-Herriot fit by REML with D treated as known. Returns (mu, A, var_mu)."""
    def nll(logA):
        A = np.exp(logA); w = 1 / (A + D); mu = (w * V).sum() / w.sum()
        return 0.5 * (np.log(A + D).sum() + np.log(w.sum()) + (w * (V - mu)**2).sum())
    r = minimize_scalar(nll, bounds=(np.log(1e-6 * np.var(V) + 1e-12),
                                     np.log(10 * np.var(V) + 1e-9)), method='bounded')
    A = np.exp(r.x); w = 1 / (A + D)
    return (w * V).sum() / w.sum(), A, 1 / w.sum()


def fay_herriot_boot_pivot(V, D, mu, A, rng, B=300):
    """Parametric-bootstrap draws of (theta* - mu*) / sqrt(A* + v*) for a new area."""
    piv = np.empty(B)
    for b in range(B):
        Vb = mu + rng.normal(0, np.sqrt(A), len(V)) + rng.normal(0, np.sqrt(D))
        thb = mu + rng.normal(0, np.sqrt(A))
        mub, Ab, vb = fay_herriot_reml(Vb, D)
        piv[b] = (thb - mub) / np.sqrt(Ab + vb)
    return piv


def pac_rank(K, level=0.90, delta=0.05):
    """Smallest k with P(Beta(k, K+1-k) >= level) >= 1 - delta: the k-th order statistic of
    K exchangeable scores then has conditional coverage >= level with probability >= 1 - delta."""
    for k in range(1, K + 1):
        if stats.beta.sf(level, k, K + 1 - k) >= 1 - delta:
            return k
    return None


def fay_herriot_boot_pac(V, D, mu, A, rng, B=300, level=0.90, delta=0.05):
    """One parametric bootstrap giving (i) pivot draws as in fay_herriot_boot_pivot and
    (ii) a tolerance multiplier lam: the (1 - delta) bootstrap quantile of the multiplier each
    refit needs for its interval mu* +- lam sqrt(A* + v*) to cover N(mu, A) with prob. level."""
    piv, lam = np.empty(B), np.empty(B)
    sA = np.sqrt(max(A, 1e-12))
    for b in range(B):
        Vb = mu + rng.normal(0, np.sqrt(A), len(V)) + rng.normal(0, np.sqrt(D))
        thb = mu + rng.normal(0, np.sqrt(A))
        mub, Ab, vb = fay_herriot_reml(Vb, D)
        se = np.sqrt(Ab + vb)
        piv[b] = (thb - mub) / se
        cov = lambda l: (stats.norm.cdf((mub + l * se - mu) / sA)
                         - stats.norm.cdf((mub - l * se - mu) / sA) - level)
        hi = 1.0
        while cov(hi) < 0 and hi < 1e4:
            hi *= 2
        lam[b] = brentq(cov, 0, hi) if hi < 1e4 else hi
    return piv, np.quantile(lam, 1 - delta)
