"""Interval rules for an unsampled area, all centred at a separately trained predictor.

Scores on calibration areas: V_c = (survey estimate) - (centre), with design variance D_c.
Target on a new area: W = theta - centre.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
_FEASIBLE_X = 0.3696          # above this, P(|e| <= t) < .9: data contradict the noise level


class ShrinkTable:
    """Conservative lookup of r_alpha(x) from results/r_table.json.

    r is nonincreasing in x (checked on the grid), so the value at the grid point at or
    below x is >= r(x): the resulting interval is at least as wide as the exact rule.
    """

    def __init__(self, alpha='0.1', path=ROOT / 'results' / 'r_table.json'):
        tab = json.load(open(path))[alpha]
        self.x = np.array(sorted(float(k) for k in tab))
        self.r = np.array([tab[str(k)] for k in self.x])

    def __call__(self, x):
        if x >= _FEASIBLE_X or x < self.x[0]:
            return 1.0
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
