"""E06. Synthetic Monte Carlo at the apipop-like regime (Var W = 1, D = 0.577, K = 110).

Latent-score shapes: log-concave (normal, Laplace, uniform, truncated-Laplace extremal,
gamma(2), shifted normal) and not log-concave (bimodal, t3). Rules: noisy CP (k = 100),
Gaussian plug-in shrink t*sqrt(A/(A+D)), LDC (k = 100, conservative table lookup).
Writes results/synthetic_mc.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats

from _common import RESULTS
from uai.procedures import ShrinkTable

rng = np.random.default_rng(7)
table = ShrinkTable('0.1')


def draw(kind, n):
    if kind == 'normal': return rng.normal(size=n)
    if kind == 'laplace': return rng.laplace(size=n) / np.sqrt(2)
    if kind == 'uniform': return rng.uniform(-np.sqrt(3), np.sqrt(3), n)
    if kind == 'trunc_laplace':
        c = 2.705; u = rng.uniform(size=n)
        w = -np.log(1 - u * (1 - np.exp(-c))) * rng.choice([-1, 1], n)
        return w / np.sqrt((2 - np.exp(-c) * (c * c + 2 * c + 2)) / (1 - np.exp(-c)))
    if kind == 'gamma2_skew': return (rng.gamma(2, size=n) - 2) / np.sqrt(2)
    if kind == 'shifted_normal': return rng.normal(size=n) + .3
    if kind == 'bimodal_not_LC':
        return (rng.choice([-1, 1], n) * 1.2 + rng.normal(0, .3, n)) / np.sqrt(1.44 + .09)
    if kind == 't3_not_LC': return rng.standard_t(3, n) / np.sqrt(3)


D, K, reps, alpha, nu_pool = .577, 110, 2000, .10, 110 * 10
rows = []
for kind in ['normal', 'laplace', 'uniform', 'trunc_laplace', 'gamma2_skew', 'shifted_normal',
             'bimodal_not_LC', 't3_not_LC']:
    big = np.abs(draw(kind, 400000)); q_or = np.quantile(big, 1 - alpha)
    res = {m: [] for m in ['noisy_CP', 'gauss_plugin', 'LDC']}
    for _ in range(reps):
        W = draw(kind, K); V = W + rng.normal(0, np.sqrt(D), K)
        t = np.sort(np.abs(V))[int(np.ceil((K + 1) * (1 - alpha))) - 1]
        Dhat = D * rng.chisquare(nu_pool) / nu_pool
        D_low = Dhat * nu_pool / stats.chi2.ppf(.99, nu_pool)
        Ahat = max(np.mean(V**2) - Dhat, 1e-9)
        for m, s in {'noisy_CP': t, 'gauss_plugin': t * np.sqrt(Ahat / (Ahat + Dhat)),
                     'LDC': t * table(D_low / t**2)}.items():
            res[m].append((np.mean(big <= s), s))
    for m, v in res.items():
        v = np.array(v)
        rows.append(dict(shape=kind, method=m, coverage=v[:, 0].mean(), width_over_oracle=v[:, 1].mean() / q_or))
    print(kind, {m: (round(100 * np.mean([a for a, _ in v]), 1)) for m, v in res.items()})
pd.DataFrame(rows).to_csv(RESULTS / 'synthetic_mc.csv', index=False)
