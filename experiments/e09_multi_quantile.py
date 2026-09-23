"""E09. Ceiling check: does using several quantiles of |V| shrink much further?

V Gaussian with P(|V| <= 1) = .9, x = 0.136 (apipop level). Maximise the 90% quantile of |W|
over log-concave W (k-piece log-affine) subject to |P(|W+e| <= tau_j) - p_j| <= slack.
Finding: one exact quantile 0.874, two 0.859, three 0.849 (Gaussian 0.795); any realistic
slack (0.02) is worse than one exact quantile. Little room at K ~ 110.
Writes results/multi_quantile_ceiling.json.
"""
import json
from multiprocessing import Pool

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import differential_evolution
from scipy.stats import norm

from _common import RESULTS

gy, gw = leggauss(200); gy = (gy + 1) / 2; gw = gw / 2
CASES = [(0.136, 0, [.9], 1), (0.136, 0, [.5, .9], 2), (0.136, 0, [.5, .75, .9], 3),
         (0.136, .02, [.5, .75, .9], 3), (0.136, .05, [.5, .75, .9], 3)]


def solve(case):
    x, slack, levels, k = case
    sV = 1 / norm.ppf(.95); taus = [sV * norm.ppf((1 + p) / 2) for p in levels]

    def dens(p):
        fr = np.sort(p[:k - 1]); s = -np.sort(-p[k - 1:2 * k - 1])
        edges = np.r_[0, fr, 1]; seg = np.clip(np.searchsorted(edges, gy, side='right') - 1, 0, k - 1)
        lv = np.r_[0, np.cumsum(s * np.diff(edges))]; lf = lv[seg] + s[seg] * (gy - edges[seg])
        w = np.exp(lf - lf.max()) * gw; return w / w.sum()

    def obj(p):
        w = dens(p); loc, sc = p[-2], np.exp(p[-1]); xw = loc + sc * (gy - .5); sd = np.sqrt(x)
        pen = sum(max(0, abs((w * (norm.cdf((tau - xw) / sd) - norm.cdf((-tau - xw) / sd))).sum() - lev) - slack)
                  for tau, lev in zip(taus, levels))
        d = np.abs(xw); o = np.argsort(d); c = np.cumsum(w[o])
        return -d[o][min(np.searchsorted(c, .9), len(c) - 1)] + 200 * pen
    bounds = [(0, 1)] * (k - 1) + [(-60, 60)] * k + [(-1.5, 1.5), (np.log(1e-3), np.log(5))]
    best = 0
    for seed in range(4):
        r = differential_evolution(obj, bounds, seed=seed, maxiter=2000, popsize=40, tol=1e-12, polish=True)
        if r.fun < 0: best = max(best, -r.fun)
    return dict(x=x, slack=slack, levels=levels, pieces=k, r=best, gaussian=float(np.sqrt(1 - norm.ppf(.95)**2 * x)))


if __name__ == '__main__':
    with Pool(len(CASES)) as pool:
        out = pool.map(solve, CASES)
    json.dump(out, open(RESULTS / 'multi_quantile_ceiling.json', 'w'), indent=1)
    for o in out: print(o)
