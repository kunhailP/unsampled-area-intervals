"""E05. Numerical lower bound on marginal latent coverage under the log-concave class.

Split conformal: t = k-th smallest of K calibration |V|; then P(|V| <= t) ~ Beta(k, K+1-k)
exactly. Given P(|V| <= t) = p, the latent coverage of s = t * rho is >= H(p) =
inf_{log-concave W} P(|W| <= rho) s.t. P(|W+e| <= 1) >= p  (t normalised to 1).
Bound = (1 - eta) * E_Beta[ min_x H(P; x) ]  (eta: failure prob. of the noise bound; LDC only).
H comes from a penalised DE search (an upper estimate of an infimum) on four x values only,
so this is the value of the current bounding argument, not a certified or attained worst case.

Rules: noisy CP (rho = 1) and LDC (rho = r_.10(x)).
Finding (K = 110): the bound is 0.867 for noisy CP at k = 100; at k = 102 it is 0.916 (noisy CP)
and 0.905 (LDC, after the min/mean order fix). These are candidate ranks under this argument.
  python experiments/e05_exact_coverage.py compute [procs]   # H grid -> results/h_table.json
  python experiments/e05_exact_coverage.py                   # integrate -> results/exact_coverage.csv
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import beta

from _common import RESULTS
from uai.logconcave import worst_latent_coverage

XS = ['0.061', '0.101', '0.136', '0.201']


def job(a):
    x, p, s0, rule = a
    return dict(rule=rule, x=float(x), p=p, s0=s0, H=worst_latent_coverage(float(x), p, s0))


def compute(procs):
    T = json.load(open(RESULTS / 'r_table.json'))
    ps = sorted(set([.80, .82, .84, .86, .87, .88, .89, .90, .91, .92, .93, .94, .95, .96, .97,
                     .98, .99, .995] + list(np.round(np.arange(.85, .9601, .005), 3))))
    jobs = [(x, p, T['0.1'][x], 'ldc_a0.10') for x in XS for p in ps]
    jobs += [(x, p, 1.0, 'noisy') for x in XS for p in ps if .85 <= p <= .96]
    with Pool(procs) as pool:
        rows = pool.map(job, jobs)
    json.dump(rows, open(RESULTS / 'h_table.json', 'w'), indent=1)


def integrate(K=110, eta=.01):
    rows = json.load(open(RESULTS / 'h_table.json'))
    H = {}
    for r in rows:
        H.setdefault((r['rule'], r['x']), {})[r['p']] = r['H']

    def Hfun(rule, x, p):                    # step down to the grid point <= p (conservative)
        g = H[(rule, x)]; ps = np.array(sorted(g)); hs = np.maximum.accumulate([g[q] for q in ps])
        out = np.zeros_like(p)
        for i, q in enumerate(ps):
            out = np.where(p >= q, hs[i], out)
        return out
    u = np.linspace(0, 1, 400001)[1:-1]
    xs = sorted({x for (_, x) in H})
    out = []
    for rule in ['noisy', 'ldc_a0.10']:
        for k in range(98, 108):
            w = beta.pdf(u, k, K + 1 - k); w /= w.sum()
            Hx = np.array([Hfun(rule, x, u) for x in xs])
            per_x = {x: float((w * h).sum()) for x, h in zip(xs, Hx)}
            # x = D / T^2 moves with T, so the pointwise bound is E_U[min_x H], not min_x E_U[H]
            g = float((w * Hx.min(0)).sum()) * ((1 - eta) if rule != 'noisy' else 1)
            out.append(dict(rule=rule, K=K, k=k, guarantee=g, **{f'x={x}': v for x, v in per_x.items()}))
    df = pd.DataFrame(out); df.to_csv(RESULTS / 'exact_coverage.csv', index=False)
    print(df[['rule', 'k', 'guarantee']].to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'compute':
        compute(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    integrate()
