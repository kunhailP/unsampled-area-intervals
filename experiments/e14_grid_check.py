"""E14. Independent dense-grid check of the DE shrink table r_alpha(x) (E04).

Localization (Fradelizi-Guedon, one continuous constraint, support truncated to [-M, M] and
M -> infinity) reduces the sup to Dirac masses and W = loc + sc*Y, Y on [0,1] with density
prop. to exp(beta y). Here that 3-parameter family is scanned on a grid (no root finding, no
optimizer): beta in +-[0, 120] (sinh spacing, 241), loc in [-3, 1.5] (181), log sc in
[1e-4, 6] (181). Among grid points with P(|W+e| <= 1) >= 1 - alpha the largest (1-alpha)-
quantile of |W| is r_grid(x). A grid can also miss the sup, so this is a cross-check of two
independent searches, not a certificate.
  python experiments/e14_grid_check.py [procs] [step=5] [alphas=0.1,0.07]
Writes results/r_grid_check.csv (x, alpha, r_table, r_grid, diff).
"""
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

from _common import RESULTS

GY, GW = leggauss(200); GY = (GY + 1) / 2; GW = GW / 2
BETAS = np.sinh(np.linspace(-np.arcsinh(120 / 2), np.arcsinh(120 / 2), 241)) * 2
LOCS = np.linspace(-3, 1.5, 181)
SCS = np.exp(np.linspace(np.log(1e-4), np.log(6), 181))


def cdf_y(beta, y):
    y = np.clip(y, 0, 1)
    if abs(beta) < 1e-8:
        return y
    if beta > 0:
        return np.exp(beta * (y - 1)) * (-np.expm1(-beta * y)) / (-np.expm1(-beta))
    return np.expm1(beta * y) / np.expm1(beta)


def job(args):
    x, alpha = args
    lev, s = 1 - alpha, np.sqrt(x)
    L, S = np.meshgrid(LOCS, SCS, indexing='ij')
    best = 0.0
    for b in BETAS:
        lw = b * GY; w = np.exp(lw - lw.max()) * GW; w /= w.sum()
        Wn = L[..., None] + S[..., None] * GY                          # (loc, sc, node)
        m = ((ndtr((1 - Wn) / s) - ndtr((-1 - Wn) / s)) * w).sum(-1)
        ok = m >= lev
        if not ok.any():
            continue
        l, sc = L[ok], S[ok]
        lo, hi = np.zeros_like(l), np.abs(l) + sc
        for _ in range(50):                                            # quantile of |W|
            mid = (lo + hi) / 2
            mass = cdf_y(b, (mid - l) / sc) - cdf_y(b, (-mid - l) / sc)
            lo, hi = np.where(mass < lev, mid, lo), np.where(mass < lev, hi, mid)
        best = max(best, hi.max())
    locs = np.linspace(0, 3, 300001)                                   # Dirac masses
    okd = ndtr((1 - locs) / s) - ndtr((-1 - locs) / s) >= lev
    if okd.any():
        best = max(best, locs[okd].max())
    return x, alpha, best


if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tab = json.load(open(RESULTS / 'r_table.json'))
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 5         # every step-th table point
    alphas = sys.argv[3].split(',') if len(sys.argv) > 3 else ['0.1', '0.07']
    jobs = [(float(x), float(a)) for a in alphas for x in sorted(tab[a], key=float)[::step]]
    with Pool(procs) as pool:
        res = pool.map(job, jobs)
    df = pd.DataFrame(res, columns=['x', 'alpha', 'r_grid'])
    df['r_table'] = [tab[str(a)][str(x)] if str(x) in tab[str(a)] else tab[str(a)][f'{x:g}']
                     for x, a in zip(df.x, df.alpha)]
    df['diff'] = df.r_grid - df.r_table
    df.to_csv(RESULTS / 'r_grid_check.csv', index=False)
    print(df.groupby('alpha')['diff'].describe().to_string())
    print(df.loc[df['diff'].abs().nlargest(8).index].to_string(index=False))
