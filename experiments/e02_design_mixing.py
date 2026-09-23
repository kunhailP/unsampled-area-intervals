"""E02. Does mixing two-stage designs across areas shorten the identified 90% interval?

Infinite-sample limit (observed law known exactly). For a design set D, the identified lower
mass of an interval I is  min { g'(I) : A_D g' = A_D g }  (an LP over all 210 population
types, H=6, L=4). Report the shortest I with identified mass >= .90, for three "true" g
(hierarchical beta-binomial with unimodal / bimodal / rare-tail area means; cluster ICC ~ 1/7).
Finding: the mixed design never beats the better single design. Writes
results/design_mixing_identified_width.csv.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from _common import RESULTS
from uai.kernel import kernel

H, L = 6, 4
rng = np.random.default_rng(0)
designs = [(1, 4), (2, 2), (2, 4), (3, 2), (4, 2), (6, 1), (3, 4), (6, 2), (6, 4)]
A = {d: kernel(*d)[0] for d in designs}
T = kernel(1, 1)[1]; theta = T @ np.arange(L + 1) / (H * L)
tidx = {tuple(t): i for i, t in enumerate(T)}; grid = np.unique(theta)


def true_g(kind, n=400000):
    if kind == 'unimodal': mu = rng.beta(8, 8, n)
    elif kind == 'bimodal': mu = np.where(rng.random(n) < .5, rng.beta(20, 40, n), rng.beta(40, 20, n))
    else: mu = np.where(rng.random(n) < .93, rng.beta(20, 20, n), rng.beta(2, 12, n))
    p = rng.beta(mu * 6, (1 - mu) * 6, (H, n)); k = rng.binomial(L, p)
    hist = np.stack([(k == v).sum(0) for v in range(L + 1)], 1)
    g = np.zeros(len(T)); u, c = np.unique(hist, axis=0, return_counts=True)
    for h, cc in zip(u, c): g[tidx[tuple(h)]] += cc
    return g / n


def lower_mass(AD, g, mask):
    p = len(g)
    return linprog(mask.astype(float), A_eq=np.vstack([AD, np.ones(p)]), b_eq=np.r_[AD @ g, 1],
                   bounds=(0, None), method='highs').fun


def shortest(g, AD=None, q=.9):
    pairs = sorted(itertools.combinations_with_replacement(range(len(grid)), 2),
                   key=lambda ab: grid[ab[1]] - grid[ab[0]])
    for a, b in pairs:
        mask = (theta >= grid[a] - 1e-12) & (theta <= grid[b] + 1e-12)
        if (g @ mask if AD is None else lower_mass(AD, g, mask)) >= q - 1e-9:
            return grid[b] - grid[a]


sets = [(d,) for d in designs] + [((2, 4), (4, 2)), ((1, 4), (6, 1)), ((3, 4), (6, 2)),
                                  ((2, 4), (6, 2))]
cost = lambda d: 20 + d[0] * (10 + d[1])
rows = []
for kind in ['unimodal', 'bimodal', 'rare_tail']:
    g = true_g(kind)
    rows.append(dict(g=kind, designs='oracle', cost='', identified_width=shortest(g)))
    for S in sets:
        w = shortest(g, np.vstack([A[d] for d in S]))
        rows.append(dict(g=kind, designs=' + '.join(f'{m}x{r}' for m, r in S),
                         cost='/'.join(str(cost(d)) for d in S), identified_width=w))
        print(kind, S, round(w, 3))
pd.DataFrame(rows).to_csv(RESULTS / 'design_mixing_identified_width.csv', index=False)
