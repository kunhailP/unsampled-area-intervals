"""Exact observation kernel for two-stage SRSWOR on a binary finite population.

A region has H PSUs of L units each. A population type is the histogram of PSU success
counts (how many PSUs have 0, 1, ..., L ones). Design d = (m, r) samples m PSUs and r units
in each, without replacement. Column j of the kernel is the law of the observed histogram
of per-PSU success counts under population type j.

The mixing law over population types is arbitrary; nothing here assumes a model for it.
"""
from functools import lru_cache
from math import comb

import numpy as np
from scipy.optimize import linprog
from scipy.stats import hypergeom


@lru_cache(None)
def compositions(total, parts):
    if parts == 1:
        return ((total,),)
    return tuple((k,) + tail for k in range(total + 1)
                 for tail in compositions(total - k, parts - 1))


def kernel(m, r, H=6, L=4):
    """Return (A, types): A[o, t] = P(observed histogram o | population type t)."""
    populations = compositions(H, L + 1)
    sampled = compositions(m, L + 1)
    obs = compositions(m, r + 1)
    obs_index = {o: i for i, o in enumerate(obs)}
    b = np.zeros((len(obs), len(sampled)))
    for j, a in enumerate(sampled):
        dist = {(0,) * (r + 1): 1.0}
        for ones, npsu in enumerate(a):
            pp = hypergeom.pmf(np.arange(r + 1), L, ones, r)
            for _ in range(npsu):
                nxt = {}
                for hist, p in dist.items():
                    for x, px in enumerate(pp):
                        if px == 0:
                            continue
                        hh = list(hist); hh[x] += 1; hh = tuple(hh)
                        nxt[hh] = nxt.get(hh, 0.0) + p * px
                dist = nxt
        for hist, p in dist.items():
            b[obs_index[hist], j] = p
    c = np.zeros((len(sampled), len(populations)))
    denom = comb(H, m)
    for i, a in enumerate(sampled):
        for j, t in enumerate(populations):
            if all(x <= y for x, y in zip(a, t)):
                value = 1
                for x, y in zip(a, t):
                    value *= comb(y, x)
                c[i, j] = value / denom
    out = b @ c
    assert np.max(np.abs(out.sum(axis=0) - 1)) < 1e-12
    return out, np.array(populations)


def ambiguity(a, mask):
    """Max over g1, g2 with A g1 = A g2 of g1(mask) - g2(mask) (an LP)."""
    p = a.shape[1]
    ae = np.vstack([np.c_[a, -a], np.r_[np.ones(p), np.zeros(p)],
                    np.r_[np.zeros(p), np.ones(p)]])
    be = np.r_[np.zeros(a.shape[0]), 1, 1]
    opt = linprog(np.r_[-mask.astype(float), mask.astype(float)], A_eq=ae, b_eq=be,
                  bounds=(0, None), method="highs")
    assert opt.success, opt.message
    g1, g2 = opt.x[:p], opt.x[p:]
    err = np.max(np.abs(a @ (g1 - g2)))
    assert err < 1e-7
    return -opt.fun, g1, g2, err
