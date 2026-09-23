"""E01. Exact small-population checks from the start-gate memo (docs/START_GATE_MEMO.md).

1. Two-world counterexample (H=100 PSUs of 10): observed laws identical for m <= 3 PSUs.
   Connected 90% intervals valid in world A alone need expected width >= 12.8 pp;
   valid in both worlds, the exact LP minimum is about 14.01 pp; knowing B, 1.6 pp suffices.
2. H=6, L=4, all 210 population types: kernel ranks, identified moments of theta,
   fixed-interval ambiguity [.4,.6], single vs mixed designs.
3. Continuous-outcome extension numbers.
Writes results/start_gate_results.json.
"""
import json
from fractions import Fraction
from math import comb

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm

from _common import RESULTS
from uai.kernel import ambiguity, kernel


def modular_rank(matrix, prime=1000003):
    """Rank over a prime field: a lower bound on the rational rank."""
    a = np.asarray(matrix, dtype=np.int64).copy() % prime
    row = 0
    for col in range(a.shape[1]):
        piv = np.flatnonzero(a[row:, col])
        if not len(piv):
            continue
        p = row + int(piv[0]); a[[row, p]] = a[[p, row]]
        a[row] = (a[row] * pow(int(a[row, col]), -1, prime)) % prime
        for other in range(row + 1, len(a)):
            a[other] = (a[other] - a[other, col] * a[row]) % prime
        row += 1
        if row == len(a):
            break
    return row


def integer_kernel(m, r):
    a, t = kernel(m, r)
    raw = a * comb(6, m) * comb(4, r)**m
    integer = np.rint(raw).astype(np.int64)
    assert np.max(np.abs(raw - integer)) < 1e-7
    return a, integer, t


def two_world():
    q = Fraction(33, 1666); p = Fraction(400, 833)
    worlds = [{40: Fraction(1, 2), 60: Fraction(1, 2)}, {0: q, 100: q, 49: p, 51: p}]
    ch = lambda n, k: comb(n, k) if 0 <= k <= n else 0
    tv = []
    for m in (1, 2, 3, 4):
        pm = [[sum(w * Fraction(ch(k, j) * ch(100 - k, m - j), comb(100, m)) for k, w in W.items())
               for j in range(m + 1)] for W in worlds]
        d = sum(abs(x - y) for x, y in zip(*pm)) / 2
        assert (d == 0) == (m <= 3)
        tv.append({'m': m, 'TV': str(d)})
    # connected-interval LP on region proportions (pp): A-only and both-world constraints
    A = {42: .5, 58: .5}; B = {10: float(q), 49.2: float(p), 50.8: float(p), 90: float(q)}
    pts = sorted(set(A) | set(B))
    ints = [(a, b) for a in pts for b in pts if a <= b]
    c = [0.0] + [b - a for a, b in ints]
    ca = [0.0] + [sum(v for k, v in A.items() if a <= k <= b) for a, b in ints]
    cb = [0.0] + [sum(v for k, v in B.items() if a <= k <= b) for a, b in ints]
    both = linprog(c, A_ub=[[-v for v in ca], [-v for v in cb]], b_ub=[-.9, -.9],
                   A_eq=[[1] * len(c)], b_eq=[1], bounds=(0, None)).fun
    a_only = linprog(c, A_ub=[[-v for v in ca]], b_ub=[-.9], A_eq=[[1] * len(c)], b_eq=[1],
                     bounds=(0, None)).fun
    return {'TV_by_m': tv, 'min_expected_width_pp_A_only': a_only,
            'min_expected_width_pp_both_worlds': both, 'oracle_B_width_pp': 1.6,
            'oracle_B_coverage': float(2 * p)}


def designs():
    ds = [(2, 4), (4, 2), (3, 2), (2, 3), (6, 1), (6, 3), (5, 4)]
    mats = {d: integer_kernel(*d) for d in ds}
    total = mats[(2, 4)][2] @ np.arange(5); theta = total / 24
    mask = (theta >= .4) & (theta <= .6)
    groups = [(d,) for d in ds] + [((2, 4), (4, 2)), ((3, 2), (2, 3)), ((2, 4), (6, 1)),
                                   ((6, 3), (5, 4))]
    out = []
    for grp in groups:
        a = np.vstack([mats[d][0] for d in grp]); integer = np.vstack([mats[d][1] for d in grp])
        if len(grp) == 1:
            (m, r), = grp; pred = comb(m + r, r)
        else:
            (m, r), (mm, rr) = grp
            pred = comb(m + r, r) + comb(mm + rr, rr) - comb(min(m, mm) + min(r, rr), min(r, rr))
        rank = modular_rank(integer)
        assert rank == pred
        identified = []
        for qq in range(1, 7):
            beta = np.linalg.lstsq(a.T, theta**qq, rcond=None)[0]
            resid = float(np.max(np.abs(a.T @ beta - theta**qq)))
            if resid < 1e-9 and modular_rank(np.vstack([integer, total**qq])) == rank:
                identified.append(qq)
        gap = ambiguity(a, mask)[0]
        out.append({'designs': [list(d) for d in grp], 'rank': rank, 'predicted_rank': pred,
                    'identified_moment_orders': identified,
                    'max_mass_gap_interval_.4_.6': float(gap)})
        print(grp, 'rank', rank, 'moments', identified, 'gap', round(gap, 6))
    return out


def continuous():
    rows = []
    for eps in (0, .001, .002, .005, .01):
        lb = max(.08 - 2 * eps, 0) * .8
        r2 = (.04 / 3) / (.04 / 3 + .25 * .0064 + eps**2 / 12 * (1 + 1 / 1000))
        rows.append({'epsilon': eps, 'lower_bound_A_only': lb, 'known_B_width': .008 + 2 * eps,
                     'R2_of_exact_mean': r2})
    return rows


if __name__ == '__main__':
    res = {'two_world': two_world(), 'design_comparisons': designs(),
           'continuous_extension': continuous()}
    print(json.dumps(res['two_world'], indent=1))
    json.dump(res, open(RESULTS / 'start_gate_results.json', 'w'), indent=1)
