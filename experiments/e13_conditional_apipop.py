"""E13. apipop: conditional reliability of FH, CP and LDC, and widths at equal reliability.

Same pipeline and conditions as E08. Target population of new areas = the 105 districts not
used for training or calibration, so the share of them covered is the exact conditional
coverage given the sampled data D (finite-population view). Rules as in E12 (FH_normal,
FH_boot, FH_PAC; noisy_CP k=100, CP_k102, CP_PAC k=105; LDC_k102, LDC_PAC).
Each replication stores centre and offsets, so every rule can be rescaled afterwards by one
factor per (condition, rule) to reach (a) mean coverage .90 or (b) Pr_D[cov >= .90] = .95.
  python experiments/e13_conditional_apipop.py [reps] [procs]
Writes results/conditional_apipop_summary.csv, results/conditional_apipop_matched.csv.
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from _common import DATA, RESULTS, apipop_replication, load_apipop
from uai.procedures import (ShrinkTable, conformal_threshold, fay_herriot_boot_pac,
                            fay_herriot_reml, noise_lower_bound, pac_rank)

CONDITIONS = [('api99', 3), ('api99', 2), ('socio', 3), ('socio', 2)]
LEV = 0.90


def worker(args):
    fset, n, seed, reps = args
    d, feats = load_apipop(); table = ShrinkTable('0.1'); rng = np.random.default_rng(seed)
    z = stats.norm.ppf(.5 + LEV / 2)
    rows, evals = [], []
    for _ in range(reps):
        V, D, Wc, We = apipop_replication(d, feats[fset], n, rng)
        K = len(V); k0, kp = int(np.ceil((K + 1) * LEV)), pac_rank(K, LEV, .04)
        mu, A, vmu = fay_herriot_reml(V, D); se = np.sqrt(A + vmu)
        piv, lam = fay_herriot_boot_pac(V, D, mu, A, rng, level=LEV)
        ql, qu = np.quantile(piv, [.05, .95])
        D_low = noise_lower_bound(D, K * (n - 1))
        t0, t102, tp = (conformal_threshold(V, k) for k in (k0, 102, kp))
        ints = {'FH_normal': (mu, -z * se, z * se), 'FH_boot': (mu, ql * se, qu * se),
                'FH_PAC': (mu, -lam * se, lam * se),
                'noisy_CP': (0, -t0, t0), 'CP_k102': (0, -t102, t102), 'CP_PAC': (0, -tp, tp)}
        for name, t in (('LDC_k102', t102), ('LDC_PAC', tp)):
            s = t * table(D_low / t**2); ints[name] = (0, -s, s)
        key = f'{seed}_{len(evals)}'; evals.append((key, We))
        for m, (c, lo, hi) in ints.items():
            rows.append(dict(condition=f'{fset}_n{n}', key=key, method=m, centre=c, lo=lo, hi=hi,
                             cov=np.mean((We >= c + lo) & (We <= c + hi))))
    return rows, evals


def matched(g, W):
    c, lo, hi = g.centre.values, g.lo.values, g.hi.values
    Ws = [W[k] for k in g.key]
    cov = lambda l: np.array([np.mean((w >= ci + l * a) & (w <= ci + l * b))
                              for w, ci, a, b in zip(Ws, c, lo, hi)])
    la = brentq(lambda l: cov(l).mean() - LEV, 1e-2, 10, xtol=1e-5)
    lb = brentq(lambda l: np.mean(cov(l) >= LEV) - .95 + 1e-9, 1e-2, 10, xtol=1e-5)
    ca = cov(la)
    return dict(lam_mean=la, width_mean_matched=np.mean(la * (hi - lo)), sd_cov_mean_matched=ca.std(),
                share_below_mean_matched=np.mean(ca < LEV),
                lam_pac=lb, width_pac_matched=np.mean(lb * (hi - lo)))


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    chunks = 25
    jobs = [(f, n, 9000 + 100 * i + j, reps // chunks) for i, (f, n) in enumerate(CONDITIONS) for j in range(chunks)]
    with Pool(procs) as pool:
        out = pool.map(worker, jobs)
    r = pd.DataFrame([row for rows, _ in out for row in rows])
    W = {k: w for _, ev in out for k, w in ev}
    r.to_pickle(DATA / 'conditional_apipop_reps.pkl')
    ne = len(next(iter(W.values())))
    summ, mat = [], []
    for (cond, m), g in r.groupby(['condition', 'method']):
        cv = g['cov']
        # superpopulation view: remove binomial evaluation noise from the spread (no fpc)
        sd_super = np.sqrt(max(cv.var() - np.mean(cv * (1 - cv)) / ne, 0))
        summ.append(dict(condition=cond, method=m, reps=len(g), mean_cov=cv.mean(), sd_cov=cv.std(),
                         sd_cov_minus_eval_noise=sd_super, pr_cov_ge_90=np.mean(cv >= LEV),
                         q05_cov=cv.quantile(.05), width=(g.hi - g.lo).mean()))
        mat.append(dict(condition=cond, method=m, **matched(g, W)))
    S = pd.DataFrame(summ); M = pd.DataFrame(mat)
    S.to_csv(RESULTS / 'conditional_apipop_summary.csv', index=False)
    M.to_csv(RESULTS / 'conditional_apipop_matched.csv', index=False)
    pd.set_option('display.width', 200)
    print(S.round(3).to_string(index=False)); print(M.round(3).to_string(index=False))
