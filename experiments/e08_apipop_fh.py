"""E08. apipop: do standard Fay-Herriot intervals for an unsampled district fail?

Same pipeline as E07. Area model on calibration scores V_c = mu + u_c + e_c, u ~ N(0, A),
e_c ~ N(0, D_c) with D_c treated as known (standard practice).
  FH_normal : mu_hat +- z sqrt(A_hat + v(mu_hat))   (REML)
  FH_boot   : parametric-bootstrap pivot, B = 300
  noisy_CP  : split conformal on |V|
  LDC_cert  : k = 102, 90% level only
Finding: FH meets nominal coverage in all four conditions and is ~15-17% narrower than LDC.
  python experiments/e08_apipop_fh.py [reps] [procs]
Writes results/apipop_fh_summary.csv.
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from _common import DATA, RESULTS, apipop_replication, load_apipop
from uai.procedures import (ShrinkTable, conformal_threshold, fay_herriot_boot_pivot,
                            fay_herriot_reml, noise_lower_bound)

CONDITIONS = [('api99', 3), ('api99', 2), ('socio', 3), ('socio', 2)]


def worker(args):
    fset, n, seed, reps = args
    d, feats = load_apipop(); table = ShrinkTable('0.1'); rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        V, D, Wc, We = apipop_replication(d, feats[fset], n, rng)
        K = len(V); mu, A, vmu = fay_herriot_reml(V, D); se = np.sqrt(A + vmu)
        piv = fay_herriot_boot_pivot(V, D, mu, A, rng)
        t102 = conformal_threshold(V, 102); D_low = noise_lower_bound(D, K * (n - 1))
        for lev in (.90, .95):
            z = stats.norm.ppf(.5 + lev / 2); ql, qu = np.quantile(piv, [(1 - lev) / 2, (1 + lev) / 2])
            tn = conformal_threshold(V, min(int(np.ceil((K + 1) * lev)), K))
            ints = {'FH_normal': (mu - z * se, mu + z * se), 'FH_boot': (mu + ql * se, mu + qu * se),
                    'noisy_CP': (-tn, tn)}
            if lev == .90:
                s = t102 * table(D_low / t102**2); ints['LDC_cert'] = (-s, s)
            for m, (lo, hi) in ints.items():
                c = np.mean((We >= lo) & (We <= hi))
                rows.append(dict(condition=f'{fset}_n{n}', level=lev, method=m, cov=c,
                                 width=hi - lo, DA=D.mean() / max(A, 1e-9)))
    return rows


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    chunks = 10
    jobs = [(f, n, 5000 + 100 * i + j, reps // chunks) for i, (f, n) in enumerate(CONDITIONS) for j in range(chunks)]
    with Pool(procs) as pool:
        r = pd.DataFrame([row for part in pool.map(worker, jobs) for row in part])
    r.to_pickle(DATA / 'apipop_fh_reps.pkl')
    out = []
    for (cond, lev, m), g in r.groupby(['condition', 'level', 'method']):
        n = len(g)
        out.append(dict(condition=cond, level=lev, method=m, reps=n, coverage=g['cov'].mean(),
                        mcse=g['cov'].std() / np.sqrt(n),
                        share_rep_cov_below_lev_minus_5pp=np.mean(g['cov'] < lev - .05),
                        width=g['width'].mean(), DA_median=g['DA'].median()))
    s = pd.DataFrame(out); s.to_csv(RESULTS / 'apipop_fh_summary.csv', index=False)
    print(s.to_string(index=False))
