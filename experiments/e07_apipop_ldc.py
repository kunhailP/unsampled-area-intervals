"""E07. apipop finite population: LDC vs noisy CP at an equal log-concave guarantee.

Areas = districts with >= 5 schools (325); split 110 train / 110 calib / 105 eval per
replication; n schools SRSWOR per sampled district; Ridge unit model; centre = district
population mean of h. Target = district mean api00 minus centre.
Rules: oracle (calib W known), noisy CP (k=100), Gaussian plug-in, noisy CP (k=102),
LDC (k=102). k=102 is what both rules need for a 90% guarantee under log-concavity (E05).
  python experiments/e07_apipop_ldc.py [reps] [procs]
Writes results/apipop_ldc_k102_summary.csv (and a per-replication pickle in data/).
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import DATA, RESULTS, apipop_replication, load_apipop
from uai.procedures import ShrinkTable, conformal_threshold, noise_lower_bound

CONDITIONS = [('api99', 3), ('api99', 4), ('socio', 3)]


def worker(args):
    fset, n, seed, reps = args
    d, feats = load_apipop(); table = ShrinkTable('0.1'); rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        V, D, Wc, We = apipop_replication(d, feats[fset], n, rng)
        K = len(V); k0 = int(np.ceil((K + 1) * .9))
        t = conformal_threshold(V, k0); t102 = conformal_threshold(V, 102)
        D_low = noise_lower_bound(D, K * (n - 1)); Dbar = D.mean(); Ahat = max(np.mean(V**2) - Dbar, 1e-9)
        s = dict(oracle=conformal_threshold(Wc, k0), noisy_CP=t,
                 gauss_plugin=t * np.sqrt(Ahat / (Ahat + Dbar)), noisy_CP_k102=t102,
                 LDC_k102=t102 * table(D_low / t102**2))
        for m, h in s.items():
            rows.append(dict(condition=f'{fset}_n{n}', method=m, cov=np.mean(np.abs(We) <= h),
                             width=2 * h, DA=Dbar / max(np.mean(Wc**2), 1e-9)))
    return rows


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    chunks = 10
    jobs = [(f, n, 1000 * i + j, reps // chunks) for i, (f, n) in enumerate(CONDITIONS) for j in range(chunks)]
    with Pool(procs) as pool:
        r = pd.DataFrame([row for part in pool.map(worker, jobs) for row in part])
    r.to_pickle(DATA / 'apipop_ldc_reps.pkl')
    out = []
    for (cond, m), g in r.groupby(['condition', 'method']):
        n = len(g)
        out.append(dict(condition=cond, method=m, reps=n, coverage=g['cov'].mean(),
                        mcse=g['cov'].std() / np.sqrt(n), width=g['width'].mean(), DA=g['DA'].mean()))
    s = pd.DataFrame(out); s.to_csv(RESULTS / 'apipop_ldc_k102_summary.csv', index=False)
    print(s.to_string(index=False))
