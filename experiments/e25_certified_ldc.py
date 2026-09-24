"""E25. LDC_PAC with certified shrink values (E24) in the E12 setting.

Three lookups for the same threshold T = |V|_(k), k = pac_rank(110, .9, .04) = 105, and the
same pooled noise lower bound D_low (eta = .01):
  LDC_PAC        old table (converged grid maxima of R_{.9,.9}, step lookup; not certified)
  LDC_cert       certified upper bound of R_{.9,.9}
  LDC_cert_pk    certified upper bound of R_{.9036,.9}; valid because on the order-statistic
                 event the noisy mass at T is >= p_k = .90361 >= .9036 (Beta law)
Conditional coverage from closed-form CDFs. Same target Pr_D[cov >= .90] >= .95.
  python experiments/e25_certified_ldc.py [reps] [procs]
Writes results/certified_ldc.csv and results/certified_ldc_summary.csv.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import RESULTS
from e12_conditional_synth import DBAR, K, LEV, NU, REGIMES, SHAPES, draw
from uai.latent_laws import cdf
from uai.procedures import (CertifiedShrinkTable, ShrinkTable, conformal_threshold,
                            noise_lower_bound, pac_rank)

warnings.filterwarnings('ignore')


def one(args):
    shape, regime, seed, reps = args
    rng = np.random.default_rng(seed)
    tabs = {'LDC_PAC': ShrinkTable('0.1'), 'LDC_cert': CertifiedShrinkTable(0.9),
            'LDC_cert_pk': CertifiedShrinkTable(0.9036)}
    kp = pac_rank(K, LEV, .04)
    rows = []
    for rep in range(reps):
        if regime == 'common':
            Dc = np.full(K, DBAR)
        else:
            w = rng.lognormal(0, .7, K); Dc = DBAR * w / np.exp(.7**2 / 2)
        V = draw(shape, K, rng) + rng.normal(0, np.sqrt(Dc))
        Dh = Dc * rng.chisquare(NU, K) / NU
        D_low = noise_lower_bound(Dh, K * NU)
        T = conformal_threshold(V, kp)
        for m, tab in tabs.items():
            h = T * tab(D_low / T**2)
            rows.append(dict(shape=shape, regime=regime, seed=seed, rep=rep, method=m, half=h,
                             x=D_low / T**2, cov=float(cdf(shape, h) - cdf(shape, -h))))
    return rows


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    chunks = 10
    jobs = [(s, g, 250000 + 10000 * i + 1000 * j + c, reps // chunks)
            for i, s in enumerate(SHAPES) for j, g in enumerate(REGIMES) for c in range(chunks)]
    with Pool(procs) as pool:
        r = pd.DataFrame([row for rows in pool.imap_unordered(one, jobs) for row in rows])
    r.to_csv(RESULTS / 'certified_ldc.csv', index=False)
    s = r.groupby(['shape', 'regime', 'method']).agg(
        reps=('cov', 'size'), mean_cov=('cov', 'mean'),
        pr_cov_ge_90=('cov', lambda c: np.mean(c >= LEV)),
        width=('half', lambda h: 2 * h.mean())).reset_index()
    s.to_csv(RESULTS / 'certified_ldc_summary.csv', index=False)
    print(s.round(4).to_string(index=False))
