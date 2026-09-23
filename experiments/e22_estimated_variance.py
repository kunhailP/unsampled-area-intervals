"""E22. Robustness to estimated noise variances (synthetic, heterogeneous D as in E20).

Every rule receives D_hat_i = D_i chi2_nu / nu instead of D_i (nu = 4 and 10; D_hat
independent of V_i, as for normal unit data), and uses it as if known. No rule's guarantee
covers this case; the question is which ones keep the target Pr_D[cov >= .90] >= .95 in
practice. Rules as in E20 (FH_PAC, CP_PAC, LDC_PAC with the pooled lower bound of the mean,
HetLDC).
  python experiments/e22_estimated_variance.py [reps] [procs]
Writes results/estimated_variance.csv and results/estimated_variance_summary.csv.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from _common import RESULTS
from e12_conditional_synth import draw
from uai.latent_laws import cdf
from uai.procedures import (ShrinkTable, conformal_threshold, fay_herriot_boot_pac,
                            fay_herriot_reml, hetldc_halfwidth, noise_lower_bound, pac_rank)

warnings.filterwarnings('ignore')
K, DBAR, LEV = 110, 0.577, 0.90
SHAPES = ['normal', 'trunc_laplace']
NUS = [4, 10]


def one(args):
    shape, nu, seed = args
    rng = np.random.default_rng(seed); table = ShrinkTable('0.1')
    w = rng.lognormal(0, .7, K); D = DBAR * w / np.exp(.7**2 / 2)
    V = draw(shape, K, rng) + rng.normal(0, np.sqrt(D))
    Dh = D * rng.chisquare(nu, K) / nu
    kp = pac_rank(K, LEV, .05); T = conformal_threshold(V, kp)
    mu, A, vmu = fay_herriot_reml(V, Dh); se = np.sqrt(A + vmu)
    _, lam = fay_herriot_boot_pac(V, Dh, mu, A, rng, level=LEV)
    D_low = noise_lower_bound(Dh, K * nu)
    ints = {'FH_PAC': (mu, lam * se), 'CP_PAC': (0, T),
            'LDC_PAC': (0, T * table(D_low / T**2)), 'HetLDC': (0, hetldc_halfwidth(V, Dh, kp))}
    return [dict(shape=shape, nu=nu, seed=seed, method=m, half=h,
                 cov=float(cdf(shape, c + h) - cdf(shape, c - h))) for m, (c, h) in ints.items()]


if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = [(s, nu, 90000 + 1000 * i + 100000 * nu + r) for i, s in enumerate(SHAPES) for nu in NUS
            for r in range(reps)]
    with Pool(procs) as pool:
        r = pd.DataFrame([row for rows in pool.imap_unordered(one, jobs, chunksize=2) for row in rows])
    r.to_csv(RESULTS / 'estimated_variance.csv', index=False)
    s = r.groupby(['shape', 'nu', 'method']).agg(reps=('cov', 'size'), mean_cov=('cov', 'mean'),
                                                pr_cov_ge_90=('cov', lambda c: np.mean(c >= LEV)),
                                                width=('half', lambda h: 2 * h.mean())).reset_index()
    s.to_csv(RESULTS / 'estimated_variance_summary.csv', index=False)
    print(s.round(4).to_string(index=False))
