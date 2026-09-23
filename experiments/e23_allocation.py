"""E23. Budget allocation and the certified latent half-width (population limit, K -> inf).

Unit-level variance fixed so that equal allocation n = 3 gives D = 0.577 (Var W = 1). A design
gives share pi of areas n1 units and the rest n2, with pi n1 + (1 - pi) n2 = 3, so the budget
is fixed. Noise D_i = 1.731 / n_i. For a latent law G the noisy threshold T is the p-quantile of
|W + e| under the design's noise mixture (p = p_k = .9036, the K = 110 order-statistic level),
and the certified half-width is s = T * R^mix_{p,q}(D_i / T^2) (worst case over log-concave G).
The oracle half-width is the q-quantile of |W| under G.
  python experiments/e23_allocation.py [procs]
Writes results/allocation.csv.
"""
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from _common import RESULTS
from uai.latent_laws import cdf
from uai.procedures import shrink_mix

warnings.filterwarnings('ignore')
P, Q, S2 = 0.9036, 0.90, 0.577 * 3
DESIGNS = [('equal n=3', [(1.0, 3)]), ('half 2 / half 4', [(.5, 2), (.5, 4)]),
           ('half 1 / half 5', [(.5, 1), (.5, 5)]), ('1/4 at 1, 3/4 at 3.67', [(.25, 1), (.75, 11 / 3)]),
           ('3/4 at 2, 1/4 at 6', [(.75, 2), (.25, 6)])]
LAWS = ['normal', 'laplace', 'gamma2_skew']


def noisy_abs_cdf(kind, t, comps):
    """P(|W + e| <= t) for the noise mixture, by quadrature over W."""
    w = np.linspace(-12, 12, 24001); dw = w[1] - w[0]
    f = np.gradient(cdf(kind, w), dw)
    out = 0.0
    for pi, n in comps:
        s = np.sqrt(S2 / n)
        out += pi * np.sum(f * (stats.norm.cdf((t - w) / s) - stats.norm.cdf((-t - w) / s))) * dw
    return out


def job(args):
    kind, (name, comps) = args
    T = brentq(lambda t: noisy_abs_cdf(kind, t, comps) - P, 0.1, 20)
    xs = np.array([S2 / n for _, n in comps]) / T**2          # exact mixture: distinct values
    R = shrink_mix(P, Q, xs, wts=np.array([pi for pi, _ in comps]))
    oracle = brentq(lambda s: cdf(kind, s) - cdf(kind, -s) - Q, 0.01, 20)
    return dict(law=kind, design=name, mean_D=sum(pi * S2 / n for pi, n in comps), T=T, R=R,
                half_width=T * R, oracle=oracle, ratio=T * R / oracle)


if __name__ == '__main__':
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with Pool(procs) as pool:
        df = pd.DataFrame(pool.map(job, [(k, d) for k in LAWS for d in DESIGNS]))
    df.to_csv(RESULTS / 'allocation.csv', index=False)
    pd.set_option('display.width', 200)
    print(df.round(4).to_string(index=False))
