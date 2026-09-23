"""E10. Negative result: giving training and calibration areas different survey depths.

Rough Gaussian planning model (A = 1): OLS centre with p covariates on K_T areas with label
noise S2/n_T; calibration on K_C areas with noise S2/n_C; width = 2 z sqrt(min(UCL of Var W,
Var W + D_C)). Budget B = K(c0 + n). Differentiated depths beat the best uniform design by
only 0.6-2.3%. Writes results/depth_allocation.csv.
"""
import numpy as np
import pandas as pd

from _common import RESULTS

z = zeta = 1.645


def width(KT, nT, KC, nC, S2, p):
    AW = 1 + p * (1 + S2 / nT) / KT; DC = S2 / nC
    ucl = AW + zeta * np.sqrt(2 / KC) * (AW + DC)
    return 2 * z * np.sqrt(min(ucl, AW + DC))


def best(B, c0, S2, p, uniform):
    res = (9e9,)
    ns = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    for nT in ns:
        for nC in ([nT] if uniform else ns):
            for fT in np.linspace(.05, .95, 19):
                KT = int(fT * B / (c0 + nT)); KC = int((1 - fT) * B / (c0 + nC))
                if KT <= p + 2 or KC < 10: continue
                w = width(KT, nT, KC, nC, S2, p)
                if w < res[0]: res = (w, KT, nT, KC, nC)
    return res


rows = []
for S2 in (155, 220, 52):
    for B in (20000, 60000):
        u = best(B, 20, S2, 5, True); d = best(B, 20, S2, 5, False)
        rows.append(dict(S2_over_A=S2, budget=B, uniform_width=u[0], split_width=d[0],
                         gain_pct=100 * (1 - d[0] / u[0]), split_design=f'KT={d[1]} nT={d[2]} KC={d[3]} nC={d[4]}'))
df = pd.DataFrame(rows); df.to_csv(RESULTS / 'depth_allocation.csv', index=False); print(df.to_string(index=False))
