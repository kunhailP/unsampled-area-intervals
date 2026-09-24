"""Interval rules for an unsampled area, all centred at a separately trained predictor.

Scores on calibration areas: V_c = (survey estimate) - (centre), with design variance D_c.
Target on a new area: W = theta - centre.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

ROOT = Path(__file__).resolve().parents[2]
_FEASIBLE_X = 0.3696          # above this, P(|e| <= t) < .9: data contradict the noise level


class ShrinkTable:
    """Conservative lookup of the shrink function r(x) = R_{1-a,1-a}(x).

    alpha = '0.1' uses the exact two-dimensional reduction (results/r_exact_p0.9_q0.9.csv, E16,
    plus the fine small-x grid of E19). r rises for small x (r > 1: widen) to a peak near
    x = .0023 and then decreases. At or right of the peak the value at the grid point at or
    below x is >= r(x); left of it the overall maximum is returned. Other alphas use the E04
    differential-evolution table (same lookup, first value below the grid). Table values are
    converged grid maxima, not certified upper bounds (docs/THEORY_NOTE.md).
    """

    def __init__(self, alpha='0.1', path=ROOT / 'results' / 'r_table.json'):
        if alpha == '0.1' and (ROOT / 'results' / 'r_exact_p0.9_q0.9.csv').exists():
            import pandas as pd
            d = pd.read_csv(ROOT / 'results' / 'r_exact_p0.9_q0.9.csv')[['x', 'R']]
            b = ROOT / 'results' / 'boundary_map.csv'
            if b.exists():
                e = pd.read_csv(b)
                d = pd.concat([d, e[np.isclose(e.q, 0.9)][['x', 'R']]])
            d = d[np.isfinite(d.R)].sort_values('x')
            self.x, self.r = d.x.values, d.R.values
        else:
            tab = json.load(open(path))[alpha]
            self.x = np.array(sorted(float(k) for k in tab))
            self.r = np.array([tab[str(k)] for k in self.x])
        self.peak = int(np.argmax(self.r))

    def __call__(self, x):
        if x >= _FEASIBLE_X:        # alpha = .10 only; the data contradict the noise level
            return 1.0
        i = np.searchsorted(self.x, x, side='right') - 1
        if i <= self.peak:
            return float(self.r[self.peak])
        return float(self.r[i])



class CertifiedShrinkTable:
    """Certified upper bound U(x) >= R_{p,.9}(x) for p in {.90, .9036} (E24).

    Grid points x_i >= .01 carry branch-and-bound certificates U_i (`uai.certify`). For any x
    the lookup returns min over x_i <= x of U_i + c_q sqrt(x - x_i) (THEORY_NOTE Proposition 9),
    together with the closed bound of Theorem 5 (p = q: 1 + c_q sqrt x) or Theorem 5+
    (p = .9036: 1 + C sqrt x, C = -.057). Both constants are proved upper bounds in ball
    arithmetic (E27, `uai.interval`). Above the feasibility
    edge (P(|e| <= 1) < p) the data contradict the noise level and 1 is returned, as in
    ShrinkTable. Certificates rest on Proposition 1 (extremal family) and double-precision
    closed forms with a 1e-9 margin.
    """
    C_Q = 0.019062                       # c_.9 in [0.01906181, 0.01906188] (E27)
    C_SPLIT = {0.9: 0.019062, 0.9036: -0.057}   # C_{.9036,.9} in [-0.0571845, -0.057] (E27)

    def __init__(self, p=0.9, path=ROOT / 'results' / 'certified_R.csv'):
        import pandas as pd
        d = pd.read_csv(path)
        d = d[np.isclose(d.p, p) & np.isfinite(d.U)].sort_values('x')
        self.p, self.x, self.u = p, d.x.values, d.U.values
        self.c_split = self.C_SPLIT[round(p, 4)]
        self.x_edge = (1 / stats.norm.ppf((1 + p) / 2)) ** 2

    def __call__(self, x):
        if x >= self.x_edge:
            return 1.0
        best = 1 + self.c_split * np.sqrt(x)
        m = self.x <= x
        if m.any():
            best = min(best, float(np.min(self.u[m] + self.C_Q * np.sqrt(x - self.x[m]))))
        return float(best)

def conformal_threshold(scores, k):
    """k-th smallest |score| (1-based)."""
    return np.sort(np.abs(scores))[k - 1]


def noise_lower_bound(D_hat, df_total, eta=0.01):
    """Pooled chi-square lower confidence bound for a common noise variance."""
    return np.mean(D_hat) * df_total / stats.chi2.ppf(1 - eta, df_total)


def ldc_halfwidth(scores, D_low, k, table):
    t = conformal_threshold(scores, k)
    return t * table(D_low / t**2)


def fay_herriot_reml(V, D):
    """Intercept-only Fay-Herriot fit by REML with D treated as known. Returns (mu, A, var_mu)."""
    def nll(logA):
        A = np.exp(logA); w = 1 / (A + D); mu = (w * V).sum() / w.sum()
        return 0.5 * (np.log(A + D).sum() + np.log(w.sum()) + (w * (V - mu)**2).sum())
    r = minimize_scalar(nll, bounds=(np.log(1e-6 * np.var(V) + 1e-12),
                                     np.log(10 * np.var(V) + 1e-9)), method='bounded')
    A = np.exp(r.x); w = 1 / (A + D)
    return (w * V).sum() / w.sum(), A, 1 / w.sum()


def fay_herriot_boot_pivot(V, D, mu, A, rng, B=300):
    """Parametric-bootstrap draws of (theta* - mu*) / sqrt(A* + v*) for a new area."""
    piv = np.empty(B)
    for b in range(B):
        Vb = mu + rng.normal(0, np.sqrt(A), len(V)) + rng.normal(0, np.sqrt(D))
        thb = mu + rng.normal(0, np.sqrt(A))
        mub, Ab, vb = fay_herriot_reml(Vb, D)
        piv[b] = (thb - mub) / np.sqrt(Ab + vb)
    return piv


def pac_rank(K, level=0.90, delta=0.05):
    """Smallest k with P(Beta(k, K+1-k) >= level) >= 1 - delta: the k-th order statistic of
    K exchangeable scores then has conditional coverage >= level with probability >= 1 - delta."""
    for k in range(1, K + 1):
        if stats.beta.sf(level, k, K + 1 - k) >= 1 - delta:
            return k
    return None


def fay_herriot_boot_pac(V, D, mu, A, rng, B=300, level=0.90, delta=0.05):
    """One parametric bootstrap giving (i) pivot draws as in fay_herriot_boot_pivot and
    (ii) a tolerance multiplier lam: the (1 - delta) bootstrap quantile of the multiplier each
    refit needs for its interval mu* +- lam sqrt(A* + v*) to cover N(mu, A) with prob. level."""
    piv, lam = np.empty(B), np.empty(B)
    sA = np.sqrt(max(A, 1e-12))
    for b in range(B):
        Vb = mu + rng.normal(0, np.sqrt(A), len(V)) + rng.normal(0, np.sqrt(D))
        thb = mu + rng.normal(0, np.sqrt(A))
        mub, Ab, vb = fay_herriot_reml(Vb, D)
        se = np.sqrt(Ab + vb)
        piv[b] = (thb - mub) / se
        cov = lambda l: (stats.norm.cdf((mub + l * se - mu) / sA)
                         - stats.norm.cdf((mub - l * se - mu) / sA) - level)
        hi = 1.0
        while cov(hi) < 0 and hi < 1e4:
            hi *= 2
        lam[b] = brentq(cov, 0, hi) if hi < 1e4 else hi
    return piv, np.quantile(lam, 1 - delta)


def union_bound_halfwidth(t, D_up, p, q):
    """Assumption-light baseline: if P(|V| <= t) >= p > q, V = W + e, e ~ N(0, D), D <= D_up,
    then P(|W| <= s) >= q for s = t + sqrt(D_up) z_{1-(p-q)/2}, because |W| <= |V| + |e|.
    Needs no shape assumption; used as a safe reference against which LDC is measured."""
    if p <= q:
        return np.inf
    return t + np.sqrt(D_up) * stats.norm.ppf(1 - (p - q) / 2)


def clopper_pearson_lower(n_success, n, delta):
    """One-sided lower confidence bound for a success probability. By Hoeffding (1956) it is
    also valid for the mean success probability of independent, non-identical Bernoulli trials
    (a Poisson-binomial count is more concentrated than the binomial with the same mean)."""
    return 0.0 if n_success == 0 else float(stats.beta.ppf(delta, n_success, n - n_success + 1))


def shrink_mix(p, q, xs, wts=None, betas=None, ells=None, m=100):
    """R_{p,q} for heterogeneous known noise (average kernel over scaled variances xs):
    grid over (slope, length) with the exact endpoint reduction, then a local polish.
    A converging feasible value, not a certified upper bound (docs/THEORY_NOTE.md)."""
    from scipy.optimize import minimize
    from uai.extremal import shape_values
    betas = np.r_[0.0, np.exp(np.linspace(np.log(1e-2), np.log(1e3), 16))] if betas is None else betas
    ells = np.exp(np.linspace(np.log(1e-3), np.log(20), 30)) if ells is None else ells
    best, arg = -np.inf, None
    for beta in betas:
        v = shape_values(p, q, beta, ells, xs, wts, m=m)
        i = int(np.argmax(v))
        if v[i] > best:
            best, arg = v[i], (beta, ells[i])
    if arg is None:
        return np.inf                    # no log-concave law fits: fall back to no guarantee
    f = lambda z: -shape_values(p, q, np.exp(z[0]), [np.exp(z[1])], xs, wts, m=m)[0]
    r = minimize(f, [np.log(max(arg[0], 1e-3)), np.log(arg[1])], method='Nelder-Mead',
                 options={'xatol': 1e-6, 'fatol': 1e-10, 'maxiter': 60})
    return max(best, -r.fun) if np.isfinite(r.fun) else best


def quantised_kernel(xs, n_pts=32):
    """Compress scaled variances xs to n_pts group means (weights = group sizes) and return them
    with eps >= sup_w |gbar(w) - gbar_quantised(w)|, so that a constraint int gbar dG >= p can be
    replaced by the valid int gbar_q dG >= p - eps. The sup is taken on a grid of spacing h and
    padded by M h^2 / 8, M a bound on the second derivative of the difference (the maximum of a
    C^2 function between two grid points exceeds the larger endpoint value by at most M h^2/8)."""
    from scipy.special import ndtr
    xs = np.sort(np.asarray(xs, dtype=float))
    if len(xs) <= n_pts:
        return (xs, np.full(len(xs), 1 / len(xs))), 0.0
    tiny = xs < 1e-3                                  # near-indicator kernels: kept exactly
    groups = [xs[i:i + 1] for i in np.where(tiny)[0]] + \
        [g for g in np.array_split(xs[~tiny], max(n_pts - tiny.sum(), 1)) if len(g)]
    pts = np.array([np.mean(g) for g in groups]); wts = np.array([len(g) for g in groups]) / len(xs)
    span = 1 + 8 * np.sqrt(xs.max())
    wg = np.arange(-span, span + 1e-3, 1e-3)          # beyond +-span both kernels are < 1e-15
    kern = lambda x: ndtr((1 - wg[:, None]) / np.sqrt(x)) - ndtr((-1 - wg[:, None]) / np.sqrt(x))
    diff = np.abs(kern(xs).mean(1) - kern(pts) @ wts)
    # |g_x''| <= 2 phi(1) / x; a group's error has second derivative <= 4 phi(1) / min(group)
    M = sum(w * 4 * 0.2420 / g.min() for g, w in zip(groups, wts) if len(g) > 1)
    return (pts, wts), float(diff.max() + M * 1e-6 / 8)


def hetldc_halfwidth(V, D, k=None, q=0.90, delta=0.05, n_pts=32):
    """Heterogeneous-noise LDC with an order-statistic threshold T = |V|_(k).

    Let Fbar be the average CDF of |V_i| (independent, non-identical because D_i differ). For
    t_p = Fbar^{-1}(p), {Fbar(T) < p} = {#{|V_i| <= t_p} >= k}; the count is Poisson-binomial
    with mean K p, so by Hoeffding (1956) its upper tail at k >= K p + 1 is at most the
    Bin(K, p) tail. Hence p_k = Beta(k, K + 1 - k) delta-quantile still gives
    P(Fbar(T) >= p_k) >= 1 - delta. On that event int gbar_T dG >= p_k with gbar_T the average
    Gaussian kernel at x_i = D_i / T^2, and s = T * R_{p_k - eps, q}(kernel) covers the latent
    target with probability >= q for every log-concave G (eps: kernel quantisation error).
    Assumes W_i iid log-concave, e_i ~ N(0, D_i) independent of W with D_i known."""
    T, p_k, eps, (pts, wts), R = hetldc_parts(V, D, k, q, delta, n_pts)
    return T * R


def hetldc_parts(V, D, k=None, q=0.90, delta=0.05, n_pts=32):
    """Pieces of HetLDC: threshold T, level p_k, quantisation error eps, compressed kernel
    (pts, wts) and the grid value R of R^mix_{p_k - eps, q} (a lower bound of it)."""
    V, D = np.asarray(V), np.asarray(D)
    K = len(V)
    k = pac_rank(K, q, delta) if k is None else k
    p_k = float(stats.beta.ppf(delta, k, K + 1 - k))
    assert k >= K * p_k + 1, 'Hoeffding comparison needs k >= K p + 1'
    T = np.sort(np.abs(V))[k - 1]
    (pts, wts), eps = quantised_kernel(D / T**2, n_pts)
    return T, p_k, eps, (pts, wts), shrink_mix(p_k - eps, q, pts, wts=wts)


def hetldc_certified(V, D, k=None, q=0.90, delta=0.05, n_pts=32, parts=None,
                     tols=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1)):
    """HetLDC with a certified radius: the smallest R (1 + tol) that the branch and bound of
    `uai.certify` clears for the mixture kernel at level p_k - eps. If none clears, the
    shape-free union bound (C28) with the largest D is returned. Returns (half-width, tol);
    tol is nan for the fallback."""
    from uai.certify import certified_upper
    T, p_k, eps, (pts, wts), R = hetldc_parts(V, D, k, q, delta, n_pts) if parts is None else parts
    if not np.isfinite(R):
        return np.inf, np.nan
    s, tol, _ = certified_upper(p_k - eps, q, pts, R, rel_tols=tols, ws=wts, max_boxes=8_000_000)
    if np.isfinite(s):
        return T * s, tol
    return union_bound_halfwidth(T, np.max(D), p_k - eps, q), np.nan
