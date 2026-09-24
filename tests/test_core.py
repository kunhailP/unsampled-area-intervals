"""Fast checks of the core numbers. Run: python -m pytest -q"""
import json
import sys
from fractions import Fraction
from math import comb
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from uai.kernel import kernel                                   # noqa: E402
from uai.logconcave import (quantile_sd_constant, worst_coverage_normal_interval,  # noqa: E402
                            noisy_mass, latent_quantile)
from uai.procedures import ShrinkTable                          # noqa: E402


def test_two_world_observationally_equivalent_up_to_three_psus():
    q, p = Fraction(33, 1666), Fraction(400, 833)
    A = {40: Fraction(1, 2), 60: Fraction(1, 2)}; B = {0: q, 100: q, 49: p, 51: p}
    fm = lambda d, j: sum(w * np.prod([k - i for i in range(j)]) for k, w in d.items())
    assert [fm(A, j) for j in (1, 2, 3)] == [fm(B, j) for j in (1, 2, 3)]
    assert fm(A, 4) != fm(B, 4)


@pytest.mark.parametrize('m,r', [(2, 4), (4, 2), (3, 2), (6, 1)])
def test_kernel_rank_formula(m, r):
    a, _ = kernel(m, r)
    assert np.linalg.matrix_rank(a) == comb(m + r, r)


def test_mixed_design_rank_formula():
    a = np.vstack([kernel(2, 4)[0], kernel(4, 2)[0]])
    assert np.linalg.matrix_rank(a) == 15 + 15 - comb(2 + 2, 2)


def test_log_concave_quantile_constant():
    c90, _ = quantile_sd_constant(.10); c95, _ = quantile_sd_constant(.05)
    assert abs(c90 - 1.75585) < 1e-4 and abs(c95 - 2.16907) < 1e-4


def test_worst_normal_interval_coverage():
    assert abs(worst_coverage_normal_interval(.10)[0] - .8775) < 5e-4
    assert abs(worst_coverage_normal_interval(.05)[0] - .9297) < 5e-4


def test_shrink_table_monotone_and_conservative_lookup():
    T = json.load(open(ROOT / 'results' / 'r_table.json'))
    for a, tab in T.items():
        xs = sorted(float(k) for k in tab); rs = [tab[str(x)] for x in xs]
        assert all(rs[i + 1] <= rs[i] + 1e-6 for i in range(len(rs) - 1))
    st = ShrinkTable('0.1')
    assert st(0.136) >= st(0.137) and st(0.5) == 1.0


def test_gaussian_is_feasible_and_below_ldc_bound():
    # Gaussian W with P(|V|<=1) = .9 at x = .136: its 90% quantile must not exceed r_.10(.136)
    from scipy.stats import norm
    x = .136; sW = np.sqrt((1 / norm.ppf(.95))**2 - x)
    assert 1.645 * sW <= ShrinkTable('0.1')(0.136) + 1e-9


def test_pac_rank():
    from uai.procedures import pac_rank
    assert pac_rank(110, .90, .05) == 105
    assert pac_rank(110, .90, .04) == 105


def test_asymmetric_small_noise_counterexample():
    """Log-affine W on [0, b]: noisy mass >= .90 at t = 1 but latent mass < .90, so r > 1."""
    from scipy.integrate import quad
    from scipy.stats import norm
    b, x = 1.0681, 0.0005
    f = lambda w: np.exp(w) / np.expm1(b)
    s = np.sqrt(x)
    noisy = quad(lambda w: f(w) * (norm.cdf((1 - w) / s) - norm.cdf((-1 - w) / s)), 0, b,
                 points=[1], epsabs=1e-14, limit=500)[0]
    latent = np.expm1(1) / np.expm1(b)
    assert noisy > 0.90 > latent


def test_shrink_table_small_x_not_one():
    from uai.procedures import ShrinkTable
    t = ShrinkTable('0.1')
    assert t(1e-5) == t.r.max() > 1              # left of the peak: overall maximum
    assert t(0.003) >= t(0.004) >= t(0.2)


def test_union_bound_baseline():
    """Monte Carlo check of the shape-free baseline on a skewed W."""
    from uai.procedures import union_bound_halfwidth
    rng = np.random.default_rng(0)
    W = rng.exponential(1, 400000) - 1; D = 0.3
    V = W + rng.normal(0, np.sqrt(D), W.size)
    t = np.quantile(np.abs(V), 0.95); s = union_bound_halfwidth(t, D, 0.95, 0.90)
    assert np.mean(np.abs(W) <= s) >= 0.90
    assert union_bound_halfwidth(1, D, 0.9, 0.9) == np.inf


def test_latent_laws_match_e12_draws():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
    from e12_conditional_synth import SHAPES, draw
    from uai.latent_laws import cdf, matched_scales
    rng = np.random.default_rng(3)
    grid = np.linspace(-3, 3, 61)
    for k in SHAPES:
        w = np.sort(draw(k, 400000, rng))
        emp = np.searchsorted(w, grid, side='right') / w.size
        assert np.max(np.abs(emp - cdf(k, grid))) < 0.004, k
    m = matched_scales('normal', [0, 0, 0], [-1, -2, -3], [1, 2, 3])
    assert abs(m['lam_pac'] - 1.6448536269514722) < 1e-9 and m['achieved_reliability'] == 1


def test_small_noise_upper_bound_all_x():
    """Theorem 5: R_{q,q}(x) <= 1 + c_q sqrt(x); grid values of R are lower bounds of R."""
    from uai.extremal import R_grid, one_sided_constant
    cq = one_sided_constant(.9, .9)
    assert abs(cq - 0.0190618) < 1e-6
    betas = np.concatenate([[0], np.geomspace(.05, 3000, 30)]); ells = np.geomspace(1e-3, 4, 30)
    for x in (1e-5, 1e-3, 1e-2):
        R, _ = R_grid(.9, .9, x, betas, ells)
        assert R <= 1 + cq * np.sqrt(x) + 1e-12


def test_box_bounds_are_sound():
    """Proposition 10: every law inside a box obeys the box's noisy upper / latent lower bound."""
    from uai.certify import _cdf_latent, _cdf_noisy, box_status
    rng = np.random.default_rng(3)
    for _ in range(300):
        x, s = 10 ** rng.uniform(-2, -0.5), rng.uniform(.2, 1.3)
        u0, v0 = rng.uniform(0, .95, 2); u1, v1 = u0 + rng.uniform(0, .05), v0 + rng.uniform(0, .05)
        b0 = rng.uniform(-1.2, 2); b1 = b0 + rng.uniform(0, .2)
        box = np.array([[u0, u1, v0, v1, b0, b1]])
        u, v, b = rng.uniform(u0, u1, 20), rng.uniform(v0, v1, 20), rng.uniform(b0, b1, 20)
        be, el = u / (1 - u), v / (1 - v)
        N = _cdf_noisy(1.0, b, el, be, x) - _cdf_noisy(-1.0, b, el, be, x)
        L = _cdf_latent(s, b, el, be) - _cdf_latent(-s, b, el, be)
        ok_n, _ = box_status(N.max() - 1e-6, 2.0, s, [x], [1.0], box)     # p just below max N
        assert not ok_n[0]                                                  # N_up >= max N
        bad_l, _ = box_status(-1.0, L.min() + 1e-6, s, [x], [1.0], box)
        assert not bad_l[0]                                                 # L_low <= min L


def test_certify_brackets_R():
    from uai.certify import certify
    assert certify(.9, .9, .8984, .116)[0]
    assert not certify(.9, .9, .8970, .116, max_boxes=2_000_000)[0]


def test_certified_lookup_dominates_grid_values():
    """Feasible grid values (E16) are lower bounds of R, so they may never exceed the lookup."""
    import pandas as pd
    from uai.procedures import CertifiedShrinkTable
    tab = CertifiedShrinkTable(0.9)
    e = pd.read_csv(ROOT / 'results' / 'r_exact_p0.9_q0.9.csv')
    for x, r in zip(e.x, e.R):
        if np.isfinite(r) and x < tab.x_edge:
            assert r <= tab(x) + 1e-12


def test_interval_c_enclosure_contains_float_value():
    """Ball-arithmetic enclosure of c_.9 (E27) brackets the double-precision value."""
    from uai.extremal import one_sided_constant
    from uai.interval import c_enclosure
    lo, hi, _ = c_enclosure('0.9', '0.9', n0=200, max_iter=3000)
    assert lo <= one_sided_constant(.9, .9) <= hi and hi - lo < 1e-3


def test_interval_b_bounds_and_monotone_in_u():
    """b_u(p) is enclosed and nonincreasing in u (kappa_u(b) decreases in u)."""
    from uai.interval import A, b_bounds, kappa
    p = A('0.9')
    prev = np.inf
    for u in (0.05, 0.3, 1.0, 4.0, 20.0):
        lo, hi = b_bounds(u, p)
        assert lo < hi and hi <= prev + 1e-12
        prev = hi
        assert kappa(u, lo) > p and kappa(u, hi) < p
    assert kappa(2.0, 0.3) < kappa(1.0, 0.3)


def test_interval_constants_file():
    """E27 certificates: C_{.9036,.9} <= -0.057 and the slack brackets."""
    d = json.loads((ROOT / 'results' / 'interval_constants.json').read_text())
    assert all(s['ok'] for s in d['split'])
    assert any(s['g'] == -0.057 for s in d['split'])
    for q, s in d['slack'].items():
        assert s['enough']['ok'] and s['not_enough']['positive']
    from uai.procedures import CertifiedShrinkTable
    assert CertifiedShrinkTable.C_Q >= d['c']['0.9'][1]
    assert CertifiedShrinkTable.C_SPLIT[0.9036] >= -0.057


def test_min_variance_plugin_not_conservative():
    """External review counterexample (checked in 192-bit balls): noisy coverage >= .9 under an
    equal mixture of N(0, 1e-8) and N(0, 5e-4), yet the min-variance radius 1 + c_.9 * 1e-4 covers
    W = b - Exp(2) with probability < .9."""
    from flint import arb, ctx
    old = ctx.prec
    ctx.prec = 192
    try:
        Phi = lambda z: (-z / arb(2).sqrt()).erfc() / 2
        b, rate = arb('1.04356470165499'), arb(2)

        def noisy_cdf(t, D):
            sd = D.sqrt(); z = (t - b) / sd
            return Phi(z) + (rate * (t - b) + rate**2 * D / 2).exp() * Phi(-z - rate * sd)

        nm = lambda D: noisy_cdf(1, D) - noisy_cdf(-1, D)
        D0, D1 = arb('1e-8'), arb('0.0005')
        h = 1 + arb('0.019062') * D0.sqrt()
        assert (nm(D0) + nm(D1)) / 2 > arb('0.9')
        assert (rate * (h - b)).exp() - (rate * (-h - b)).exp() < arb('0.9')
    finally:
        ctx.prec = old


def test_outward_rounding():
    from flint import arb
    from uai.interval import A, fdown, fup, log_grid
    z = arb(1) / 3
    assert A(float(z.upper())) < z                 # plain conversion can round down
    assert A(fup(z)) >= z and A(fdown(z)) <= z
    T = 1 - A('0.9036')
    assert A(fup(T / 32)) >= T / 32
    g = log_grid(0.0189, 64.0, 50)
    assert g[0] == 0.0189 and g[-1] == 64.0 and np.all(np.diff(g) >= 0)


def test_envelope_dominates_lookup_to_the_right():
    """envelope(x_low) >= lookup(x) for every x >= x_low below the feasibility edge."""
    from uai.procedures import CertifiedShrinkTable
    for p in (0.9, 0.9036, 0.9068):
        tab = CertifiedShrinkTable(p)
        xs = np.linspace(1e-5, tab.x_edge * 0.999, 400)
        look = np.array([tab(x) for x in xs])
        for i in range(0, 400, 23):
            assert tab.envelope(xs[i]) >= look[i:].max() - 1e-12


def test_quantised_kernel_merges_equal_variances():
    from uai.procedures import quantised_kernel
    (pts, wts), eps = quantised_kernel(np.r_[np.full(60, .1), np.full(50, .2)])
    assert eps == 0.0 and np.allclose(pts, [.1, .2]) and np.allclose(wts, [60 / 110, 50 / 110])


def test_scale_lemma_on_grid_values():
    """Lemma 12: R(lambda x) <= sqrt(lambda) R(x), checked on converged grid values."""
    from uai.procedures import shrink_mix
    xs, w = np.array([.05, .15]), np.array([.5, .5])
    r1 = shrink_mix(.9036, .9, xs, wts=w)
    for lam in (1.1, 1.5, 3.0):
        assert shrink_mix(.9036, .9, lam * xs, wts=w) <= np.sqrt(lam) * r1 + 1e-9


def test_areawise_envelope_dominates_true_kernel():
    """Model H: on the event that every area-wise interval covers, u >= gbar pointwise."""
    from scipy.special import ndtr
    from uai.estimated import h_kernel
    rng = np.random.default_rng(0)
    D = rng.uniform(.1, 1, 40); Dh = D * rng.chisquare(20, 40) / 20; T = 2.0
    w, u, ns = h_kernel(Dh, 20, T, alpha=1e-9, eta=.5)
    from scipy import stats
    L = 20 * Dh / stats.chi2.ppf(1 - 5e-10, 20); U = 20 * Dh / stats.chi2.ppf(5e-10, 20)
    assert np.all((L <= D) & (D <= U))
    x = D[:, None] / T**2
    g = (ndtr((1 - w) / np.sqrt(x)) - ndtr((-1 - w) / np.sqrt(x))).mean(0)
    assert np.all(u >= g - 1e-12)


def test_kernel_envelope_majorises_and_vanishes():
    from scipy.special import ndtr
    from uai.estimated import kernel_envelope
    w = np.linspace(-6, 6, 1201)
    M = kernel_envelope(w)
    for x in np.geomspace(1e-4, 100, 60):
        g = ndtr((1 - w) / np.sqrt(x)) - ndtr((-1 - w) / np.sqrt(x))
        assert np.all(g <= M + 1e-12)
    assert kernel_envelope(np.array([50.0]))[0] < .02
    near = kernel_envelope(np.array([1 - 1e-9, 1 + 1e-9, -1 - 1e-9]))
    assert np.all(np.abs(near - 1) < 1e-6)                  # continuous at |w| = 1 (ramp)


def test_shape_free_markov_is_valid_for_a_bimodal_law():
    """The Markov bound needs no shape: check the population version on a two-point latent law."""
    from scipy.special import ndtr
    from uai.procedures import shape_free_markov
    rng = np.random.default_rng(1)
    K = 110
    D = np.full(K, .3)
    W = rng.choice([-1.5, 1.5], K)
    V = W + rng.normal(0, np.sqrt(D))
    h, p_k, T = shape_free_markov(V, D, 108)
    # on the order-statistic event, P(|V| <= T) >= p_k; the latent two-point law is covered
    # with probability 1 or 0, and the bound says it is covered when the noisy mass allows
    noisy = ndtr((T - 1.5) / np.sqrt(.3)) - ndtr((-T - 1.5) / np.sqrt(.3))
    if noisy >= p_k:
        assert h >= 1.5
