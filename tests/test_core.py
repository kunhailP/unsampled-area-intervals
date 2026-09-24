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
