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
