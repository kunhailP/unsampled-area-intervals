# Draft core (Biometrika Miscellanea candidate)

Status: working draft, 2026-09-24. Every claim carries its status from `docs/THEORY_NOTE.md`. Parts marked [open] are not yet proved and must not be stated as theorems.

---

**Title (working).** When is calibration on noisy responses conservative for the latent target? Sharp transfer under log-concavity

**Summary (draft).** Prediction intervals for a latent quantity, such as the true mean of a small area that was never sampled, are often calibrated on noisy proxies: survey direct estimates whose sampling variances are known. For symmetric unimodal latent errors, Anderson's theorem implies that noisy calibration is conservative. We ask what can be guaranteed without symmetry. Over all log-concave latent laws, we characterise the smallest latent radius implied by a noisy coverage level. The problem reduces exactly to a two-parameter family, with closed-form Gaussian convolutions. Three consequences follow.
- For every coverage level, noisy calibration can undercover the latent target when the noise is small. The shortfall is of order √x, where x is the noise-to-threshold variance ratio. We identify the leading constant [lower bound proved; matching upper bound by a proof skeleton with two technical gaps].
- This defines a critical noise level below which the noisy threshold must be widened and above which it may be shrunk.
- With heterogeneous known noise variances, the exact constraint uses the average Gaussian kernel. Plugging in the mean variance can be anti-conservative, and plugging in the minimum variance can waste up to 14% of the width.

Order-statistic calibration keeps its Beta coverage law under heterogeneous independent noise, by Hoeffding's comparison of Poisson-binomial and binomial tails. This gives a finite-sample procedure whose conditional reliability holds for every log-concave latent law.

---

## 1. Problem

Let W be a latent residual (target minus a centre fixed on independent training data) and V = W + e with e ~ N(0, D) independent of W. After scaling a threshold t to 1, write x = D/t². Define

  R_{p,q}(x) = sup{ Q_q(|W|) : W log-concave, pr(|W + x^{1/2}Z| ≤ 1) ≥ p }.

If calibration certifies noisy coverage ≥ p at threshold t, then t·R_{p,q}(D/t²) is the smallest radius that guarantees latent coverage ≥ q for every log-concave W. "Sharp" refers to this class and this constraint only. It does not mean optimal among all procedures that use the full calibration sample.

Relation to prior work:
- Anderson (1955) gives R_{q,q} ≤ 1 on the symmetric unimodal subclass. Einbinder et al. (2024) use this for conformal prediction with noisy labels.
- Cohen, Goldberger & Tirer (2025) deconvolve the noisy score distribution to shrink the threshold, but without a finite-sample guarantee.
- For symmetric log-concave laws, the quantile/sd extremal problem is Barthe & Koldobsky (2003); see He, Tkocz & Wyczesany (2023, Lemma 3).
- Fradelizi & Guédon (2004) supply the localisation step.
- Our contribution is the asymmetric transfer problem, its exact reduction, and the phenomena it reveals.

## 2. Exact reduction

**Proposition 1** [sketch; truncation limit to be written]. R_{p,q}(x) is the supremum over Dirac masses and log-affine densities on a segment.

**Proposition 2** [proved]. For those laws the noisy mass has the closed form given in the note (integration by parts), so no quadrature is needed.

**Proposition 3** [proved]. Fix the slope β and length ℓ and vary the location c. Then pr(|W + c + e| ≤ 1) and pr(|W + c| ≤ s) are log-concave in c (Prékopa), so the feasible locations form an interval and the latent quantile is quasi-convex in c. Hence

  R_{p,q}(x) = max{ Dirac value, sup_{β ≥ 0, ℓ > 0} max(Q at the two feasible endpoints) },

a two-dimensional maximisation of closed-form quantities. Numerically, this and an independent differential-evolution search over the three-parameter family agree to 1.5 × 10⁻¹³ at all 73 tabulated x ≥ 0.006.

## 3. Small noise

**Theorem 4** [lower bound proved]. For every q ∈ (0, 1), R_{q,q}(x) ≥ 1 + c_q x^{1/2} + o(x^{1/2}), with

  c_q = sup_{u>0} u⁻¹ log{ e^{u²/2} Φ(b_u − u) + e^{u b_u} Φ(−b_u) } > 0,

where b_u solves Φ(−b) + e^{−ub+u²/2}Φ(b − u) = q. The extremal sequence is a density rising exponentially towards a cut-off just outside the threshold, with a boundary layer of width x^{1/2}. Values: c_{0.8} = 0.046, c_{0.9} = 0.019, c_{0.95} = 0.0084, c_{0.99} = 0.0014.

**Theorem 5** [proof skeleton; two technical gaps listed in the note]. Let c_q be the value of the scaled one-sided problem, sup{F_Y⁻¹(q) : Y log-concave, pr(Y + Z ≤ 0) ≥ q}. Then R_{q,q}(x) = 1 + c_q x^{1/2} + o(x^{1/2}). The proof has four steps:
- By the symmetry W ↦ −W, restrict to increasing log-affine laws. Jensen's inequality then shows that noise can only lose mass at the left endpoint.
- Condition on W ≥ −1, and use scale invariance to reduce to the one-sided problem at a level q′ ≥ q.
- Show that q ↦ c_q is nonincreasing, by a left-truncation argument.
- The formula of Theorem 4 equals c_q numerically: the gap is ≤ 3 × 10⁻⁸ over the whole extremal family. The exact two-dimensional maximum matches 1 + c_{0.9}x^{1/2} to seven digits at x = 10⁻⁶, 10⁻⁵ and 10⁻⁴.

Interpretation. A naive Taylor expansion suggests an O(x) effect, driven by f′(1) − f′(−1). The x^{1/2} rate comes from the boundary layer. Symmetry removes the effect entirely.

**Critical noise level** [numerical]. R_{q,q}(x) = 1 at x*(q) = 0.034, 0.017, 0.0062 and 0.0012 for q = 0.80, 0.85, 0.90 and 0.95. Below x*(q) the noisy threshold must be widened, by at most 0.37%, 0.17%, 0.063% and 0.012% respectively. Above it the threshold may be shrunk. [Figure: R_{q,q}(x) against x on a log scale, four q, with the symmetric bound 1.]

**Corollary 6** [numerical; depends on Theorem 5]. A finite-sample slack p − q ≥ 8 × 10⁻⁴ removes the small-noise widening at q = 0.9. The order-statistic slack at K = 110 (0.0036) exceeds this.

## 4. Heterogeneous noise and a finite-sample procedure

With e_i ~ N(0, D_i) and D_i known, the scores are independent but not identically distributed.

**Proposition 7** [proved]. Let T = |V|_(k), F̄ be the average distribution function of |V_i|, and p_k be the δ-quantile of Beta(k, K+1−k). If k ≥ Kp_k + 1, then pr{F̄(T) ≥ p_k} ≥ 1 − δ. The proof is Hoeffding's (1956) tail comparison for the Poisson-binomial count #{|V_i| ≤ F̄⁻¹(p)}.

On that event the latent law satisfies one linear constraint with the average kernel ḡ_T(w) = K⁻¹Σ_i pr(|w + e_i| ≤ T). Propositions 1–3 apply unchanged, except that feasible locations may now form a union of intervals; only the outermost endpoints matter. This gives HetLDC:

  s = T · R^{mix}_{p_k − ε, q}(D_1/T², …, D_K/T²),

where ε bounds the error from compressing the D_i to 32 support points. [proved, modulo the numerical evaluation of R^{mix}.]

**Mean-variance plug-in** [numerical]. The common-variance shortcut R_{p,q}(D̄/T²) is not conservative in general. At D̄/T² = 0.2 with lognormal spread 1.0 it is 1.4% too short. The minimum-variance plug-in, which is conservative, is up to 14% longer than exact.

## 5. Empirical illustration [to be filled from E20 and the apipop study]

Target: pr_D{ pr(W_new ∈ C | D) ≥ 0.90 } ≥ 0.95, for synthetic latent laws and the apipop finite population.

Comparators:
- Fay–Herriot with a parametric-bootstrap tolerance multiplier. It is shortest, but misses the target under non-normal latent laws: 0.916 for truncated Laplace with heterogeneous D.
- Conformal on noisy scores with the order-statistic rank.
- Mean-variance LDC.
- HetLDC.

## 6. Limits

- Log-concavity of the latent residual is assumed, not tested.
- The noise is Gaussian and independent of W, with known variances.
- Estimated variances need an outer confidence set.
- Certification of R by interval bounds on the two-dimensional grid is not yet done.
- The effect sizes of the small-noise phenomenon are small. The practical gains are in the heterogeneous-noise constraint and the finite-sample procedure.
