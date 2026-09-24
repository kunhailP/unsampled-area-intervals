> **Superseded (2026-09-24).** The current manuscript is [`paper/main.tex`](../paper/main.tex), and the current status of every claim is in [`FINDINGS.md`](FINDINGS.md). This early Markdown draft is kept for history only; some statements here were later corrected (e.g. C44–C49, C54).

# Draft core (Biometrika Miscellanea candidate)

Status: working draft, 2026-09-24. Every claim carries its status from `docs/THEORY_NOTE.md`. Parts marked [open] are not yet proved and must not be stated as theorems.

---

**Title (working).** When is calibration on noisy responses conservative for the latent target? Sharp transfer under log-concavity

**Summary (draft).** Prediction intervals for a latent quantity, such as the true mean of a small area that was never sampled, are often calibrated on noisy proxies: survey direct estimates whose sampling variances are known. For symmetric unimodal latent errors, Anderson's theorem implies that noisy calibration is conservative. We ask what can be guaranteed without symmetry. Over all log-concave latent laws, we characterise the smallest latent radius implied by a noisy coverage level. The problem reduces exactly to a two-parameter family, with closed-form Gaussian convolutions. Three consequences follow.
- For every coverage level, noisy calibration can undercover the latent target when the noise is small. The shortfall is of order √x, where x is the noise-to-threshold variance ratio. The radius shortfall is at most c_q x^{1/2} for every x, and this constant is sharp as x → 0 [proved].
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

**Proposition 1** [proved]. R_{p,q}(x) is the supremum over Dirac masses and log-affine densities on a segment.

**Proposition 2** [proved]. For those laws the noisy mass has the closed form given in the note (integration by parts), so no quadrature is needed.

**Proposition 3** [proved]. Fix the slope β and length ℓ and vary the location c. Then pr(|W + c + e| ≤ 1) and pr(|W + c| ≤ s) are log-concave in c (Prékopa), so the feasible locations form an interval and the latent quantile is quasi-convex in c. Hence

  R_{p,q}(x) = max{ Dirac value, sup_{β ≥ 0, ℓ > 0} max(Q at the two feasible endpoints) },

a two-dimensional maximisation of closed-form quantities. Numerically, this and an independent differential-evolution search over the three-parameter family agree to 1.5 × 10⁻¹³ at all 73 tabulated x ≥ 0.006.

**Proposition 9 (continuity in the noise level)** [proved]. For all x, h ≥ 0, R_{p,q}(x + h) ≤ R_{p,q}(x) + c_q h^{1/2}.

Proof. If W is feasible at noise x + h, then W + e_h is log-concave and feasible at noise x, so t = Q_q(|W + e_h|) ≤ R_{p,q}(x). Theorem 5 at scale t then gives Q_q(|W|) ≤ t + c_q h^{1/2}.

Theorem 5 is the case x = 0.

**Certified values** [method proved; values certified in floating point]. Write the extremal law as W = b − Y, where Y has density ∝ e^{−βy} on [0, ℓ]. W is stochastically increasing in β and b and decreasing in ℓ. Both pr(W + e ≤ ±1) are decreasing functionals of W. So on any box of (β, ℓ, b) the noisy mass has an upper bound, and the latent mass a lower bound, given by closed forms at two corners.

Branch and bound over boxes, with β, ℓ ∈ [0, ∞] compactified and b confined analytically, then certifies R_{p,q}(x) ≤ s. Proposition 9 extends certificates from grid points to all x.

For q = 0.9 and x ∈ [0.05, 0.3], the certified upper bound is within 3 × 10⁻⁴ (relative) of a feasible value. The procedures in §4 use these certified values [Table: R_{0.9,0.9} and R_{0.9036,0.9} with certified bounds]. At the order-statistic level p_k = 0.9036, the certified table is a median 2.1% below the p = q table. Using it shortens LDC_PAC by 1.6–1.7% at no cost to the guarantee.

## 3. Small noise

Let Z ~ N(0, 1). The scaled one-sided problem is

  c_{p,q} = sup{ F_Y⁻¹(q) : Y log-concave, pr(Y + Z ≤ 0) ≥ p },  c_q = c_{q,q}.

By Remark 5′ below, t + D^{1/2}c_{p,q} is the sharp one-sided transfer for every D.

**Lemma A** [proved]. The one-sided problem is solved by exponential tails. For Y = b − E, with E exponential of rate u, pr(Y + Z ≤ 0) = Φ(−b) + e^{−ub+u²/2}Φ(b − u). If b_u(p) is the root of this at level p, then c_{p,q} = sup_{u>0}{b_u(p) − u⁻¹log(1/q)}. The proof takes a supergradient of the concave function log F_Y at the q-quantile: it bounds F_Y by the exponential-tail distribution function with the same q-quantile. In particular

  c_q = sup_{u>0} u⁻¹ log{ e^{u²/2} Φ(b_u − u) + e^{u b_u} Φ(−b_u) } > 0,  b_u = b_u(q).

Values: c_{0.8} = 0.046, c_{0.9} = 0.019, c_{0.95} = 0.0084, c_{0.99} = 0.0014.

**Theorem 4** [proved]. For every q ∈ (0, 1), R_{q,q}(x) ≥ 1 + c_q x^{1/2} − o(x^{1/2}). The extremal sequence is W = 1 + x^{1/2}Y with Y an exponential tail at a slightly raised level. This is a density rising exponentially towards a cut-off just outside the threshold, with a boundary layer of width x^{1/2}. The mass lost through the far endpoint is exponentially small in x^{−1/2}.

**Theorem 5** [proved]. For every q ∈ (0, 1) and **every** x > 0, R_{q,q}(x) ≤ 1 + c_q x^{1/2}. Hence R_{q,q}(x) = 1 + c_q x^{1/2} + o(x^{1/2}) as x → 0.

The proof splits the failure budget between the two endpoints.
- If pr(|W + e| ≤ 1) ≥ q, the noisy mass leaving on the right, α_R, and on the left, α_L, satisfy α_L + α_R ≤ 1 − q.
- On the right, (W − 1)/x^{1/2} is feasible for the one-sided problem at level 1 − α_R ≥ q. Left truncation shows that q ↦ c_q is nonincreasing, so pr(W > 1 + c_q x^{1/2}) ≤ α_R.
- The same argument applied to −W gives pr(W < −1 − c_q x^{1/2}) ≤ α_L.
- Adding the two gives the claim.

The argument uses only log-concavity of distribution functions, translation, scaling and truncation. It does not use the reduction of Proposition 1.

**Theorem 5+** [proved; constant numerical]. For p ≥ q and every x > 0, R_{p,q}(x) ≤ 1 + C_{p,q}x^{1/2}, where

  C_{p,q} = sup_{α_L+α_R=1−p} inf_{β_L+β_R=1−q} max(c_{1−α_L,1−β_L}, c_{1−α_R,1−β_R}) ≥ c_{p,q}.

Numerically C_{p,q} = c_{p,q}, so the worst split puts the whole budget on one side.

Interpretation. A naive Taylor expansion suggests an O(x) effect, driven by f′(1) − f′(−1). The x^{1/2} rate comes from the boundary layer. Symmetry removes the effect entirely. The bound of Theorem 5 is uniform in x, so the undercoverage of an uncorrected noisy threshold never exceeds c_q x^{1/2} in radius.

**Remark 5′ (one-sided transfer)** [proved]. The map W ↦ (W − t)/D^{1/2} preserves log-concavity, so the sharp one-sided transfer is exactly t + D^{1/2}c_{p,q} for every D, not only asymptotically. At the order-statistic level for K = 110 and q = 0.95 per side, c = −0.011, which is negligible. The gain from deconvolution therefore comes from two-sided calibration, where noise pushes mass out through both ends at once. The small-noise law is the one-sided effect seen locally at each endpoint.

**Critical noise level** [numerical]. R_{q,q}(x) = 1 at x*(q) = 0.034, 0.017, 0.0062 and 0.0012 for q = 0.80, 0.85, 0.90 and 0.95. Below x*(q) the noisy threshold must be widened, by at most 0.37%, 0.17%, 0.063% and 0.012% respectively. Above it the threshold may be shrunk. [Figure: R_{q,q}(x) against x on a log scale, four q, with the symmetric bound 1.]

**Corollary 6** [inequality proved; constants numerical]. If C_{p,q} ≤ 0, then R_{p,q}(x) ≤ 1 for every x, so no widening is ever needed. The smallest slack p − q with C_{p,q} ≤ 0 is 4.4 × 10⁻³, 8.0 × 10⁻⁴ and 1.6 × 10⁻⁴ at q = 0.80, 0.90 and 0.95. The order-statistic slack at K = 110 and k = 105 is 0.0036. There C_{0.9036,0.90} = −0.057, so the finite-sample procedure shrinks the noisy threshold by at least a factor 1 − 0.057x^{1/2} at every noise level.

## 4. Heterogeneous noise and a finite-sample procedure

With e_i ~ N(0, D_i) and D_i known, the scores are independent but not identically distributed.

**Proposition 7** [proved]. Let T = |V|_(k), F̄ be the average distribution function of |V_i|, and p_k be the δ-quantile of Beta(k, K+1−k). If k ≥ Kp_k + 1, then pr{F̄(T) ≥ p_k} ≥ 1 − δ. The proof is Hoeffding's (1956) tail comparison for the Poisson-binomial count #{|V_i| ≤ F̄⁻¹(p)}.

On that event the latent law satisfies one linear constraint with the average kernel ḡ_T(w) = K⁻¹Σ_i pr(|w + e_i| ≤ T). Propositions 1–3 apply unchanged, except that feasible locations may now form a union of intervals; only the outermost endpoints matter. This gives HetLDC:

  s = T · R^{mix}_{p_k − ε, q}(D_1/T², …, D_K/T²),

where ε bounds the error from compressing the D_i to 32 support points. [proved, modulo the numerical evaluation of R^{mix}.]

**Mean-variance plug-in** [numerical]. The common-variance shortcut R_{p,q}(D̄/T²) is not conservative in general. At D̄/T² = 0.2 with lognormal spread 1.0 it is 1.4% too short. The minimum-variance plug-in, which is conservative, is up to 14% longer than exact.

## 5. Empirical illustration

Target: pr_D{ pr(W_new ∈ C | D) ≥ 0.90 } ≥ 0.95. K = 110 calibration areas with Var W = 1 and heterogeneous known noise variances D_i = 0.577 × lognormal(0, 0.7)/mean. Conditional coverage is computed exactly from closed-form latent distribution functions; 150 replications per law (Monte Carlo s.e. of a reliability ≈ 0.018).

Table 1 (E20). Reliability / mean width.

| latent law | FH, bootstrap tolerance | noisy conformal, k = 105 | LDC, mean variance | HetLDC |
|---|---|---|---|---|
| normal | 0.960 / 3.93 | 1.000 / 4.96 | 0.993 / 4.60 | 0.980 / 4.44 |
| Laplace | 0.927 / 3.96 | 1.000 / 5.15 | 0.993 / 4.81 | 0.987 / 4.65 |
| gamma(2), centred | 1.000 / 3.95 | 1.000 / 4.97 | 1.000 / 4.60 | 1.000 / 4.44 |
| truncated Laplace | 0.847 / 3.89 | 1.000 / 4.93 | 0.980 / 4.57 | 0.967 / 4.41 |

- HetLDC meets the target for every law. It is 3.4–3.5% shorter than the mean-variance rule and about 10% shorter than noisy conformal.
- The Fay–Herriot tolerance interval is shortest, but it relies on normality and misses the target under the truncated Laplace law.
- The width HetLDC pays over it, 12–18%, is the price of a guarantee that holds for every log-concave law.

apipop finite population (E21). There are 325 school districts: 110 are used for training and 110 for calibration, with n = 2 or 3 schools sampled per district, and coverage is the share of the remaining 105 districts covered. Design variances are estimated with 1–2 degrees of freedom and plugged in as known by every rule.
- Reliability over four conditions (120 replications each): HetLDC 0.942–0.992, mean-variance LDC 0.983–1.000, Fay–Herriot tolerance 0.908–0.992 (below target in two conditions).
- HetLDC is 0.8–5.2% shorter than the mean-variance rule and 10–16% longer than Fay–Herriot.

## 6. Limits

- Log-concavity of the latent residual is assumed, not tested.
- The noise is Gaussian and independent of W, with known variances.
- Estimated variances need an outer confidence set.
- Certified values use double-precision closed forms with a 10⁻⁹ margin; interval arithmetic is not used.
- The effect sizes of the small-noise phenomenon are small. The practical gains are in the heterogeneous-noise constraint and the finite-sample procedure.
