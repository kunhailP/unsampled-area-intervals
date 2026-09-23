# Appendix draft: proofs

Status labels: [complete], [sketch], [gap]. Notation as in `MANUSCRIPT_CORE.md`: g_x(w) = pr(|w + x^{1/2}Z| ≤ 1), Φ and φ are the standard normal distribution and density functions, and LC is the class of log-concave laws on the real line (including Dirac masses).

## A.1 Proposition 1 (reduction to Diracs and log-affine segments) [sketch]

Fix s > 0. R_{p,q}(x) ≤ s if and only if

  inf{ μ(−s, s) : μ ∈ LC, ∫ g_x dμ ≥ p } ≥ q.   (A1)

Fix M > s + 1. On LC_M, the log-concave laws supported in [−M, M], the constraint ∫ g_x dμ ≥ p is given by a continuous function. The map μ ↦ −μ(−s, s) is linear and upper semicontinuous for weak convergence, because (−s, s) is open. By Fradelizi & Guédon (2004, Theorem 2), the infimum over the constrained set is attained at an extreme point of its convex hull. Such a point is either a Dirac mass or a law with density proportional to e^{βw} on a segment [a, b] ⊂ [−M, M] on which the constraint holds.

Passing M → ∞ [gap to write out]. Let μ ∈ LC be feasible, and let μ_M be μ conditioned on [−M, M], which is log-concave. Then |∫ g_x dμ_M − ∫ g_x dμ| ≤ 2μ(ℝ∖[−M, M]) → 0, and μ_M(−s, s) → μ(−s, s). So the infimum in (A1) over LC is the limit of infima over LC_M at constraint level p − ε_M. Right-continuity of the extremal-family value in p then closes the argument; this still has to be written. The family found does not depend on s, so R equals the supremum of Q_q(|W|) over that family.

## A.2 Proposition 2 (closed form) [complete]

For β ≠ 0, integration by parts with d/dw e^{βw}/β = e^{βw} and d/dw Φ((c − w)/σ) = −φ((c − w)/σ)/σ gives

  ∫_a^b e^{βw}Φ((c−w)/σ)dw = β⁻¹[e^{βw}Φ((c−w)/σ)]_a^b + (βσ)⁻¹∫_a^b e^{βw}φ((c−w)/σ)dw.

The last integral equals σe^{βc+β²σ²/2}{Φ((b−c−βσ²)/σ) − Φ((a−c−βσ²)/σ)}, by completing the square. The case β = 0 follows from ∫Φ = uΦ(u) + φ(u). Negative β reduces to positive β under W ↦ −W.

## A.3 Proposition 3 (two endpoints suffice) [complete]

Write W_c = c + Y, where Y has a fixed log-affine density on [0, ℓ].
- c ↦ pr(|W_c + e| ≤ 1) is the convolution of the log-concave density of −(Y + e) with the indicator of [−1, 1]. It is therefore log-concave in c (Prékopa, 1973), and {c : pr(|W_c + e| ≤ 1) ≥ p} is an interval [c₁, c₂].
- For every s, c ↦ pr(|W_c| ≤ s) is log-concave by the same argument. So {c : Q_q(|W_c|) ≤ s} = {c : pr(|W_c| ≤ s) ≥ q} is an interval, and Q_q(|W_c|) is quasi-convex in c.
- A quasi-convex function on an interval attains its maximum at an endpoint, so the maximum is at c₁ or c₂.

With heterogeneous noise the first map becomes an average of log-concave functions. It need not be log-concave, and the feasible set can be a union of intervals S. Quasi-convexity on the whole line still gives max_S Q ≤ max{Q(inf S), Q(sup S)}, and both of these points belong to the closure of S.

## A.4 Theorem 4 (small-noise lower bound) [complete for the one-sided construction; transfer sketch]

One-sided family. Let Y = b − E with E exponential with rate u > 0, and let Z ~ N(0, 1) be independent of E. Then

  pr(Y + Z ≤ 0) = pr(E ≥ b + Z) = Φ(−b) + E{e^{−u(b+Z)}1(Z > −b)} = Φ(−b) + e^{−ub+u²/2}Φ(b−u).

Its derivative in b is −u e^{−ub+u²/2}Φ(b−u) < 0. So for each u there is a unique b_u at which this equals q. For Y, pr(Y ≤ y) = e^{−u(b−y)}, so F_Y⁻¹(q) = b − log(1/q)/u. Substituting the constraint gives F_Y⁻¹(q) = u⁻¹ log{e^{u²/2}Φ(b_u − u) + e^{ub_u}Φ(−b_u)}.

As u → 0 we have ub_u → log(1/q), so Φ(−b_u) and Φ(u − b_u) are o(u^k) for every k. The logarithm is then u²/2 + o(u²), and F_Y⁻¹(q) = u/2 + o(u) > 0. Hence c_q > 0.

Transfer [sketch]. Put W = 1 + x^{1/2}Y and condition on W ≥ −1/2. This keeps W log-concave, and it changes the mass, the noisy constraint and the quantile by amounts exponentially small in 1/x. The endpoint −1 lies 2x^{−1/2} standard deviations away, so its contribution is also exponentially small. Hence R_{q,q}(x) ≥ 1 + c_q x^{1/2} − o(x^{1/2}).

## A.5 Theorem 5 (matching upper bound) [skeleton; gaps (i) and (ii)]

Steps 1 to 7 are as in `THEORY_NOTE.md` §2:
1. Symmetry. It suffices to consider β ≥ 0.
2. Convexity of F on (−∞, b]. By Jensen the left endpoint loses mass, up to an exponentially small correction τ.
3. Condition on W ≥ −1, which moves the level to q′ = q/(1−m) ≥ q.
4. Q_q(|W|) ≤ s₀, where F̃(s₀) = q′.
5. By scaling, (s₀ − 1)/x^{1/2} ≤ c^{(M)}_{q′−τ′, q′}, the one-sided value with support restricted to [−M, ∞), M = 2x^{−1/2}. The restriction must be kept: without it c_{p,q} = +∞ for every p < q (exponential tails with u → 0).
6. Left truncation shows that q ↦ c_{q,q} is nonincreasing, and moves the constraint slack τ′ to τ″.
7. Absorb the slack by a left shift of size about τ″M/(1 − q). A log-concave law with support of length at most M has density at least about min(u, 1−u)/M at its interior u-quantiles. Since τ″ is exponentially small, the shift is o(1).

Gaps:
- (i) The correction τ in step 2 must be uniformly small. Its bound grows with the density at b, so it fails when β exceeds about e^{1/x}. Such laws are within tiny total variation of a point mass, and a point mass has value below 1.
- (ii) (narrowed) The regime q′ → 1 is handled separately.
  - If q′ ≥ 1 − τ^{1/3}: for β ≥ 0 the density on [1, s₀] is at least the average density on [−1, 1], F̃(1)/2 ≥ (2q′ − 1)/2 − O(τ′) (assuming q > 1/2), and pr(e > 1 − w) ≥ 1/2 for w ≥ 1. So F̃(s₀) − F̃(1) ≤ 1 − q′ + 2τ′, and s₀ − 1 ≤ (τ^{1/3} + 2τ′)/((2q − 1)/2), which is exponentially small.
  - Otherwise τ″ = O(τ^{2/3}).
  - What remains is to state, with constants, the lower bound on the interior-quantile density of log-concave laws with bounded support, used in step 7. An earlier draft invoked continuity of the unrestricted c_{p,q} in p; that is false, since the unrestricted value is infinite for p < q.

## A.6 Proposition 7 (order statistic under heterogeneous noise) [complete]

Let V_i be independent, and let F̄ = K⁻¹Σ_i F_i be the average distribution function of |V_i|, assumed continuous. Let T = |V|_(k) and t_p = F̄⁻¹(p). Then F̄(T) < p exactly when T < t_p, that is, when N := #{i : |V_i| < t_p} ≥ k. N is a sum of independent Bernoulli variables with success probabilities F_i(t_p), whose mean is F̄(t_p) = p. By Hoeffding (1956; Ann. Math. Statist. 27, 713–721 — theorem number to be checked), pr(N ≥ c) ≤ pr{Bin(K, p) ≥ c} for every integer c ≥ Kp + 1. With c = k, and p = p_k the δ-quantile of Beta(k, K+1−k), pr{Bin(K, p_k) ≥ k} = pr{Beta(k, K+1−k) ≤ p_k} = δ. Hence pr{F̄(T) ≥ p_k} ≥ 1 − δ, provided k ≥ Kp_k + 1. At K = 110, k = 105 we have Kp_k + 1 = 100.4 ≤ 105.

On this event ∫ ḡ_T dG ≥ p_k, where ḡ_T(w) = K⁻¹Σ_i pr(|w + e_i| ≤ T). Every log-concave G satisfying the constraint then has Q_q(|W|) ≤ T·R^{mix}_{p_k,q}(D/T²), by the definition of R^{mix}. Replacing ḡ_T by a compressed kernel ḡ′ with sup|ḡ_T − ḡ′| ≤ ε and p_k by p_k − ε preserves the implication.
