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

## A.4 Lemmas A and B (the one-sided problem) [complete]

Exponential tails. Let Y_{u,b} = b − E, where E is exponential with rate u > 0 and independent of Z ~ N(0, 1). Then

  κ_u(b) := pr(Y_{u,b} + Z ≤ 0) = pr(E ≥ b + Z) = Φ(−b) + e^{−ub+u²/2}Φ(b − u).

Its derivative in b is −u e^{−ub+u²/2}Φ(b − u) < 0. So κ_u(b) = p has a unique root b_u(p), which is continuous in p. Since pr(Y_{u,b} ≤ y) = min(e^{−u(b−y)}, 1), the q-quantile is b − log(1/q)/u. Write v_u(p, q) = b_u(p) − log(1/q)/u.

Lemma A. c_{p,q} = sup_{u>0} v_u(p, q).

Proof.
- Exponential tails are log-concave, which gives ≥.
- Conversely, let Y be log-concave with pr(Y + Z ≤ 0) ≥ p. If Y = d is a point mass, then Φ(−d) ≥ p, so d ≤ −Φ⁻¹(p) = lim_{u→∞} v_u(p, q).
- Otherwise F = F_Y is continuous and log F is concave (Prékopa, 1973). Let v = F⁻¹(q) and G(y) = F(v + y), so G(0) = q.
- The point 0 is interior to {G > 0}. Hence the right derivative u of log G at 0 is a supergradient: log G(y) ≤ log q + uy for all y. Also u ≥ 0 because G is nondecreasing. If u = 0 then G ≤ q < 1, which is impossible, so u > 0.
- Hence G ≤ G_u := min(q e^{uy}, 1), the distribution function of Y_{u,b₀} with b₀ = log(1/q)/u, and

  p ≤ E G(−v − Z) ≤ E G_u(−v − Z) = κ_u(v + b₀).

- As κ_u is decreasing, v + b₀ ≤ b_u(p), that is, v ≤ v_u(p, q). □

Lemma B. q ↦ c_q is nonincreasing.

Proof. Let q′ > q. Let Y be feasible for (q′, q′) with v = F_Y⁻¹(q′), and set m = (q′ − q)/(1 − q). Then Y* = Y | {Y ≥ F_Y⁻¹(m)} is log-concave, and

  pr(Y* + Z ≤ 0) ≥ (q′ − m)/(1 − m) = q,  F_{Y*}(v) = q.

A point mass has value −Φ⁻¹(q′) < −Φ⁻¹(q) ≤ c_q. □

Small-u expansion. As u → 0, ub_u → log(1/q), and v_u(q, q) = u/2 + o(u). Hence c_q > 0.

## A.5 Theorems 5 and 5+ (upper bound for every x) [complete]

Theorem 5. Let W be log-concave, e ~ N(0, x) independent of W, and pr(|W + e| ≤ 1) ≥ q. Put s = 1 + c_q x^{1/2}.
- Let α_R = pr(W + e > 1) and α_L = pr(W + e < −1); then α_L + α_R ≤ 1 − q.
- Y = (W − 1)/x^{1/2} is log-concave, with pr(Y + Z ≤ 0) = 1 − α_R ≥ q. By the definition of c and Lemma B, F_Y⁻¹(1 − α_R) ≤ c_{1−α_R} ≤ c_q. So pr(W ≤ s) ≥ 1 − α_R.
- Applying this to −W, and using that −e has the same law as e, gives pr(W < −s) ≤ α_L.
- So pr(|W| ≤ s) ≥ 1 − α_L − α_R ≥ q, and Q_q(|W|) ≤ s. □

Theorem 5+. Now let the noisy level be p ≥ q, so α_L + α_R ≤ 1 − p. For any β_L + β_R = 1 − q, the one-sided step at levels (1 − α, 1 − β) gives pr(W > 1 + x^{1/2}c_{1−α_R,1−β_R}) ≤ β_R, and symmetrically on the left. Optimising over β, and using that c_{p′,q′} decreases in p′ (so that the worst case has α_L + α_R = 1 − p), gives R_{p,q}(x) ≤ 1 + C_{p,q}x^{1/2}. □

## A.6 Theorem 4 (lower bound) [complete]

Fix ε > 0 and choose u with v_u(q, q) > c_q − ε. On p′ ∈ [q, (1 + q)/2], b_u(p′) ≥ b_min > −∞.

Let Y = Y_{u,b_u(p′)} and W = 1 + x^{1/2}Y, which is log-concave. With t = 2x^{−1/2},

  pr(|W + e| ≤ 1) = p′ − pr(Y + Z < −t) ≥ p′ − η_x,  η_x = e^{−u(t+b_min)/2} + Φ(−(t + b_min)/2).

Take p′ = q + η_x, so the constraint holds.

Since pr(|W| ≤ s) ≤ F_W(s) and F_W is strictly increasing near its q-quantile, Q_q(|W|) ≥ 1 + x^{1/2}v_u(q + η_x, q). Now η_x → 0 exponentially, and v_u is continuous in p. Hence liminf_{x→0}(R_{q,q}(x) − 1)/x^{1/2} ≥ c_q − ε. □

## A.7 Proposition 7 (order statistic under heterogeneous noise) [complete]

Let V_i be independent, and let F̄ = K⁻¹Σ_i F_i be the average distribution function of |V_i|, assumed continuous. Let T = |V|_(k) and t_p = F̄⁻¹(p). Then F̄(T) < p exactly when T < t_p, that is, when N := #{i : |V_i| < t_p} ≥ k. N is a sum of independent Bernoulli variables with success probabilities F_i(t_p), whose mean is F̄(t_p) = p. By Hoeffding (1956; Ann. Math. Statist. 27, 713–721 — theorem number to be checked), pr(N ≥ c) ≤ pr{Bin(K, p) ≥ c} for every integer c ≥ Kp + 1. With c = k, and p = p_k the δ-quantile of Beta(k, K+1−k), pr{Bin(K, p_k) ≥ k} = pr{Beta(k, K+1−k) ≤ p_k} = δ. Hence pr{F̄(T) ≥ p_k} ≥ 1 − δ, provided k ≥ Kp_k + 1. At K = 110, k = 105 we have Kp_k + 1 = 100.4 ≤ 105.

On this event ∫ ḡ_T dG ≥ p_k, where ḡ_T(w) = K⁻¹Σ_i pr(|w + e_i| ≤ T). Every log-concave G satisfying the constraint then has Q_q(|W|) ≤ T·R^{mix}_{p_k,q}(D/T²), by the definition of R^{mix}. Replacing ḡ_T by a compressed kernel ḡ′ with sup|ḡ_T − ḡ′| ≤ ε and p_k by p_k − ε preserves the implication.
