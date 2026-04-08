# CS 5080 - Lecture 9
**Date:** Tuesday, February 24, 2026, 4:43 PM
**Topic:** Off-Policy RL — Importance Sampling (Ch. 5.5 continuation)

---

## Off-Policy RL Algorithm

- **Behavior policy b:** The policy that acts (generates episodes)
- **Target policy π:** The policy being learned/evaluated
- Transfer learning from b → π requires **importance sampling**
- **Coverage assumption:** For IS to work, π(a|s) > 0 must imply b(a|s) > 0 — the behavior policy must cover all actions the target policy might take

### Episode Structure

Episodes are lined up sequentially in continuous time:
- Episode 1 ends at T₁
- Episode i starts at T_{i-1}+1, ends at Tᵢ
- Last episode N ends at T_N

ρ = importance sampling ratio (rho)

τ(s) = TAU = set of time points in which state s occurs in behavior b

### Importance Sampling Ratio (S&B Eq. 5.3)

ρ_{t:T-1} = ∏_{k=t}^{T-1} π(A_k | S_k) / b(A_k | S_k)

The ratio runs from time t to T-1 (one step before terminal). Each W_n in the incremental formulas below represents one full trajectory ratio ρ_{t:T(t)-1}.

### Ordinary Importance Sampling (S&B Eq. 5.5)

V_π(s) = Σ_{t ∈ τ(s)} ρ_t · G_t / |τ(s)|

Where:
- τ(s) = set of time points in which state s occurs in behavior b
- G_t = return at state s (if learning V_π(s))
- Or G_t = return for (s,a) pair (if learning q_π(s,a))
- **Unbiased** but can have **infinite variance**

### Weighted Importance Sampling (S&B Eq. 5.6)

V_π(s) = Σ_{t ∈ τ(s)} ρ_t · G_t / Σ_{t ∈ τ(s)} ρ_t

- **Biased** (bias converges to zero) but always has **finite variance** (assuming bounded returns)
- Generally preferred over ordinary IS in practice

*Looking at Section 5.6 — Incremental Implementation*

---

## Incremental Update Derivation (Weighted IS)

**Goal:** Compute V_{n+1}(s) given V_n(s) and G_t of the last occurrence

Given n-1 occurrences of state s so far:

**Equation (1):**
V_n(s) = Σ_{k=1}^{n-1} W_k · G_k / Σ_{k=1}^{n-1} W_k

**Equation (2):**
V_{n+1}(s) = Σ_{k=1}^{n} W_k · G_k / Σ_{k=1}^{n} W_k

### Computing V_{n+1}(s) - V_n(s)

Subtract (1) from (2), find common denominator:

V_{n+1}(s) - V_n(s) = [(Σ^n W_k G_k)(Σ^{n-1} W_k) - (Σ^{n-1} W_k G_k)(Σ^n W_k)] / [(Σ^n W_k)(Σ^{n-1} W_k)]

After expanding (split Σ^n into Σ^{n-1} + W_n terms) and canceling:

= W_n · G_n · Σ^{n-1} W_k - W_n · Σ^{n-1} W_k G_k / [(Σ^n W_k)(Σ^{n-1} W_k)]

Factor out W_n:

**Result:**
V_{n+1}(s) - V_n(s) = W_n / C_n · [G_n - V_n(s)]

Where C_n = Σ_{k=1}^{n} W_k (cumulative importance weights)
- Recursive update: C_{n+1} = C_n + W_{n+1}, with C_0 = 0
- V_1 is arbitrary (since C_0 = 0, the first update effectively replaces V_1)

**Final incremental update formula (S&B Eq. 5.8):**
V_{n+1}(s) = V_n(s) + W_n / C_n · [G_n - V_n(s)]

**Note:** The textbook uses W_n (not ρ_n) as the weight notation in this formula. Each W_n represents the full trajectory importance sampling ratio for the nth return. The professor's handwritten notes (303-WeightedImportanceSamplingFormula.pdf) use ρ_n, which is equivalent.

---

**Reading:** Sutton & Barto Ch. 5.5-5.6
