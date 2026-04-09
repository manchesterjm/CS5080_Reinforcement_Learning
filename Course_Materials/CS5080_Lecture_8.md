# CS 5080 — Lecture 8
**Date:** Thursday, February 12, 2026, 4:39 PM
**Topic:** Off-Policy Learning (continued) — Importance Sampling
**Reading:** Sutton & Barto Ch. 5.5–5.6

---

## Off-Policy Learning (cont.)

Off-policy learning — at least two cases:
- Agent is acting using **B** (behavior policy), trying to learn **π** (target policy)
- Someone else uses B well; agent learns π from B

Two policies:
- **Behavior policy** — B (on-policy)
- **Target policy** — π

### Behavior Policy B
- Generates a lot of episodes of training: ep₁ → ep_n
- Can be large amounts of episodes
- Can learn discounted return at any time t: G_{t,B} = discounted return

### Target Policy π
- Performs actions here ↑ but learns values here →
- Q_B(s, a) ⇒? Q_π(s, a)

**Question:** Compute V_B(s) and Q_B(s, a)
- From V_B(s) —learn→ V_π(s)
- From Q_B(s, a) —learn→ Q_π(s, a)
  - Acting here → Learning this

Computing G_t or Return values is necessary for learning V(s) or Q(s, a) values.

**Key assumption:** Agent assumes the episode's state-action sequence is exactly the same in B and π.

---

## The Importance Sampling Ratio

    G_{t,pi} = rho_t * G_{t,B}

This ρ is the **scaling/weighted factor** — the **importance sampling ratio**.

> **[From Sutton & Barto, Section 5.5, p. 104]:** The ratio ρ_{t:T-1} is formally called the **importance-sampling ratio**. It is defined as "the relative probability of the trajectory under the target and behavior policies, depending only on the two policies and the sequence, not on the MDP." The professor was unsure of the exact term in lecture — Sutton & Barto consistently uses "importance-sampling ratio" throughout Chapter 5.5.
>
> The key insight: because the state transition probabilities p(s'|s, a) appear in both the numerator and denominator, they cancel out, leaving only the ratio of the action probabilities under each policy.

---

## Deriving ρ — Why It Works

When estimating values, we might have:
- **Distribution 1** — e.g., rainfall in DT (downtown) Colorado Springs
- **Distribution 2** — e.g., rainfall in N (north) Colorado Springs

We take original values in one distribution and multiply by a weighted factor to estimate another distribution.

### Formal Definition

    V_pi(s) = E[G_{t,pi} | S_{t,pi} = s]

    = E[rho_t * G_{t,B} | S_t = s]

(where S_{t,π} = S_{t,B} = s, or just S_t = s)

**Scale B to get π.**

**What is the value of ρ?** The weight or the importance sampling factor for transfer from G_t values from B policy to target policy.

    rho_t = Probability of states and actions in target policy pi in the imaginary ep. from time t to end time T
            / Probability of states and actions in behavior policy B in the real ep. from t to T

### Expanding the Ratio

    rho_t = Prob(a_{t,pi} | S_{t,pi}) * Prob(a_{t+1,pi} | S_{t+1,pi}) * ...
            / Prob(a_{t,B} | S_{t,B}) * Prob(a_{t+1,B} | S_{t+1,B}) * ...

    = pi(a_t | S_t) * pi(a_{t+1} | S_{t+1}) * ... * pi(a_{T-1} | S_{T-1})
      / B(a_t | S_t) * B(a_{t+1} | S_{t+1}) * ... * B(a_{T-1} | S_{T-1})

### Product Form

    rho_{t:T-1} = product_{k=t}^{T-1} pi(A_k | S_k) / B(A_k | S_k)

    G_{t,pi} = rho_t * G_{t,B}   (importance sampling ratio)

**We now have a formula to transfer G_{t,B} to G_{t,π}, or transfer values from B to π.**

---

## Computing V_π(s) — Two Methods

**Next question:** To actually transfer/compute V_π(s), i.e., values of states in target policy (or Q_π(s, a), i.e., values of state-action pairs in target policy):

    V_pi(s) = E[rho_t * G_{t,B} | S_t = s]

(ρ_t is the transfer/importance ratio or weight)

From the book, two ways to compute V_π(s):
1. **Ordinary Sampling**
2. **Weighted Sampling**

### Ordinary Sampling

Define **τ(s)** = the set of time points at which state s occurs in episodes of training using behavior policy B. (τ = tau)

Professor draws the standard 2x3 maze.

Then |τ(s)| = number of times state s occurs in the episodes of training using behavior policy B.

    V_pi(s) = sum_{t in tau(s)} rho_t * G_{t,B} / |tau(s)|

**Example:** Suppose we want to compute V_π(s₂):

    V_pi(s_2) = [G_t | S_t = s_2] * rho

(Scaled by importance ratio)

### Weighted Sampling

    V_pi(s) = sum_{t in tau(s)} rho_t * G_{t,B} / sum_{t in tau(s)} rho_t

---

## Section 5.6 — Incremental Implementation for MC Algorithms

We did not go over this but we should know it. After presentations next week.

---

*Original handwritten notes: `CS_5080_Lecture_8.pdf`*
*Transcribed: 2026-02-16*
