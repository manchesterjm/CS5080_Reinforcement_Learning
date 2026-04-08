# CS 5080 Lecture 7 - Monte Carlo Methods (Continued)

**Date:** Tuesday, February 10, 2026, 4:46 PM
**Topic:** Chapter 5 (cont.) - MC Exploring Starts, ε-Greedy, Off-Policy
**Reading:** Sutton & Barto Ch. 5, Sections 5.3–5.5

---

## Review: Learning Values of States

Learn values of states in our environment.

**Example:** Professor draws a 2×3 maze to illustrate.

**Notation:**
- \* = optimal or near-optimal
- π* = the optimal policy
- v(s) ∀s = value of each state

---

## Section 5.3: Monte Carlo ES for Estimating π ≈ π*

### First-Visit MC Prediction (Review)

For estimating V ≈ V_π:
- Perform episodes of training
- Large number of episodes
- Compute returns in reverse order
- Looking for **unique states only** (first-visit)

**Goal:** π is an equi-random policy (say). What is the best action in each state? Can we learn a better policy over time? Make a small change, and see if we improve Q(s, a) values.

### Monte Carlo ES: Evaluation and Improvement

Two phases:
- **Evaluation**
- **Improvement**

$$\pi_0 \xrightarrow{\text{eval}} Q(s,a) \xrightarrow{\text{improve}} \pi_1 \quad \text{repeat until} \rightarrow \pi_*$$

**Key property:** First state can be **any state in our state space** (exploring starts).

---

## Section 5.4: MC Without Exploring Starts (On-Policy ε-Soft Methods)

> **📚 From Sutton & Barto (p. 101):**
>
> ```
> On-policy first-visit MC control (for ε-soft policies),
> for estimating π ≈ π*
>
> Algorithm parameter: small ε > 0
> Initialize:
>     π ← an arbitrary ε-soft policy
>     Q(s, a) ∈ ℝ (arbitrarily), for all s ∈ S, a ∈ A(s)
>     Returns(s, a) ← empty list, for all s ∈ S, a ∈ A(s)
>
> Loop forever (for each episode):
>     Generate an episode following π: S₀, A₀, R₁, S₁, A₁, R₂, ..., S_{T-1}, A_{T-1}, R_T
>     G ← 0
>     Loop for each step of episode, t = T-1, T-2, ..., 0:
>         G ← γG + R_{t+1}
>         Unless the pair S_t, A_t appears in S₀, A₀, S₁, A₁, ..., S_{t-1}, A_{t-1}:
>             Append G to Returns(S_t, A_t)
>             Q(S_t, A_t) ← average(Returns(S_t, A_t))
>             A* ← argmax_a Q(S_t, a)  (with ties broken arbitrarily)
>             For all a ∈ A(S_t):
>                 π(a|S_t) ← { 1 - ε + ε/|A(S_t)|  if a = A*
>                             { ε/|A(S_t)|            if a ≠ A*
> ```

### On-Policy First-Visit MC Control

**Goal:** Estimate π ≈ π* **without** requiring exploring starts.

### Exploitation vs Exploration

| Strategy        | Definition                          |
| --------------- | ----------------------------------- |
| **Exploitation** | Do best all the time — pick the best action in state all the time |
| **Exploration**  | Not all the time — try non-best actions sometimes               |

### The ε-Greedy Algorithm

**Algorithm parameter:** small ε > 0

Let there be A(s) = actions possible in state s.

**Example:** A(s) = {↑, ↓, ←, →, ?, ?} so |A(s)| = number of actions possible in state s.

```
         non-best
           ↗
    (S) ──→ a*  (current best action)
           ↘
         non-best
```

**Given a small ε > 0:**
- Every action is possible with **at least** probability:

$$\frac{\varepsilon}{|A(s)|}$$

**Example:** ε = 0.10, |A(s)| = 5

$$\text{Every action is possible with prob at least } = \frac{0.10}{5} = 0.02$$

**Probability assigned to a\* (the best action):**

$$\pi(a^*|s) = 1 - \varepsilon + \frac{\varepsilon}{|A(s)|}$$

**Example:** With ε = 0.10, |A(s)| = 5:
- Best action probability: 1 - 0.10 + 0.10/5 = 0.92
- Each non-best action: 0.10/5 = 0.02
- Check: 0.92 + 4(0.02) = 1.0 ✓

---

## Section 5.5: Off-Policy Prediction

### On-Policy vs Off-Policy Learning

| Method          | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| **On-policy**   | Starts with policy π, improves repeatedly towards π* (the best policy) |
| **Off-policy**  | Uses two separate policies                                       |

### Off-Policy: Two Policies

1. **Target policy π** — the policy we want to learn/optimize
2. **Behavior policy B** — the policy that actually generates experience

**Goal:** Learn to estimate π* by acting using policy B.

### Off-Policy Trajectory

```
Policy B generates experience:

    S₀,B ──A₀,B──→ R₁,B ──→ ... ──→ S_t,B ──A_t,B──→ ... ──→ G_T,B
                                        ↑
                         Compute G_t,B for Q(S_t,B, A_t,B)

Policy π: G_t,B from B gets inserted here to obtain G_t,π
          (check with book here — professor's note)
```

**Intuition:** The agent learns from an expert (behavior policy B) that already has knowledge. Can the agent learn from the expert?

### Off-Policy Properties

| Property     | Detail                                                    |
| ------------ | --------------------------------------------------------- |
| **Variance** | Off-policy has **high variance**                          |
| **Cost**     | Needs another environment to learn in, or an expert (can cost a lot $$$) |
| **Generality** | Off-policy learning is **general**                      |
| **Relationship** | On-policy learning is the special case where **B = π** |

---

## Key Takeaways

| Section | Method          | Key Idea                                                |
| ------- | --------------- | ------------------------------------------------------- |
| 5.3     | MC ES           | Exploring starts guarantees coverage; evaluation → improvement loop |
| 5.4     | ε-Greedy (On-Policy) | No exploring starts needed; ε ensures exploration  |
| 5.5     | Off-Policy      | Learn target policy π from behavior policy B            |

---

*Transcribed from handwritten class notes. On-policy MC control algorithm inserted from Sutton & Barto Chapter 5, p. 101 (noted in class for later addition).*
