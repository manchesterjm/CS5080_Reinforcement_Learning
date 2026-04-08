# CS 5080 - Lecture 10
**Date:** Thursday, February 26, 2026, 4:45 PM
**Topic:** Temporal Difference Learning — TD(0), SARSA, Q-Learning, Double Q-Learning (Ch. 6)

---

## Why TD Learning?

- Need RL approaches that work with **long episodes** efficiently or **tasks with no episodes** (continuing tasks)
- MC → computes G_t values (state or state-action) and uses them to update V(s) or Q(s,a)
- If an episode is **not completed**, we do not have real G_t values
- TD solves this by **bootstrapping** — using V(S') as an estimate of the remaining return instead of waiting for the actual return G_t

**Connection to MC (S&B Eq. 6.1):** Constant-α MC update is V(S_t) ← V(S_t) + α[G_t - V(S_t)]. TD replaces G_t with the bootstrap estimate R_{t+1} + γV(S_{t+1}).

---

## TD(0) Algorithm (Tabular, estimating v_π)

**Input:** policy π to be evaluated
**Parameter:** step size α ∈ (0, 1]
**Initialize:** V(s) for all s ∈ S⁺, V(terminal) = 0

```
Loop for each episode:
    Initialize S
    Loop for each step of episode:
        A ← action given by π for S
        Take action A, observe R, S'
        V(S) ← V(S) + α[R + γV(S') - V(S)]       ... (S&B Eq. 6.2)
        S ← S'
    until S is terminal
```

### TD(0) Update Explained

Initialize vector of V(s) values to 0 for all states s₁...s_n. Terminal state has value 0.

**Update rule breakdown:**
- V_new(s) ← part from V_old(s) + part from current actions
- V_new(s) ← (1-α)·V_old(s) + α·(bootstrap estimate of return)
- V_new(s) ← (1-α)·V_old(s) + α·[r + γ·V(s')]

**Rearranged (S&B Eq. 6.2):**
V(S) ← V(S) + α·[R + γ·V(S') - V(S)]

### TD Error (S&B Eq. 6.5)

δ_t = R_{t+1} + γ·V(S_{t+1}) - V(S_t)

The bracketed term in the update is the **TD error** — the difference between the bootstrap target [R + γV(S')] and the current estimate V(S).

**Note:** The MC error G_t - V(S_t) can be decomposed as a sum of TD errors: Σ_{k=t}^{T-1} γ^{k-t} · δ_k (S&B Eq. 6.6). This connects MC and TD mathematically.

**α (alpha) = learning rate:**
- If α = 0.1 → 90% V_old + 10% new estimate
- Controls how much to trust the new observation vs. existing estimate

---

## SARSA (On-Policy TD Control, estimating Q ≈ q_*) — S&B Eq. 6.7

Uses **Q(s,a) table** (states × actions).

Two versions shown:
1. **Evaluation only** — if we just want to evaluate or obtain Q(s,a) values
2. **Control** — with ε-greedy exploration

```
Initialize Q(s,a) for all s ∈ S⁺, a ∈ A(s), Q(terminal, ·) = 0

Loop for each episode:
    Initialize S
    Choose A from S using policy derived from Q (e.g., ε-greedy)
    Loop for each step of episode:
        Take action A, observe R, S'
        Choose A' from S' using policy derived from Q (e.g., ε-greedy)
        Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]    ... (S&B Eq. 6.7)
        S ← S'; A ← A'
    until S is terminal
```

To make it a control algorithm: introduce exploration using **ε-greedy** approach (choose non-optimal actions with non-zero probability to explore).

**Name origin:** The update uses the quintuple (S, A, R, S', A') — SARSA.

---

## Q-Learning (Off-Policy TD Control, Watkins 1989) — S&B Eq. 6.8

**Off-policy** since the learned/update policy is different from the behavior policy.

- Behavior policy: ε-greedy (minor variation of Q)
- Update policy: always uses **best action** (greedy) — very different from behavior
- **No importance sampling needed** — the max operator implicitly uses the greedy (target) policy

**Diagram:** From state S, take real action a, get real reward r, arrive at S'. At S', make an **optimistic prediction** — pick the best action.

```
Q(S,A) ← Q(S,A) + α[R + γ·max_a Q(S',a) - Q(S,A)]    ... (S&B Eq. 6.8)
```

The key difference from SARSA: uses **max_a Q(S',a)** instead of Q(S',A').

- `R` = reward for taking action A in state S
- `γ · max_a Q(S',a)` = value for the best action that can be performed in state S'
- The bracket [...] is the **TD error**

**Problem:** Can quickly start to **overestimate** values (maximization bias).

**Not covered in lecture but in textbook — Expected SARSA (S&B Eq. 6.9):**
Uses Σ_a π(a|S') · Q(S',a) instead of Q(S',A') or max_a Q(S',a). Generalizes both SARSA and Q-learning — when π is greedy, Expected SARSA = Q-learning.

---

## Double Q-Learning (S&B Eq. 6.10)

**Problem:** Single Q-table leads to **positive bias** when estimating maximum of random values (shown in statistics generally).

**Formal example (S&B p.134):** Consider state s where all true values q(s,a) = 0, but estimates Q(s,a) are uncertain (some above, some below zero). The maximum of the estimates will be positive — a positive bias — even though the true maximum is zero.

**Solution:** Use **two Q tables** — Q₁(s,a) and Q₂(s,a)

```
Loop for each episode:
    Initialize S
    Loop for each step of episode:
        Choose A from S using ε-greedy in Q₁ + Q₂
        Take action A, observe R, S'
        With 0.5 probability:
            Q₁(S,A) ← Q₁(S,A) + α(R + γQ₂(S', argmax_a Q₁(S',a)) - Q₁(S,A))
        else:
            Q₂(S,A) ← Q₂(S,A) + α(R + γQ₁(S', argmax_a Q₂(S',a)) - Q₂(S,A))
        S ← S'
    until S is terminal
```

**Key insight:** One table selects the action (argmax), the other evaluates it. Using more than one estimator reduces maximization bias. Both Q₁ and Q₂ individually converge to q_*. The behavior policy uses the sum (or average) of the two tables for ε-greedy action selection.

---

**Reading:** Sutton & Barto Ch. 6
**Handout:** 305-TD0Algorithms.pdf (algorithm boxes)
