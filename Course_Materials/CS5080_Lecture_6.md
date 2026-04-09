# CS 5080 Lecture 6 - Monte Carlo Methods

**Date:** Thursday, February 5, 2026, 4:38 PM
**Topic:** Chapter 5 - Monte Carlo Methods
**Reading:** Sutton & Barto Ch. 5

---

## MDP Dynamics Review

### The Four-Argument Function

Every detail of the dynamics of an MDP is captured by:

    p(s', r | s, a)

This is a **4-argument function** describing:

```
        a (action)
    ┌───────────────┐
    │               ↓
   (s) ──────────→ (s')
         r (reward)
```

**If all these values are known**, we have a **completely specified MDP system**.

From state s:
- Try right, try down, etc.
- Probability of each action
- Rewards for each action

---

## The Reality of RL

### Incomplete Knowledge

**In a realistic RL task, all values are not always possible.**

- The agent knows **some things but not all**
- The agent will learn, even without full knowledge, by interacting:
  1. **With the environment** (real interaction)
  2. **And/or simulating** interacting with the environment

### Random Exploration

**This means that actions will be random often in the beginning.**

**Approach:** Monte Carlo methods to estimate values

> Learn good stuff / useful stuff by **doing a lot of random things**

---

## Monte Carlo Estimation

### Episode Structure

```
    S₀ ──→ ... ──→ S_t ──A_t──→ R_{t+1} ──→ ... ──→ G_e (terminal)
```

### Return Calculation

    G_t = return

    = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+3} + ...

    = R_{t+1} + gamma * G_{t+1}

---

## First-Visit MC Prediction

**For estimating V ≈ V_π**

> **📚 From Sutton & Barto (p. 92):**
>
> ```
> First-visit MC prediction, for estimating V ≈ v_π
>
> Input: a policy π to be evaluated
> Initialize:
>     V(s) ∈ ℝ, arbitrarily, for all s ∈ S
>     Returns(s) ← an empty list, for all s ∈ S
>
> Loop forever (for each episode):
>     Generate an episode following π: S₀, A₀, R₁, S₁, A₁, R₂, ..., S_{T-1}, A_{T-1}, R_T
>     G ← 0
>     Loop for each step of episode, t = T-1, T-2, ..., 0:
>         G ← γG + R_{t+1}
>         Unless S_t appears in S₀, S₁, ..., S_{t-1}:
>             Append G to Returns(S_t)
>             V(S_t) ← average(Returns(S_t))
> ```

### Key Points

- **Updates value for state at first visit only**
- First visit and every visit will converge as episodes → ∞

---

## From V(s) to Action Selection

Given V(s) values, we can obtain which actions to perform.

**For state S, perform action:**

    a = argmax_{x'} [r + V(x')]   where x' = delta(s, a)

### Why Q(s,a) is Often Preferred

**Many RL algorithms compute Q(s, a) values instead of V(s) values**

Reason: Q-values directly tell us which action to take without needing the transition model.

---

## Action-Value Function Q(s, a)

### Episode with Actions

```
    S₀ ──→ ... ──→ S_t ──(A_t)──→ S_{t+1} ──→ ...
                         ↓
                       R_{t+1}

    Q(s, a) = expected return from taking action a in state s
```

---

## Monte Carlo Control

### Generalized Policy Iteration (GPI)

> **📚 From Sutton & Barto (p. 97):**

    pi_0 --E--> Q_{pi_0} --I--> pi_1 --E--> Q_{pi_1} --I--> ... --I--> pi* --E--> Q*

Where:
- **E** = Policy Evaluation (estimate Q for current π)
- **I** = Policy Improvement (make π greedy w.r.t. Q)

### Policy Improvement

    pi(s) = argmax_a Q(s, a)

**Process:**
1. Get Q(s, a) values
2. Pick the one with the highest value
3. The arg that gets the max value becomes the policy

---

## Monte Carlo ES (Exploring Starts)

**For estimating π ≈ π***

> **📚 From Sutton & Barto (p. 99):**
>
> ```
> Monte Carlo ES (Exploring Starts), for estimating π ≈ π*
>
> Initialize:
>     π(s) ∈ A(s) (arbitrarily), for all s ∈ S
>     Q(s, a) ∈ ℝ (arbitrarily), for all s ∈ S, a ∈ A(s)
>     Returns(s, a) ← empty list, for all s ∈ S, a ∈ A(s)
>
> Loop forever (for each episode):
>     Choose S₀ ∈ S, A₀ ∈ A(S₀) randomly such that all pairs have probability > 0
>     Generate an episode from S₀, A₀, following π: S₀, A₀, R₁, ..., S_{T-1}, A_{T-1}, R_T
>     G ← 0
>     Loop for each step of episode, t = T-1, T-2, ..., 0:
>         G ← γG + R_{t+1}
>         Unless the pair S_t, A_t appears in S₀, A₀, S₁, A₁, ..., S_{t-1}, A_{t-1}:
>             Append G to Returns(S_t, A_t)
>             Q(S_t, A_t) ← average(Returns(S_t, A_t))
>             π(S_t) ← argmax_a Q(S_t, a)
> ```

### Key Components

| Component | Formula | Purpose |
| --------- | ------- | ------- |
| **Q-value update** | Q(S_t, A_t) ← average(Returns(S_t, A_t)) | Compute rewards for state-action pairs |
| **Policy improvement** | π(S_t) ← argmax_a Q(S_t, a) | Improvement function at end of loop |

### Tabular Solution

**Q(s, a) table** to store values:

| State | Action 1 | Action 2 | Action 3 | Action 4 |
| ----- | -------- | -------- | -------- | -------- |
| S₁    | Q(S₁,a₁) | Q(S₁,a₂) | Q(S₁,a₃) | Q(S₁,a₄) |
| S₂    | Q(S₂,a₁) | Q(S₂,a₂) | Q(S₂,a₃) | Q(S₂,a₄) |
| ...   | ...      | ...      | ...      | ...      |

---

## Summary: Monte Carlo Methods

### Key Concepts

| Concept | Description |
| ------- | ----------- |
| **MC Prediction** | Estimate V_π or Q_π from sample episodes |
| **MC Control** | Find optimal policy using GPI with MC evaluation |
| **First-visit MC** | Update only on first visit to state in episode |
| **Every-visit MC** | Update on every visit to state in episode |
| **Exploring Starts** | Ensure all state-action pairs are visited |

### MC vs DP

| Aspect | Dynamic Programming | Monte Carlo |
| ------ | ------------------- | ----------- |
| **Model required?** | Yes (need p(s',r\|s,a)) | No (model-free) |
| **Updates** | Bootstrapping (use estimates) | Complete returns (no bootstrapping) |
| **Episode requirement** | Not required | Must complete episodes |

---

## Connection to Homework 1

This lecture covers the **Monte Carlo ES algorithm** that you'll implement for Homework 1:

1. **Initialize** Q(s,a) and π arbitrarily
2. **Generate episodes** with exploring starts
3. **Compute returns** backwards through episode
4. **Update Q-values** as average of returns
5. **Improve policy** greedily

---

*Transcribed from handwritten lecture notes. Algorithms inserted from Sutton & Barto Chapter 5.*
