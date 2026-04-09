# CS 5080 Lecture 4 - MDP Dynamics, Returns, and Value Functions

**Date:** Thursday, January 29, 2026, 4:37 PM
**Continues from:** Lecture 3 (MDPs)
**Reading:** Sutton & Barto Ch. 3.1-3.5, Grokking Ch. 2

---

## Dynamics of an RL Problem

An MDP's dynamics are fully specified by two components:

### The 6-State Maze Environment

```
    ┌─────────┬─────────┬─────────┐
    │   S₁    │   S₂    │   S₃    │
    │   (B)   │    ↔    │   (G)   │
    │  START  │    0    │  GOAL   │
    │    ↓0   │   ↓0    │   ⟲0    │
    ├─────────┼─────────┼─────────┤
    │   S₄    │   S₅    │   S₆    │
    │    ↔    │    ↔    │    ↑    │
    │    0    │    0    │   100   │
    └─────────┴─────────┴─────────┘
```

**States:** S = {S₁, S₂, S₃, S₄, S₅, S₆}
- S₁ = Start state (B)
- S₃ = Goal state (G) with self-loop reward of 0
- Arrows show possible transitions with their immediate rewards

### 1. State-Action-State Transition Table

Shows which state you transition TO for each (state, action) pair.

**Rows = From State, Columns = Action**

| From State | ↑   | ↓   | ←   | →   | ⟲   |
| ---------- | --- | --- | --- | --- | --- |
| S₁         | NA  | S₄  | NA  | S₂  | NA  |
| S₂         | NA  | S₅  | S₁  | S₃  | NA  |
| S₃         | NA  | S₆  | S₂  | NA  | S₃  |
| S₄         | S₁  | NA  | NA  | S₅  | NA  |
| S₅         | S₂  | NA  | S₄  | S₆  | NA  |
| S₆         | S₃  | NA  | S₅  | NA  | NA  |

*NA = Not Available (action not possible from that state)*

### 2. State-Action Immediate Reward Table

Shows the reward received for taking each action from each state.

**Rows = From State, Columns = Action**

| From State | ↑   | ↓   | ←   | →   | ⟲   |
| ---------- | --- | --- | --- | --- | --- |
| S₁         | NA  | 0   | NA  | 0   | NA  |
| S₂         | NA  | 0   | 0   | 100 | NA  |
| S₃         | NA  | 0   | 0   | NA  | 0   |
| S₄         | 0   | NA  | NA  | 0   | NA  |
| S₅         | 0   | NA  | 0   | 0   | NA  |
| S₆         | 100 | NA  | 0   | NA  | NA  |

*Note: Reward of 100 for transitions that reach goal state S₃ (self-loop at S₃ gives 0)*

> **📚 Textbook Reference (S&B 3.1):** The dynamics of an MDP are given by the function p(s', r | s, a) — the probability of transitioning to state s' with reward r, given state s and action a. The transition and reward tables above are simplified representations of this function for deterministic environments.

---

## Non-Determinism in MDPs

*(Reference: CS 4700 - AI course)*

### Deterministic vs. Stochastic Actions

**Deterministic Action (→):** "Go best action"
- Action → from S₁ always leads to S₂ (100% certainty)

**Stochastic Action (Try→):** "Try to go best action"
- Action Try→ from S₁:
  - 85% probability → S₂ (intended destination)
  - 10% probability → stay at S₁ (slip/fail)
  - 5% probability → S₄ (unintended)

```
                    ──→──
         ┌─── 85% ───→ S₂
         │
S₁ ──Try→┼─── 10% ───→ S₁ (stay)
         │
         └─── 5% ────→ S₄
```

*Or could use fully deterministic: 100% → S₂ and 0%, 0%*

### Reward Non-Determinism

For the **same action at the same state**, the rewards may be **stochastic**.

Example: Taking action → from S₁ might yield:
- Reward = 0 (80% of the time)
- Reward = -1 (15% of the time)
- Reward = +1 (5% of the time)

### Most Complex Case

The most complex non-deterministic case is **a combination of these two**:
1. **Transition non-determinism** (uncertain next state)
2. **Reward non-determinism** (uncertain reward)

> **📚 Textbook Reference (S&B 3.1):** This is captured in the full MDP dynamics function p(s', r | s, a), which gives the probability of each possible (next state, reward) pair.

---

## Goal of Reinforcement Learning

**Primary Goal:** To learn a better policy, possibly the ideal/best policy.

- There may be many or infinite possible policies
- But there exists at least one "best" policy (optimal policy π*)
- **Key Question:** What is the best action to perform in each state?

### Optimal Policy Visualization

```
Best Policy (arrows show best action per state):
    ┌───┬───┬───┐
    │ → │ → │ ⟲ │   S₁→S₂→S₃(stay)
    ├───┼───┼───┤
    │ ↑ │ ↑ │ ↑ │   S₄→S₁, S₅→S₂, S₆→S₃
    └───┴───┴───┘
```

> **📚 Textbook Reference (S&B 3.6):** An optimal policy π* is a policy that is better than or equal to all other policies. There always exists at least one optimal policy, though there may be more than one.

---

## Policy (π)

### Definition

A **policy** π is a mapping from states to probabilities of selecting each action.

    pi(a|s) = P(A_t = a | S_t = s)   for all s in S

### Initial Policy (π₀)

Before learning, the agent starts with an initial policy π₀.

**Equi-probable (Uniform Random) Policy:**
- All actions possible in a state are equally probable
- *[Or it may be random, or expert-based]*

**Policy Table (equi-probable example):**

| State | ↑    | ↓    | ←    | →    | ⟲    |
| ----- | ---- | ---- | ---- | ---- | ---- |
| S₁    | 0    | 0.5  | 0    | 0.5  | 0    |
| S₂    | 0    | 0.33 | 0.33 | 0.33 | 0    |
| S₃    | 0    | 0.33 | 0.33 | 0    | 0.33 |
| S₄    | 0.5  | 0    | 0    | 0.5  | 0    |
| S₅    | 0.33 | 0    | 0.33 | 0.33 | 0    |
| S₆    | 0.5  | 0    | 0.5  | 0    | 0    |

*Can be written as a table in the simple case. Only valid actions have non-zero probability.*

### Learned "Best" Policy

After training, the learned policy is deterministic (one action per state):

| State | Action |
| ----- | ------ |
| S₁    | →      |
| S₂    | →      |
| S₃    | ⟲      |
| S₄    | ↑      |
| S₅    | ↑      |
| S₆    | ↑      |

### Policy May Include Offline Knowledge

- Expert-designed initial policy
- Domain knowledge encoded in π₀
- Transfer learning from similar tasks

---

## Reinforcement Learning Process

**Given:**
1. The dynamics of the problem (transition & reward structure)
2. An initial policy π₀ (random, equi-probable, or expert-based)

**Goal:** Learn a "best" policy

### Learning Through Episodes

**Given the dynamics of the "maze", use equi-prob policy (say):**

**Episode 1:** Short path (2 transitions)
```
       →         →
  X₁ ────→ X₂ ────→ X₃ (Goal!)
       0        100
```
*Action on top of arrow, reward on bottom*

**Episode 2:** Longer path (5 transitions)
```
       →         →         →         →         →         →
  X₁ ────→ X₂ ────→ X₄ ────→ X₁ ────→ X₅ ────→ X₆ ────→ X₈ (Goal!)
       0         0         0         0         0        100
```

**Episode 3:** Very long path
```
       →                                              →
  X₁ ────→ ... lots of state transitions ... ────→ X₉ (Goal!)
       0              (many 0s)                     100
```

**Key Question:** Need to find out how good state S_t is for the RL agent.

**Answer:** The agent wants to **maximize the cumulative immediate rewards** (the return).

---

## Return (G_t)

The **return** G_t is the total accumulated reward from time step t onward.

### Simple (Undiscounted) Return

    G_t = R_{t+1} + R_{t+2} + ... + R_T

Sum of all immediate rewards from t+1 to terminal time T.

**Problem:** For the maze, all episodes reaching the goal get G_t = 100, regardless of path length! Need to track steps to make this learn.

### Discounted Return

Could use this - a **discount factor** γ (gamma):

    G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+3} + ... + gamma^{T-t-1} * R_T

Where: **1 ≥ γ ≥ 0**

*Note: In handwriting, γ often looks like a checkmark (√)*

- γ = 1: No discounting (future rewards valued equally)
- γ = 0: Only immediate reward matters
- γ ≈ 0.9-0.99: Typical values (future rewards worth less)

### Discounted Return Examples

**Assume γ = 0.9**

**Episode 1:** X₁ →⁰ X₂ →¹⁰⁰ Goal (2 steps)
    G_1 = 0 + gamma x 100 = 0 + 0.9 x 100 = 90

**Episode 2:** X₁ →⁰ X₂ →⁰ X₃ →⁰ X₄ →⁰ X₅ →¹⁰⁰ Goal (5 steps)
    G_2 = 0 + gamma x 0 + gamma^2 x 0 + gamma^3 x 0 + gamma^4 x 100
    G_2 = 0.9^4 x 100 = 0.6561 x 100 = 65.61

**Episode 3:** Very long path
    G_3 = 0.00...2656 (very small)

> **📚 Textbook Reference (S&B 3.3):** The discount factor γ determines the present value of future rewards. A reward received k steps in the future is worth only γᵏ⁻¹ times what it would be worth if received immediately.

### Why Discounting Matters

| Episode | Path Length | Return (γ=0.9) |
| ------- | ----------- | -------------- |
| 1       | 2 steps     | 90.00          |
| 2       | 5 steps     | 65.61          |
| 3       | 10 steps    | 38.74          |
| 4       | 20 steps    | 15.01          |

**Shorter paths yield higher returns!** This naturally encourages efficient behavior.

> **📚 Textbook Note (S&B 3.3):** Discounting also ensures the return is finite for continuing (non-episodic) tasks, as long as the reward sequence is bounded and γ < 1.

---

## Value of a State

### Episode-Specific Value

For a single episode i, the value of state X₁ is simply the return from that episode:

    V_{episode_i}(X_1) = G_1 in episode i

### General State Value

In general, the value of being in state S_t is the return G_t observed in some episode:

    V(S_t) = G_t in some episode

But returns vary across episodes! We need an **expected** value.

### State-Value Function V_π(s)

**Definition:** The value of state s under policy π is the **expected return** when starting from s and following π:

    V_pi(x) = E_pi[G_t | S_t = x]   x in S

**Estimated as:**
    V_pi(x) ≈ (1/#episodes) * [R_{t+1} + gamma * R_{t+2} + ...]

> **📚 Textbook Reference (S&B 3.5):** The state-value function v_π(s) for an MDP is defined as the expected return starting from state s and thereafter following policy π. This is a fundamental concept in RL.

---

## Naive Algorithm to Learn V_π(s)

### Algorithm (First-Visit Monte Carlo)

```
Initialize:
    For every state s:
        no_visits[s] = 0
        V(s) = 0

For i = 1 to num_episodes:
    Start episode_i from a start state
    Finish episode_i (run until terminal state using policy π)

    For t = 1 to T_i:   # For each time step in episode
        G_{t,i} = R_{t+1,i} + R_{t+2,i} + ...   # Calculate return

        # Update value estimate (incremental mean)
        V(S_t) = [no_visits × V(S_t) + G_{t,i}] / [no_visits + 1]
        no_visits = no_visits + 1
```

### Incremental Mean Update

The update formula maintains a running average:
    V(S_t) <-- (n * V(S_t) + G_{t,i}) / (n + 1)

Where n = no_visits[S_t]

> **📚 Textbook Reference (S&B Ch. 5):** This is the First-Visit Monte Carlo method for policy evaluation. Every-Visit MC is similar but updates for every visit to a state, not just the first visit per episode.

---

## Issues with the Naive Algorithm

| Issue                      | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| **Very, very slow**        | In big environments, need many episodes to visit all states    |
| **Convergence may not happen** | May not converge without sufficient exploration            |
| **Deterministic**          | If policy is deterministic, limited exploration                |
| **May not visit all states** | Some states may be unreachable under the current policy      |

> **📚 Textbook Reference (S&B 5.1):** Monte Carlo methods require complete episodes and can only be applied to episodic tasks. They also require sufficient exploration, which is why ε-greedy policies or exploring starts are often used.

### Solutions (Preview of Future Lectures)

1. **ε-greedy exploration:** With probability ε, take random action
2. **Exploring starts:** Start episodes from random states
3. **Temporal Difference (TD) learning:** Update after each step, not episode
4. **Function approximation:** Generalize to unseen states

---

## Key Equations Summary

| Concept              | Formula                                                        |
| -------------------- | -------------------------------------------------------------- |
| Policy               | π(a\|s) = P(A_t = a \| S_t = s)                                 |
| Undiscounted Return  | G_t = R_{t+1} + R_{t+2} + ... + R_T                            |
| Discounted Return    | G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... + γ^{T-t-1}R_T      |
| State-Value Function | V_π(s) = 𝔼_π[G_t \| S_t = s]                                   |
| Recursive Return     | G_t = R_{t+1} + γG_{t+1}                                       |

---

## Connections to Previous Lectures

| Lecture 3 Concept | Lecture 4 Extension                                |
| ----------------- | -------------------------------------------------- |
| MDP definition    | Full dynamics specification (transition + reward tables) |
| Policy definition | Equi-probable vs. learned optimal policy           |
| Rewards           | Cumulative return with discounting                 |
| Episodes          | Returns calculated per episode for value estimates |

---

## Looking Ahead

**Next topics (per syllabus):**
- Bellman equations for V_π and optimal V*
- Action-value function Q_π(s, a)
- Monte Carlo methods (formal treatment)
- Temporal Difference learning

---

*Transcribed from handwritten lecture notes. Cross-referenced with Sutton & Barto Ch. 3 and Grokking Deep RL Ch. 2.*
