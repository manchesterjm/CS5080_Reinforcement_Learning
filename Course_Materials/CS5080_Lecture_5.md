# CS 5080 Lecture 5 - Naive Policy Evaluation and Bellman Equations

**Date:** Tuesday, February 3, 2026, 4:45 PM
**Continues from:** Lecture 4 (Returns, Value Functions)
**Reading:** Sutton & Barto Ch. 3.5-3.8 (completes Chapter 3)
**Professor's Note:** This lecture completes Chapter 3 coverage

---

## The "Maze" Example (Review)

```
    ┌─────────┬─────────┬─────────┐
    │   X₁    │   X₂    │   X₃    │
    │    ↓    │    ↔    │   (G)   │
    │    ↓    │    ↓    │  GOAL   │
    ├─────────┼─────────┼─────────┤
    │   X₄    │   X₅    │   X₆    │
    │    ↑    │    ↔    │    ↑    │
    │    ↓    │    ↔    │         │
    └─────────┴─────────┴─────────┘
```

**States:** S = {X₁, X₂, X₃, X₄, X₅, X₆}
- Arrows indicate possible transition directions
- X₃ is the goal state (G)

---

## Naive Approach to RL: Policy Evaluation

### Core Idea

**Given a policy π:** Evaluate each state by computing V_π(x)

$$V_\pi(x) = \mathbb{E}[G_t | S_t = x] \quad \text{for all } x \in S$$

**To compute V_π(x):** Agent performs n episodes following π

### Episode Structure

**One Episode:**
```
         A₁        A₂              Aₖ     R_{k+1}
    S₁ ────→ S₂ ────→ ... ────→ Sₖ ────→ S_T (terminal)
         R₁        R₂              Rₖ
```

Where:
- Sᵢ = state at time i
- Aᵢ = action at time i
- Rᵢ = reward received after taking action Aᵢ₋₁

**Value at state Sᵢ in episode e:**
$$V_{\pi,e_i}(S_i) = G_{t,e_i}$$

---

## Return Calculation

### Definition

The **return** at time t in episode e:

$$G_{t,e_i} = R_{t+1,e_i} + \gamma R_{t+2,e_i} + \gamma^2 R_{t+3,e_i} + ...$$

where e = episode index, γ = discount factor

### Recursive Formulation

**Key insight:** G_t can be computed recursively (working backwards):

$$G_t = R_{t+1} + \gamma G_{t+1}$$

**Computing G backwards through an episode:**

```
At terminal time T:
    G_T = 0

At time T-1:
    G_{T-1} = R_T + γ × 0 = R_T

At time T-2:
    G_{T-2} = R_{T-1} + γ × G_{T-1} = R_{T-1} + γR_T

...continuing backwards...
```

> **📚 Textbook Reference (S&B Eq. 3.9):** The recursive relationship G_t = R_{t+1} + γG_{t+1} is fundamental. It shows that the return at time t equals the immediate reward plus the discounted return from t+1 onward.

### Why Compute Backwards?

The recursive formula makes it natural to compute returns **backwards** through an episode:
- The reward normally comes right before the end
- We know G_T = 0 at terminal state
- Work backwards: G_{T-1}, G_{T-2}, ..., G_1

---

## Naive Algorithm for V_π(s)

### The Table-Based Approach

**Keep a table** storing V_π(s) for each state:

| State | V_π(s) |
| ----- | ------ |
| S₁    | ?      |
| S₂    | ?      |
| ...   | ...    |
| Sₙ    | ?      |

### Algorithm Description

```
1. Run N episodes (N = large number) following policy π
2. In each episode, compute V(s) for every state s that occurs
3. Average V(s) with prior values (running average)
```

**Improved computation for V_π(s):**
When computing averages, compute G_t's backwards through each episode.

### Episode Visualization

```
Episode 1:    S₁ ──A₁→ S₂ ──A₂→ ... ──Aₖ→ S_T
                   R₁      R₂          Rₖ

Episode i:    S₁ ──A₁→ Sₐ ──A₂→ Sᵦ ──...→ S_T
                   R₁      Rₐ      Rᵦ

Episode N:    Compute V_π(Sᵢ) and use it to compute avg for V(sᵢ)
```

---

## Problems with the Naive Approach

### Issue 1: Large State-Action-Reward Spaces

**What happens when the S-A-R space is large?**

- Some states may **never be visited**, so nothing is learned from those states
- Poor exploration coverage

### Issue 2: Convergence

- **May or may not converge** depending on:
  - Exploration strategy
  - State space coverage
  - Episode length

### Issue 3: Speed

- **Very slow** - requires many complete episodes
- Must wait until episode ends to update values

### Issue 4: State Representation

- **States are just numbers:** 1, 2, 3, ...
- Need to provide some structure/detail to states

**Solutions for state representation:**

| Approach             | Description                                         |
| -------------------- | --------------------------------------------------- |
| **Coordinates**      | Provide (x, y) position in grid                     |
| **Two features**     | Use descriptive features to identify states         |
| **Complex state**    | Include where agent is as part of state description |
| **Partial/Full obs** | Can see entire state space, or just part of it      |

---

## MDPs Have No History/Memory

### The Markov Property

**MDPs have no history/memory** - future depends only on current state, not how we got there.

**MDP History (trajectory):**
$$S_1, A_1, R_1, S_2, A_2, R_2, ..., S_t$$

This is just a **string** of state-action-reward tuples.

```
                                          ← horizon →
    S₁ ─A₁→ R₁ ─→ S₂ ─A₂→ R₂ ─→ ... ─→ Sₜ
```

### String Processing Perspective

Could use:
- **Sequence processing approaches** (RNNs, LSTMs)
- **Explicit memory** mechanisms
- **Transformer** architectures

> **📚 Textbook Reference (S&B 3.1):** The Markov property states that the probability of each possible value for S_t and R_t depends only on the immediately preceding state and action, S_{t-1} and A_{t-1}, and not on earlier states and actions.

---

## Two Types of RL Algorithms

### Value-Based Methods

**Learn V_π(s) values** (state-value function)

- Estimate how good it is to be in each state
- Policy derived by choosing actions leading to high-value states

### Policy-Based Methods

**Learn Q_π(s, a) values** (action-value function)

- Estimate how good it is to take action a in state s
- Policy directly selects actions based on Q-values

### Hybrid Approaches

**Combination of both** - Actor-Critic methods

- Actor: learns policy
- Critic: learns value function

---

## Action-Value Function Q_π(s, a)

### Definition

$$Q_{\pi}(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

The expected return starting from state s, taking action a, then following policy π.

> **📚 Textbook Reference (S&B Eq. 3.13):** The action-value function q_π(s,a) gives the expected return starting from s, taking action a, and thereafter following policy π.

### Goal: Fill a Q(s,a) Table

| State | ↑    | ↓    | ←    | →    |
| ----- | ---- | ---- | ---- | ---- |
| S₁    |      |      |      |      |
| S₂    |      |      |      |      |
| ...   |      |      |      |      |

### Q-value Interpretation

$$Q_{e_i}(S_t, A_t) = \text{Return obtained by performing action } A_t \text{ in } S_t$$

This may **not be in policy π**, but then follow π to the end.

**Process:**
1. Perform **every** action 'a' in state S_t
2. Then follow policy π afterwards
3. Record the return for each (s, a) pair

---

## Non-Determinism in MDPs

### General MDP Dynamics

In the general case, the dynamics of an MDP are given as (according to Sutton & Barto):

$$p(x', r | x, a) = \text{Probability}(S_{t+1} = x', R_{t+1} = r | S_t = x, A_t = a)$$

Alternate notation: p(x', r, x, a)

### Probability Constraint

We must have:
$$\sum_{s' \in S} \sum_{r \in \mathcal{R}} p(s', r | s, a) = 1$$

for all s ∈ S, a ∈ A(s)

> **📚 Textbook Reference (S&B Eq. 3.2-3.3):** The function p defines the dynamics of the MDP. It gives a probability distribution over next states and rewards for each state-action pair.

---

## Bellman Equation for V_π

### Derivation

The Bellman equation is **needed for fundamentals of learning**.

Starting from the definition:
$$V_\pi(x) = \mathbb{E}_\pi[G_t | S_t = x]$$

Using the recursive return formula (Eq. 3.9):
$$= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = x]$$

**"Goes here"** - this is where the key derivation happens:

Expanding the expectation over actions, next states, and rewards:

$$V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} \sum_r p(s', r | s, a) \left[ r + \gamma V_\pi(s') \right]$$

> **📚 Textbook Reference (S&B Eq. 3.14):** This is the Bellman equation for v_π. It expresses the value of a state as the expected immediate reward plus the discounted value of successor states.

### Bellman Equation Components

| Component        | Meaning                                              |
| ---------------- | ---------------------------------------------------- |
| π(a\|s)          | Probability of taking action a in state s            |
| p(s', r \| s, a) | Probability of transitioning to s' with reward r     |
| r                | Immediate reward                                     |
| γV_π(s')         | Discounted value of next state                       |

### Backup Diagram for V_π

```
           s     ← root node (state)
          /|\
         / | \
        ↓  ↓  ↓   ← actions (chosen by π)
       •  •  •    ← action nodes
      /|  |  |\
     ↓ ↓  ↓  ↓ ↓  ← possible next states
    ○ ○  ○  ○ ○   ← s' (successor states)
```

The Bellman equation averages over:
1. Actions (weighted by policy π)
2. Next states and rewards (weighted by dynamics p)

---

## Bellman Equation for Q_π

### From State-Values to Action-Values

The Bellman equation for action values:

$$Q_\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma Q_\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]$$

Expanding:
$$Q_\pi(s, a) = \sum_{s'} \sum_r p(s', r | s, a) \left[ r + \gamma \sum_{a'} \pi(a'|s') Q_\pi(s', a') \right]$$

### Backup Diagram for Q_π

```
        s, a   ← root node (state-action pair)
         |
         ↓
        /|\
       ↓ ↓ ↓   ← possible (s', r) outcomes
      ○ ○ ○    ← successor states s'
     /|\
    • • •      ← actions a' at s'
```

---

## Optimal Policies and Value Functions

### Optimal State-Value Function

$$V_*(s) = \max_\pi V_\pi(s) \quad \text{for all } s \in S$$

The maximum value achievable from state s under any policy.

> **📚 Textbook Reference (S&B Eq. 3.15):** The optimal state-value function v* is the maximum over all policies of the expected return from each state.

### Optimal Action-Value Function

$$Q_*(s, a) = \max_\pi Q_\pi(s, a) \quad \text{for all } s \in S, a \in A(s)$$

> **📚 Textbook Reference (S&B Eq. 3.16):** The optimal action-value function q* gives the expected return for taking action a in state s and thereafter following an optimal policy.

### Relationship Between V* and Q*

$$Q_*(s, a) = \mathbb{E}[R_{t+1} + \gamma V_*(S_{t+1}) | S_t = s, A_t = a]$$

---

## Bellman Optimality Equations

### For V*

$$V_*(s) = \max_{a \in A(s)} Q_{\pi_*}(s, a)$$

$$= \max_a \mathbb{E}[R_{t+1} + \gamma V_*(S_{t+1}) | S_t = s, A_t = a]$$

$$= \max_a \sum_{s', r} p(s', r | s, a) \left[ r + \gamma V_*(s') \right]$$

> **📚 Textbook Reference (S&B Eq. 3.19):** The Bellman optimality equation for v*. It states that the value of a state under an optimal policy must equal the expected return for the best action from that state.

### For Q*

$$Q_*(s, a) = \mathbb{E}\left[ R_{t+1} + \gamma \max_{a'} Q_*(S_{t+1}, a') \Big| S_t = s, A_t = a \right]$$

$$= \sum_{s', r} p(s', r | s, a) \left[ r + \gamma \max_{a'} Q_*(s', a') \right]$$

> **📚 Textbook Reference (S&B Eq. 3.20):** The Bellman optimality equation for q*.

### Backup Diagrams for Optimal Value Functions

**For V*:**
```
        (v*)    s
               /|\
         max→ / | \  ← choose best action
             •  •  •
            /|  |  |\
           ○ ○  ○  ○ ○ ← s'
```

**For Q*:**
```
        (q*)   s, a
                |
               /|\
              ○ ○ ○   ← s'
             /|\
       max→ • • •     ← choose best a'
```

---

## Solving the Bellman Optimality Equation

### Theoretical Solution

For finite MDPs, the Bellman optimality equation has a **unique solution** for V*.

- If dynamics p(s', r | s, a) are known
- Can solve system of |S| nonlinear equations

### Practical Challenges

| Challenge                     | Description                                              |
| ----------------------------- | -------------------------------------------------------- |
| **Dynamics often unknown**    | Agent must learn from interaction                        |
| **Computational complexity**  | 10²⁰ states in backgammon → thousands of years to solve  |
| **Memory requirements**       | Can't store table for huge state spaces                  |

### Finding Optimal Policy from V* or Q*

**From V*:** For each state s, choose action that maximizes:
$$\arg\max_a \sum_{s', r} p(s', r | s, a) [r + \gamma V_*(s')]$$

Requires one-step lookahead and knowing dynamics.

**From Q*:** For each state s, simply choose:
$$\arg\max_a Q_*(s, a)$$

No lookahead needed! Q* caches all the lookahead information.

> **📚 Textbook Reference (S&B 3.6):** Having q* makes choosing optimal actions especially easy: for any state s, simply find the action that maximizes q*(s, a). The action-value function effectively caches the results of all one-step-ahead searches.

---

## Gridworld Example (S&B Example 3.5/3.8)

### Setup

```
    ┌─────┬─────┬─────┬─────┬─────┐
    │  A  │     │     │     │  B  │
    │ +10 │     │     │     │ +5  │
    ├─────┼─────┼─────┼─────┼─────┤
    │     │     │     │     │     │
    │     │     │     │     │     │
    ├─────┼─────┼─────┼─────┼─────┤
    │     │     │     │     │     │
    │     │     │     │     │     │
    ├─────┼─────┼─────┼─────┼─────┤
    │     │     │     │     │     │
    │     │     │     │     │     │
    └─────┴─────┴─────┴─────┴─────┘
```

- Actions: north, south, east, west (deterministic)
- Off-grid actions: stay in place, reward = -1
- State A: all actions → A', reward = +10
- State B: all actions → B', reward = +5

### State-Value Function (Random Policy, γ = 0.9)

| 3.3  | 8.8  | 4.4  | 5.3  | 1.5  |
| ---- | ---- | ---- | ---- | ---- |
| 1.5  | 3.0  | 2.3  | 1.9  | 0.5  |
| 0.1  | 0.7  | 0.7  | 0.4  | -0.4 |
| -1.0 | -0.4 | -0.4 | -0.6 | -1.2 |
| -1.9 | -1.3 | -1.2 | -1.4 | -2.0 |

### Optimal Value Function V*

| 22.0 | 24.4 | 22.0 | 19.4 | 17.5 |
| ---- | ---- | ---- | ---- | ---- |
| 19.8 | 22.0 | 19.8 | 17.8 | 16.0 |
| 17.8 | 19.8 | 17.8 | 16.0 | 14.4 |
| 16.0 | 17.8 | 16.0 | 14.4 | 13.0 |
| 14.4 | 16.0 | 14.4 | 13.0 | 11.7 |

### Optimal Policy π*

| →    | ←↑↓→ | ←    | ←↑↓→ | ←    |
| ---- | ---- | ---- | ---- | ---- |
| ↓    | ↑    | ↓    | ↓    | ←    |
| ↓    | ↑    | ↓    | ↓    | ↓    |
| ↓    | ↑    | ↓    | ↓    | ↓    |
| ↑    | ↑    | ↓    | ↓    | ↓    |

Where multiple arrows appear, all are optimal.

---

## Optimality and Approximation

### The Reality of RL

**Key insight from S&B 3.7:**

Solving the Bellman optimality equation directly is rarely feasible because:

1. **Dynamics usually unknown** - must learn from experience
2. **Computational resources limited** - can't enumerate all states
3. **Memory constraints** - can't store values for all states

### Approximation Methods

| Method                    | Description                                      |
| ------------------------- | ------------------------------------------------ |
| **Tabular methods**       | Store V(s) or Q(s,a) in tables (small state spaces) |
| **Function approximation** | Parameterized functions (neural networks, etc.) |
| **Online learning**       | Update values as agent interacts with environment |

> **📚 Textbook Reference (S&B 3.7):** The online nature of RL makes it possible to approximate optimal policies in ways that put more effort into learning to make good decisions for frequently encountered states, at the expense of less effort for infrequently encountered states.

---

## Chapter 3 Summary

### Key Concepts Introduced

| Concept                    | Definition                                                    |
| -------------------------- | ------------------------------------------------------------- |
| **MDP**                    | Markov Decision Process with (S, A, R, p, γ)                  |
| **Policy π**               | Mapping from states to action probabilities                   |
| **Return G_t**             | Cumulative discounted reward from time t                      |
| **State-value V_π(s)**     | Expected return starting from s, following π                  |
| **Action-value Q_π(s,a)**  | Expected return starting from s, taking a, then following π   |
| **Bellman equation**       | Recursive relationship for value functions                    |
| **Optimal value V*, Q***   | Maximum values achievable under any policy                    |
| **Bellman optimality eq.** | Consistency condition for optimal value functions             |

### Two Fundamental Problems

| Problem             | Description                        | Methods (Preview)    |
| ------------------- | ---------------------------------- | -------------------- |
| **Policy Evaluation** | Compute V_π for a given π         | MC, TD               |
| **Control**         | Find optimal π*                    | Value/Policy Iteration, Q-learning |

---

## Looking Ahead

**Chapter 3 Complete!** The professor indicated this is as far as we go in Chapter 3.

**Next topics (per syllabus):**
- Chapter 5: Monte Carlo Methods (formal treatment)
- Chapter 6: Temporal Difference Learning (TD(0), SARSA, Q-learning)

---

## Key Equations Reference

| Equation                | Formula                                                                        |
| ----------------------- | ------------------------------------------------------------------------------ |
| Return (recursive)      | G_t = R_{t+1} + γG_{t+1}                                                       |
| V_π definition          | V_π(s) = 𝔼_π[G_t \| S_t = s]                                                   |
| Q_π definition          | Q_π(s,a) = 𝔼_π[G_t \| S_t = s, A_t = a]                                        |
| Bellman for V_π         | V_π(s) = Σ_a π(a\|s) Σ_{s',r} p(s',r\|s,a)[r + γV_π(s')]                       |
| V* definition           | V*(s) = max_π V_π(s)                                                           |
| Bellman optimality V*   | V*(s) = max_a Σ_{s',r} p(s',r\|s,a)[r + γV*(s')]                               |
| Bellman optimality Q*   | Q*(s,a) = Σ_{s',r} p(s',r\|s,a)[r + γ max_{a'} Q*(s',a')]                      |

---

*Transcribed from handwritten lecture notes. Cross-referenced with Sutton & Barto Ch. 3 (Finite Markov Decision Processes). This lecture completes Chapter 3 coverage.*
