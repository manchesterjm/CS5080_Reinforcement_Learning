# CS 5080 - Lecture 11
**Date:** Tuesday, March 3, 2026

## n-step TD / Bootstrapping / Temporal Difference

### MC vs TD Comparison

- **Monte Carlo (MC):** Look at the entire episode, t to T
- **Temporal Difference (TD):** One step action forward, one step imagination/look-ahead

### TD(0) Algorithm

**V_pi estimation:**

    V(s) <-- V(s) + alpha * [r + gamma * V(s') - V(s)]
                                 ^--- TD error ---^

- Action gives reward, look at next state
- Equivalent form: V(s) = V(s) + δ[G_{t+1} - V(s)]

**SARSA:**

    q(s,a) <-- q(s,a) + alpha * [r + gamma * q(s', a') - q(s,a)]
                                   ^--- TD error ---^

- Equivalent form: q(s,a) = q(s,a) + δ[G_{t+1} - q(s,a)]

**Q-learning:**

    q(s,a) <-- q(s,a) + alpha * [r + gamma * max_{a'} q(s', a') - q(s,a)]
                                   ^--- TD error ---^

- Why only look ahead more than one step?

### TD(1) Algorithm — 2-step Actions/Look-ahead

Trajectory: S_t → (a_t, r_{t+1}) → S_{t+1} → (a_{t+1}, r_{t+2}) → S_{t+2}

**For state value (V):**

    G_{t:t+2}|_s = r_{t+1} + gamma * r_{t+2} + gamma^2 * V(S_{t+2})

**For action value (q):**

    G_{t:t+2}|_{s,a} = r_{t+1} + gamma * r_{t+2} + gamma^2 * q(S_{t+2}, a_{t+2})

- Looking at algorithm on **Figure 7.1** (Sutton & Barto)
- Also: n-step TD for estimating V ≈ V_pi
- Also: n-step SARSA for estimating Q ≈ q_* or q_pi
- Also: off-policy n-step SARSA for estimating Q ≈ q_* or q_pi

**Lecture on this chapter is done — need to still go over the rest of the chapter on our own.**

---

## Machine Learning Start (Transition Topic)

### Traditional RL Limitations
- States are assumed to be simple indices: s_1, s_2, ..., s_n
- We may want better representations of states

**Parking lot example** (professor drew): A grid-structured parking lot where an agent has limited view of parking areas — only line of sight. Agent sees the current state it is in: full parking spaces, empty ones, etc.

- There is no memory or history of what happened in the past
  - Quite likely an agent remembers the recent past
  - Agent has detailed past memory of past n episodes

- Actions are discrete (up, down, left, right)
  - Actions in real life are often parameterizable

- When state representations become complex, we need more complex machinery or computing algorithms to augment the RL process
  - Professor drew Tetris representation

- The number of states becomes large
- RL agent may have to perform well in states it has never seen

### Tabular vs Function Approximation

- V(s) table and q(s,a) table → these are **discrete functions**
- V(s) table maps: state input → value output
- q(s,a) table maps: input (s, a) → output q value

**Function Approximator** (replaces tables):
- State inputs → Function Approximator (Sum, Naive Bayes, ANN) → action/output probabilities
- Also takes "other things" as input

### ML: Finding Functions from Data

Machine learning = finding the functions based on a table of inputs and outputs

| X₁ | X₂ | Function |
|----|-----|----------|
| 1  | 2   | 3        |
| 2  | 2   | 4        |
| 10 | 1   | 11       |

- It should be able to figure out this function adds X₁ and X₂
- Easy for linear, almost impossible for non-linear
- It is also possible that there are errors in the training data that the ML should still be able to figure out
