# Chapter 2 Summary: Mathematical Foundations of Reinforcement Learning
**Source:** Grokking Deep Reinforcement Learning (Morales, 2020)
**Prepared for:** CS 5080 Lecture 3 (Thursday, Jan 29, 2026)

---

## Key Concepts at a Glance

| Term | Definition |
|------|------------|
| **MDP** | Mathematical framework for sequential decision-making under uncertainty |
| **State (S)** | Unique configuration of the environment |
| **Action (A)** | Mechanism to influence the environment |
| **Transition Function T(s,a,s')** | Probability of reaching s' from s via action a |
| **Reward Function R(s,a,s')** | Scalar signal indicating goodness of a transition |
| **Policy** | Mapping from states to actions |
| **Episode** | Sequence from initial to terminal state |
| **Discount Factor (γ)** | Reduces value of future rewards (typically 0.99) |

---

## 1. The RL Interaction Cycle

**Two Core Components:**
- **Agent:** The decision maker (the solution)
- **Environment:** The representation of the problem

**Key Insight:** RL is unique because agent and environment *interact* - the agent influences the environment through actions, and the environment reacts.

**Three Challenges of RL:**
1. **Complex** - Vast action/state spaces, learning from sampled feedback
2. **Sequential** - Delayed consequences, credit assignment problem
3. **Uncertain** - Unknown world dynamics, need for exploration

---

## 2. Markov Decision Process (MDP)

### Formal Definition
An MDP is defined by the tuple: **(S, A, T, R, γ, H, S_i)**

| Component | Symbol | Description |
|-----------|--------|-------------|
| State Space | S | Set of all possible states |
| Action Space | A(s) | Set of actions available in state s |
| Transition Function | T(s,a,s') | P(s' \| s, a) - probability of transition |
| Reward Function | R(s,a,s') | Scalar reward for transition |
| Discount Factor | γ | Value in [0,1] for discounting future rewards |
| Horizon | H | Finite or infinite time steps |
| Initial States | S_i | Distribution over starting states |

### The Markov Property
> **"The future depends only on the present, not the past."**

- The probability of the next state given current state and action is independent of history
- Current state must encapsulate all relevant information
- This is why velocity (not just position) matters in control problems

---

## 3. States: Environment Configuration

**State Space S:**
- Can be finite or infinite
- Each state is a set of variables (must be finite size)
- States must be self-contained (Markov property)

**Types of States:**
- **Terminal states (S+):** Episode ends here
- **Non-terminal states (S):** Agent continues
- **Initial states (S_i):** Where episodes begin

**Key Convention:** Terminal states have all actions transition to themselves with probability 1 and reward 0.

**Example - Frozen Lake:**
- 16 states (cells 0-15)
- State = single integer (cell ID)
- 5 terminal states (holes + goal)

---

## 4. Actions: Influencing the Environment

**Action Space A(s):**
- May vary by state (some actions unavailable)
- Can be discrete or continuous
- Agent selects deterministically or stochastically

**Example - Frozen Lake:**
- 4 actions: Left(0), Down(1), Right(2), Up(3)
- Same actions available in all non-terminal states

---

## 5. Transition Function: T(s, a, s')

**Definition:** Maps (state, action, next_state) → probability

**Properties:**
- Σ T(s,a,s') over all s' = 1 (probability distribution)
- Must be **stationary** (doesn't change during training)

**Types:**
- **Deterministic:** T(s,a,s') = 1 for exactly one s'
- **Stochastic:** Multiple possible next states with probabilities < 1

**Example - Frozen Lake (Slippery):**
- Intended direction: 33.3% probability
- Each orthogonal direction: 33.3% probability
- Bounces back if hitting wall

---

## 6. Reward Function: R(s, a, s')

**Definition:** Maps transition → scalar value

**Representations:**
- R(s, a, s') - Most explicit (depends on transition)
- R(s, a) - Marginalized over next states
- R(s) - Marginalized over actions

**Types:**
- **Dense:** Frequent non-zero rewards (faster learning, more bias)
- **Sparse:** Rare rewards (slower learning, emergent behaviors)

**Example - Frozen Lake:**
- +1 for landing on goal (state 15)
- 0 for all other transitions

---

## 7. Horizon: Time in MDPs

**Task Types:**
| Type | Description |
|------|-------------|
| **Episodic** | Finite time steps, has terminal states |
| **Continuing** | Infinite time steps, no natural ending |

**Planning Horizons:**
| Horizon | Description |
|---------|-------------|
| **Greedy (H=1)** | Plan only one step ahead |
| **Finite** | Known number of steps |
| **Infinite** | Plan for unlimited steps |
| **Indefinite** | Plan infinite, but may terminate |

**Episode:** Sequence of time steps from start to terminal state

---

## 8. Discount Factor: γ (Gamma)

**Purpose:**
1. Handle infinite horizons (ensure convergence)
2. Reduce variance in value estimates
3. Express "urgency" - prefer sooner rewards

**Effect:** Reward received t steps in future is worth γ^t of its face value

**Common Values:**
- γ = 0: Only immediate rewards matter (greedy)
- γ = 0.99: Standard choice (values future highly)
- γ = 1: No discounting (only for finite horizons)

**Mathematical Form:**
```
Return G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ... = Σ γ^k R_{t+k+1}
```

---

## 9. Value Functions (Preview for Later Chapters)

### State Value Function: V(s)
- Value of being in state s
- Expected return starting from s

### Action Value Function: Q(s, a)
- Value of taking action a in state s
- Expected return after taking a in s

### Deriving Policy from Values

**From V(s):** Go to neighbor state s' where R + V(s') is maximized

**From Q(s,a):** Choose action a that maximizes Q(s,a)

---

## 10. Example Environments

### Bandit Walk (BW) - Deterministic
```
[Hole] ← [Start] → [Goal]
  0         1         2
```
- 3 states, 2 actions (Left, Right)
- Deterministic transitions
- Reward: +1 for reaching Goal

### Bandit Slippery Walk (BSW) - Stochastic
- Same structure as BW
- 80% intended direction, 20% opposite
- Demonstrates stochastic transitions

### Frozen Lake (FL) - Full MDP
```
[S] [ ] [ ] [ ]     S = Start
[ ] [H] [ ] [H]     H = Hole
[ ] [ ] [ ] [H]     G = Goal
[H] [ ] [ ] [G]
```
- 16 states, 4 actions
- Slippery: 33.3% intended, 33.3% each orthogonal
- Reward: +1 for reaching Goal (state 15)

---

## 11. Python MDP Representation

```python
# Format: P[state][action] = [(prob, next_state, reward, is_terminal), ...]

# Frozen Lake Example (partial)
P = {
    0: {
        0: [(0.33, 0, 0.0, False), (0.33, 0, 0.0, False), (0.33, 4, 0.0, False)],
        # ... other actions
    },
    14: {
        1: [(0.33, 13, 0.0, False), (0.33, 14, 0.0, False), (0.33, 15, 1.0, True)],
        # ... reaching goal gives +1 reward
    },
    15: {  # Terminal state
        0: [(1.0, 15, 0, True)],  # All actions loop back
        1: [(1.0, 15, 0, True)],
        2: [(1.0, 15, 0, True)],
        3: [(1.0, 15, 0, True)],
    }
}
```

---

## 12. Key Takeaways for Thursday's Lecture

1. **MDPs are the mathematical foundation** for representing RL problems
2. **Markov property** = future depends only on present state
3. **Transition function** can be deterministic or stochastic
4. **Reward function** encodes the goal - design carefully!
5. **Discount factor γ** handles infinite horizons and adds urgency
6. **Value functions V(s) and Q(s,a)** are what agents learn
7. **Policy** is derived from value functions

---

## 13. Connections to Lecture 3

The lecture covered:
- Markov Chains (weather example with transition matrix)
- Markov Property (current state encapsulates history)
- MDPs (adding actions and rewards to Markov chains)
- State/Action spaces
- Episodes and trajectories
- V(s) vs Q(s,a) value functions
- Deriving policies from value functions
- Actor-Critic methods (compute both V and Q)

**Next:** Chapter 3 covers how to balance immediate vs long-term goals (temporal difference learning, returns, value function estimation).

---

*Summary prepared: January 28, 2026*
