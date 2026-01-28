# CS 5080 Lecture 3 - Markov Decision Processes (Enhanced)
**Date:** Tuesday, January 27, 2026, 4:45 PM
**Source:** Chapter 2 of Grokking Deep Reinforcement Learning (Morales)

**Legend:**
- Regular text = Your lecture notes (verified correct)
- ⚠️ = Correction or clarification needed
- 📚 **[ADDITIONAL FROM TEXTBOOK]** = Supplementary information from Chapter 2

---

## Markov Chains

### Weather Example
- **States:** S = {cold, rain, sun}
- **Transition Table T:**

| From/To | Cold | Rain | Sun |
|---------|------|------|-----|
| Cold    | 0.2  | 0.5  | 0.3 |
| Rain    | 0.5  | 0.3  | 0.2 |
| Sun     | 0.5  | 0.1  | 0.4 |

*Note: Each row sums to 1* ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK]:** This table is called a **transition matrix** or **stochastic matrix**. Each entry T[i,j] represents P(next_state = j | current_state = i). The requirement that rows sum to 1 ensures it's a valid probability distribution.

### Markov Property (Simplifying Assumption)
- **The current state encapsulates all effects of history** ✓ Correct
- Next transition depends *only* on the current state ✓ Correct
- History of the chain: C, R, R, R, S, S, C, S, ...
  - "What will it be tomorrow?" depends on current state only ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK - Formal Definition]:**
> P(Sₜ₊₁ | Sₜ, Aₜ, Sₜ₋₁, Aₜ₋₁, ..., S₀, A₀) = P(Sₜ₊₁ | Sₜ, Aₜ)

The probability of the next state given the current state and action is **independent of all previous states and actions**. This is the **memoryless property**.

📚 **[ADDITIONAL FROM TEXTBOOK - Why This Matters]:**
Because RL agents are designed assuming Markov property, you must feed agents all necessary variables to make the property hold. For example:
- **Spacecraft landing:** Need position AND velocity (not just position)
- **Grid worlds:** Cell ID alone is sufficient
- The more variables you add → longer training but more accurate
- Fewer variables → faster but may not learn well

### Stock Market Example
States: Bear Market, Bull Market, Stagnant Market
- Bear → Bull: 0.3
- Bear → Bear: 0.5
- Bear → Stagnant: 0.2
- (with self-loops and other transitions) ✓ Correct

---

## Markov Decision Process (MDP)

### Definition (Finite MDP)
- **S** = set of states = {x₁, x₂, ..., xₙ} ✓ Correct
- **A** = set of actions = {a₁, a₂, ..., aₘ} ✓ Correct
- **Agent** interacts with environment ✓ Correct
- At time t: Sₜ ∈ States, Aₜ ∈ Actions ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK - Complete MDP Definition]:**
An MDP is formally defined by the tuple: **(S, A, T, R, γ, H, Sᵢ)**

| Component                   | Symbol         | Description                          |
|-----------------------------|----------------|--------------------------------------|
| State Space                 | S              | Set of all possible states           |
| Action Space                | A(s)           | Set of actions available in state s  |
| **Transition Function**     | **T(s,a,s')**  | **Probability P(s' \| s, a)**        |
| **Reward Function**         | **R(s,a,s')**  | **Scalar reward for transition**     |
| **Discount Factor**         | **γ**          | **Value in [0,1], typically 0.99**   |
| Horizon                     | H              | Finite or infinite time steps        |
| Initial State Distribution  | Sᵢ             | Where episodes start                 |

*Your notes covered S and A. The transition function T, reward function R, and discount factor γ are critical components covered later in the chapter.*

### Agent-Environment Interaction
- Agent can perform a set of actions A = {a₁, a₂, ..., aₘ} ✓ Correct
- Whenever an agent performs an action, it gets an **immediate reward Rₜ₊₁** ✓ Correct
- It transitions to state **Sₜ₊₁** ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK - The RL Cycle]:**
```
    ┌──────────────────────────────────────┐
    │                                      │
    ▼                                      │
  Agent ──Action Aₜ──► Environment         │
    ▲                      │               │
    │                      │               │
    └──── Sₜ₊₁, Rₜ₊₁ ◄────┘               │
                                           │
    (Repeat for each time step t) ─────────┘
```

📚 **[ADDITIONAL FROM TEXTBOOK - Experience Tuple]:**
Each interaction produces an **experience tuple**: (Sₜ, Aₜ, Rₜ₊₁, Sₜ₊₁)
- This is the fundamental unit of data for learning

### "Maze" as Markov Chain vs MDP
**Markov Chain:** States only, no agent control ✓ Correct
**MDP:** Agent chooses actions to influence transitions ✓ Correct

**Maze MDP:**
- S = {x₁, x₂, ..., x₆} ✓ Correct
- A = {↑, ↓, →, ←} ✓ Correct
- The agent performs action Aₜ in state Sₜ to make transitions happen ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK]:**
The key difference: In a Markov Chain, transitions happen automatically according to probabilities. In an MDP, the **agent chooses** which action to take, and then transitions happen according to T(s,a,s').

### Rewards
- **Sparse rewards:** No immediate reward, only on reaching goal state ✓ Correct
- **Example - Chess:** Taking pieces vs. winning ✓ Correct
  - Rewards for taking pieces vs. only reward for winning
  - Different reward structures lead to different behaviors ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK - Dense vs Sparse Rewards]:**
| Reward Type | Description                       | Pros                           | Cons                            |
|-------------|-----------------------------------|--------------------------------|---------------------------------|
| **Dense**   | Frequent non-zero rewards         | Faster learning, more guidance | More bias, less emergent behavior |
| **Sparse**  | Rare rewards (e.g., only at goal) | Novel solutions possible       | Much slower learning            |

📚 **[ADDITIONAL FROM TEXTBOOK - Reward Function Forms]:**
- **R(s, a, s')** - Most explicit (depends on full transition)
- **R(s, a)** - Marginalized over next states
- **R(s)** - Depends only on state reached

### Complex Maze Example
- **Barrier states:** Give negative reward (-10) for entering ✓ Correct
- **Goal state G:** Positive reward (+100) ✓ Correct
- **Objective:** Agent starts in state B and transitions to G most efficiently ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK]:**
Even negative values are called "rewards" in RL terminology. Think of them as costs or penalties. The agent's objective is to maximize **cumulative reward** (the return), which means avoiding penalties and reaching positive reward states efficiently.

---

## Episodes and Trajectories

### Episodic Environment
- Agent is "loose" in the maze ✓ Correct
- Episode: Complete sequence from start to terminal state ✓ Correct
- Could go on forever if agent doesn't reach goal ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK - Task Types]:**
| Task Type              | Description                        | Example            |
|------------------------|------------------------------------|--------------------|
| **Episodic**           | Has terminal states, finite        | Games, mazes       |
| **Continuing**         | No natural ending, infinite        | Robot locomotion   |
| **Indefinite Horizon** | Plans for infinite but may terminate | Most common in RL |

📚 **[ADDITIONAL FROM TEXTBOOK - Terminal State Convention]:**
Terminal states must have:
- All actions transition to themselves with probability 1
- All transitions from terminal state give reward 0

This convention ensures algorithms converge properly (avoids infinite sums).

### Trajectory
Sequence of agent's activities:
```
x₁ ↓ 0    x₄ ↑ 0    x₁ .....
S₀, A₀, R₁, S₁, A₁, R₁, S₂ ......
```
State → Action → Reward → State → Action → Reward → ... ✓ Correct

*Note: Time stamps need not be equally spaced* ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK]:**
A **trajectory** (also called a **rollout**) is the complete sequence:
```
τ = (S₀, A₀, R₁, S₁, A₁, R₂, S₂, ..., Sₜ)
```
The **return** is the sum of all rewards in a trajectory (possibly discounted).

---

## Policy

### Definition
A **policy** describes what action should be performed by the agent in which state. ✓ Correct

- A "good policy" is to be learned by an agent ✓ Correct
- Agent learns an **"optimal" policy** by training ✓ Correct
- At the end state, it needs to compute what it did and learn ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK - Policy Types]:**
| Policy Type       | Notation                    | Description                            |
|-------------------|-----------------------------|----------------------------------------|
| **Deterministic** | π(s) = a                    | Always same action in state s          |
| **Stochastic**    | π(a\|s) = P(Aₜ=a \| Sₜ=s)   | Probability distribution over actions  |

📚 **[ADDITIONAL FROM TEXTBOOK - Optimal Policy]:**
The **optimal policy π*** is the policy that maximizes expected return from every state. There may be multiple optimal policies, but they all achieve the same optimal value function.

### Generic Episode
```
S₀ → ... → Sₜ --Aₜ--> Sₜ₊₁ --Rₜ₊₁--> ... → Sₜ = goal
```
✓ Correct

---

## Value Functions

Reinforcement learning usually involves learning **two values**: ✓ Correct

### 1. State Value Function V(s)
- Value of being in state S ✓ Correct
- V(s) for all S ∈ States ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK - Formal Definition]:**
V^π(s) = Expected return starting from state s, following policy π
```
V^π(s) = E_π[Gₜ | Sₜ = s] = E_π[Σ γᵏ Rₜ₊ₖ₊₁ | Sₜ = s]
```

### 2. Action Value Function Q(S, A)
- Value of agent performing action A in state S ✓ Correct
- Q(S, A) for all state-action pairs ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK - Formal Definition]:**
Q^π(s, a) = Expected return starting from state s, taking action a, then following policy π
```
Q^π(s, a) = E_π[Gₜ | Sₜ = s, Aₜ = a]
```

📚 **[ADDITIONAL FROM TEXTBOOK - Relationship Between V and Q]:**
```
V^π(s) = Σₐ π(a|s) × Q^π(s, a)
```
The state value is the weighted average of action values under the policy.

*Some algorithms try to learn both V(s) and Q(S, A)* ✓ Correct

---

## RL Algorithm Types

### Type 1: Learning V(s) - State Values

**Example maze with learned V(s) values:**
```
x₁=90  →  x₂=100  →  x₃=G (Goal)
  ↑         ↑          ↑
x₄=81  →  x₅=90   →  x₆=100
```
✓ Correct

**Deriving policy from V(s):**
- From state S, go to neighbor state S' where **R' + V(S')** is best ✓ Correct
- Break ties randomly ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK - Why R + V(s')]:**
This formula comes from the **Bellman equation**:
```
V(s) = max_a [R(s,a,s') + γ V(s')]
```
When γ ≈ 1, the optimal action is the one that maximizes immediate reward R plus the value of the next state V(s').

### Type 2: Learning Q(S, A) - Action Values

**Example maze with Q(S, A) values:**
```
        x₁        x₂        x₃=Goal
      →:90      →:100      →:∞
      ↓:0       ↓:81       (Goal)

        x₄        x₅        x₆
      ↑:71      ↑:81      ↑:100
      →:72      →:90      →:81
      ↓:81
```
✓ Correct

*Values of actions, not immediate rewards* ✓ Correct - Important distinction!

**Deriving policy from Q(S, A):**
- In state S, choose action A that maximizes Q(S, A) ✓ Correct

📚 **[ADDITIONAL FROM TEXTBOOK - Q-Learning Preview]:**
This is the foundation of **Q-learning** and **DQN** (which you'll implement for your project!):
```
π*(s) = argmax_a Q*(s, a)
```
The optimal policy simply picks the action with highest Q-value in each state.

---

## Actor-Critic Methods

**Actor-Critic methods** compute both:
- V(s) - State values (Critic) ✓ Correct
- Q(S, A) - Action values (Actor)

⚠️ **[CLARIFICATION NEEDED]:**
The terminology in your notes is slightly imprecise. More accurately:

| Component  | What It Learns       | Role                                  |
|------------|----------------------|---------------------------------------|
| **Critic** | V(s) or Q(s,a)       | Evaluates how good states/actions are |
| **Actor**  | π(a\|s) - the policy | Decides which actions to take         |

The **Actor** learns the *policy* (mapping states → actions), not Q values directly. The **Critic** provides feedback to help the Actor improve. They work together:
1. Actor proposes an action
2. Critic evaluates "how good was that?"
3. Actor adjusts based on Critic's feedback

📚 **[ADDITIONAL FROM TEXTBOOK]:**
Actor-Critic methods combine the benefits of:
- **Value-based methods** (like Q-learning): Low variance, but high bias
- **Policy-based methods** (like REINFORCE): Low bias, but high variance

This is why Actor-Critic methods (like A3C, SAC, PPO) are popular in modern deep RL.

---

## Key Takeaways

1. **Markov Property:** Future depends only on present state, not history ✓ Correct
2. **MDP Components:** States, Actions, Transitions, Rewards ✓ Correct
3. **Policy:** Mapping from states to actions ✓ Correct
4. **Value Functions:** V(s) for states, Q(S,A) for state-action pairs ✓ Correct
5. **Goal:** Learn optimal policy through experience ✓ Correct

📚 **[ADDITIONAL KEY TAKEAWAYS FROM TEXTBOOK]:**

6. **Discount Factor (γ):** Makes future rewards less valuable than immediate rewards
   - γ = 0: Only care about immediate reward (greedy)
   - γ = 0.99: Care about long-term rewards (common choice)
   - γ = 1: No discounting (can cause infinite sums)

7. **Stochastic vs Deterministic Transitions:**
   - Deterministic: T(s,a,s') = 1 for exactly one s'
   - Stochastic: Multiple possible next states (like slippery frozen lake)

8. **The Stationarity Assumption:** Transition and reward functions don't change during training

9. **State Space Can Be:**
   - Discrete (finite states) - like grid worlds
   - Continuous (infinite states) - like robot joint angles

10. **Partially Observable MDPs (POMDPs):** When agent can't see full state
    - Observations ≠ States
    - More realistic but harder to solve

---

## Summary: Your Notes vs. Textbook

| Topic              | Your Notes            | Textbook Alignment            |
|--------------------|-----------------------|-------------------------------|
| Markov Property    | ✓ Correct             | Perfect match                 |
| MDP Definition     | Partial (S, A only)   | Add T, R, γ                   |
| Rewards            | ✓ Correct             | Add dense/sparse distinction  |
| Episodes           | ✓ Correct             | Add task types                |
| Policy             | ✓ Correct             | Add deterministic/stochastic  |
| V(s) and Q(s,a)    | ✓ Correct             | Add formal definitions        |
| Policy from V(s)   | ✓ Correct             | Add Bellman connection        |
| Policy from Q(s,a) | ✓ Correct             | Foundation of Q-learning      |
| Actor-Critic       | ⚠️ Slight imprecision | Critic=value, Actor=policy    |

**Overall Assessment:** Your lecture notes are technically sound and capture the key concepts correctly. The main areas to supplement from the textbook are the complete MDP definition (adding T, R, γ) and the Actor-Critic clarification.

---
*Enhanced from lecture notes with Chapter 2 cross-reference*
*Prepared: January 28, 2026*
