# CS 5080 Lecture 3 - Markov Decision Processes
**Date:** Tuesday, January 27, 2026, 4:45 PM
**Source:** Chapter 2 of Grokking Deep Reinforcement Learning (Morales)

---

## Markov Chains

### Weather Example
- **States:** S = {cold, rain, sun}
- **Transition Table T:**

| From/To | Cold | Rain | Sun |
|---------|------|------|-----|
| Cold | 0.2 | 0.5 | 0.3 |
| Rain | 0.5 | 0.3 | 0.2 |
| Sun | 0.5 | 0.1 | 0.4 |

*Note: Each row sums to 1*

### Markov Property (Simplifying Assumption)
- **The current state encapsulates all effects of history**
- Next transition depends *only* on the current state
- History of the chain: C, R, R, R, S, S, C, S, ...
  - "What will it be tomorrow?" depends on current state only

### Stock Market Example
States: Bear Market, Bull Market, Stagnant Market
- Bear → Bull: 0.3
- Bear → Bear: 0.5
- Bear → Stagnant: 0.2
- (with self-loops and other transitions)

---

## Markov Decision Process (MDP)

### Definition (Finite MDP)
- **S** = set of states = {x₁, x₂, ..., xₙ}
- **A** = set of actions = {a₁, a₂, ..., aₘ}
- **Agent** interacts with environment
- At time t: Sₜ ∈ States, Aₜ ∈ Actions

### Agent-Environment Interaction
- Agent can perform a set of actions A = {a₁, a₂, ..., aₘ}
- Whenever an agent performs an action, it gets an **immediate reward Rₜ₊₁**
- It transitions to state **Sₜ₊₁**

### "Maze" as Markov Chain vs MDP
**Markov Chain:** States only, no agent control
**MDP:** Agent chooses actions to influence transitions

**Maze MDP:**
- S = {x₁, x₂, ..., x₆}
- A = {↑, ↓, →, ←}
- The agent performs action Aₜ in state Sₜ to make transitions happen

### Rewards
- **Sparse rewards:** No immediate reward, only on reaching goal state
- **Example - Chess:** Taking pieces vs. winning
  - Rewards for taking pieces vs. only reward for winning
  - Different reward structures lead to different behaviors

### Complex Maze Example
- **Barrier states:** Give negative reward (-10) for entering
- **Goal state G:** Positive reward (+100)
- **Objective:** Agent starts in state B and transitions to G most efficiently

---

## Episodes and Trajectories

### Episodic Environment
- Agent is "loose" in the maze
- Episode: Complete sequence from start to terminal state
- Could go on forever if agent doesn't reach goal

### Trajectory
Sequence of agent's activities:
```
x₁ ↓ 0    x₄ ↑ 0    x₁ .....
S₀, A₀, R₁, S₁, A₁, R₁, S₂ ......
```
State → Action → Reward → State → Action → Reward → ...

*Note: Time stamps need not be equally spaced*

---

## Policy

### Definition
A **policy** describes what action should be performed by the agent in which state.

- A "good policy" is to be learned by an agent
- Agent learns an **"optimal" policy** by training
- At the end state, it needs to compute what it did and learn

### Generic Episode
```
S₀ → ... → Sₜ --Aₜ--> Sₜ₊₁ --Rₜ₊₁--> ... → Sₜ = goal
```

---

## Value Functions

Reinforcement learning usually involves learning **two values**:

### 1. State Value Function V(s)
- Value of being in state S
- V(s) for all S ∈ States

### 2. Action Value Function Q(S, A)
- Value of agent performing action A in state S
- Q(S, A) for all state-action pairs

*Some algorithms try to learn both V(s) and Q(S, A)*

---

## RL Algorithm Types

### Type 1: Learning V(s) - State Values

**Example maze with learned V(s) values:**
```
x₁=90  →  x₂=100  →  x₃=G (Goal)
  ↑         ↑          ↑
x₄=81  →  x₅=90   →  x₆=100
```

**Deriving policy from V(s):**
- From state S, go to neighbor state S' where **R' + V(S')** is best
- Break ties randomly

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

*Values of actions, not immediate rewards*

**Deriving policy from Q(S, A):**
- In state S, choose action A that maximizes Q(S, A)

---

## Actor-Critic Methods

**Actor-Critic methods** compute both:
- V(s) - State values (Critic)
- Q(S, A) - Action values (Actor)

---

## Key Takeaways

1. **Markov Property:** Future depends only on present state, not history
2. **MDP Components:** States, Actions, Transitions, Rewards
3. **Policy:** Mapping from states to actions
4. **Value Functions:** V(s) for states, Q(S,A) for state-action pairs
5. **Goal:** Learn optimal policy through experience

---
*Transcribed from handwritten lecture notes*
