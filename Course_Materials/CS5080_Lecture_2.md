# CS 5080 - Lecture 2: Introduction to Reinforcement Learning

**Date:** Thursday, January 22, 2026, 4:42 PM
**Instructor:** Dr. Jugal Kalita

---

## Core Concept: Agent-Environment Interaction

**Reinforcement Learning** is about an **agent** that transitions from one **state** to another by taking **actions** and receiving **rewards**.

---

## Fundamental Components

### States (S)

- The set of all possible states: **S = {x₁, x₂, ..., xₙ}**
- At time t, the agent is in state **Sₜ ∈ S**
- **Sₜ is a random variable** (the state at time t)

### Actions (A)

- The set of all possible actions: **A = {a₁, a₂, a₃, ..., aₙ}**
- At time t, the agent takes action **Aₜ ∈ A**
- **Aₜ is a random variable** (the action at time t)

### Rewards (R)

- **Rₜ₊₁** = immediate reward received after performing action Aₜ in state Sₜ
- Rewards provide feedback on action quality

---

## State Transition Diagram

```
         Aₜ (action)
    ┌────────────────┐
    │                ▼
  (Sₜ) ──────────► (Sₜ₊₁)
    │
    └── Rₜ (reward for action Aₜ)
```

The agent:
1. Observes current state Sₜ
2. Takes action Aₜ
3. Receives reward Rₜ (associated with taking Aₜ in Sₜ)
4. Transitions to new state Sₜ₊₁

---

## Episodes

An **episode** is a complete sequence from start to terminal state:

```
S₀ ──(A₀, R₀)──► S₁ ──(A₁, R₁)──► S₂ ──(A₂, R₂)──► ... ──► Sₙ (terminal)
```

| State | Action | Reward | Next State |
|-------|--------|--------|------------|
| S₀ | A₀ | R₀ | S₁ |
| S₁ | A₁ | R₁ | S₂ |
| S₂ | A₂ | R₂ | S₃ |
| ... | ... | ... | ... |
| Sₙ₋₁ | Aₙ₋₁ | Rₙ₋₁ | Sₙ (terminal) |

Each step produces a tuple: **(Sₜ, Aₜ, Rₜ, Sₜ₊₁)**

---

## Finite MDPs (Markov Decision Processes)

In a **finite situation**, all three sets are finite:

| Set | Description | Example |
|-----|-------------|---------|
| **S** | States | Finite number of positions |
| **A** | Actions | Finite number of moves |
| **R** | Rewards | Finite set of reward values |

---

## Example: Maze Problem

### State Space Visualization (3 × 2 Grid)

```
┌─────┬─────┬─────┐
│ x₁  │ x₂  │ x₃  │
│ (S) │  ↔  │     │
├─────┼─────┼─────┤
│ x₄  │ x₅  │ x₆  │
│     │     │ (G) │
└─────┴─────┴─────┘

S = Start (x₁)
G = Goal (x₆)
```

### Maze Components

| Component | Definition |
|-----------|------------|
| **S** (Start) | Initial state where agent begins (x₁) |
| **G** (Goal) | Terminal state (x₆) |
| **S** (States) | S = {x₁, x₂, x₃, x₄, x₅, x₆} (6 grid positions) |
| **A** (Actions) | A = {↑, ↓, ←, →} (up, down, left, right) |
| **R** (Rewards) | R = {0, 100} (0 for moves, 100 for reaching goal) |

### How It Works

1. Agent starts at **S** (x₁, top-left)
2. Takes actions from **A** = {↑, ↓, ←, →}
3. Receives reward **0** for each step
4. Receives reward **100** upon reaching **G** (x₆, bottom-right)
5. Episode ends when goal is reached

---

## Key Takeaways

1. **RL is about sequential decision-making** - each action affects future states
2. **States are random variables** - uncertainty in where the agent ends up
3. **Actions are choices** - the agent selects from available actions
4. **Rewards guide learning** - the agent learns to maximize cumulative reward
5. **Finite MDPs** have discrete, countable states/actions/rewards

---

## Connection to Course Topics

This lecture introduces the **MDP framework** that underlies:
- Q-Learning (learning action values)
- SARSA (on-policy TD learning)
- Policy Gradient methods
- Deep RL (DQN, Actor-Critic)

**Reading:** Sutton & Barto Chapter 3 (Finite Markov Decision Processes)

---

*Notes from CS 5080 - Reinforcement Learning, Spring 2026*
