# CS 5080 — Lecture 15: Deep Q-Learning (Atari / Mnih et al.)
**Date:** Tuesday, March 17, 2026

## HW2 Notes

- HW2 should be a **7x6 grid**, not 6x7 like in the text
- Can change dimensions per 1(a)
- Need to use CS 4300 principles — SOFA, keep it simple
- The professor will tell us which parking spot to park in during the demo
- PDF for this assignment is in the homework folder

## Atari Games Played by Deep Q-Learning (Mnih et al., 2013)

- Covered Atari Pong, Breakout, Space Invaders, Seaquest, Beam Rider

## Tabular Q-Learning (Review)

From Sutton & Barto:

    Q(S,A) <-- Q(S,A) + alpha * [R + gamma * max_a Q(S',a) - Q(S,A)]

- Q table used in **two places:**
  1. To **choose action** (epsilon-greedy)
  2. To **update/learn**
- "This looks familiar" — same algorithm from earlier lectures
- Access Q-table to obtain all actions in a state, find best action, give highest probability of being chosen, give a little chance to all actions (exploration)
- Learning by the agent happens **once per step**

## From Q-Table to Q-Neural Network

- Want to replace Q-table with a NN — need a Q-neural network
- Given a state (frame of a game) → get all actions available in that state
- Q(s,a) values → regression learning
- Get max Q(s,a) value, assign probabilities to each of the actions
- Take action a and observe R, S'
- Feed state S' into NN again but this time **with loss function**

## Deep Q-Learning Algorithm (Algorithm 1 from Mnih)

1. Initialize replay memory D to capacity N
2. Initialize action-value function Q with random weights

Key ideas:

- **Learning is decoupled** from what it's doing right now (experience replay)
- Store transitions (phi_t, a_t, r_t, phi_{t+1}) in D
- Sample random minibatch from D
- Gradient descent on (y_j - Q(phi_j, a_j; theta))^2
- **Q-table is replaced by a neural network** in the same two places:
  1. Action selection (epsilon-greedy over Q-network outputs)
  2. Target computation (max_a Q(S',a) from the network)
