# CS 5080 Lecture 16 — Policy in RL / Intro to Policy Gradient Methods

**Date:** Tuesday, April 14, 2026
**Source:** Josh's handwritten notes (`CS_5080_Lecture_16.pdf`)

> *This lecture seems to be mainly a review — part of the first half.*

---

## Policy in RL

**Deterministic policy:**

π(s) = a, ∀s ∈ S
- s ∈ S
- a ∈ A
- Probability of 1 (action fixed given state)

**Non-deterministic policy:**

π(a|s) = prob(A_t = a | S_t = s)
- Probability not fixed

### Learning policy via learning values

Three approaches:

1. **Learn v(s) and then use dynamics with v values to learn policy**
   - Looking at page 80 in the book — Policy iteration, Chap 4.2

2. **Learn Q(s,a) values → learn policy**
   - Looking at Chap 5, Monte Carlo (book page 99, Monte Carlo E)
   - Looking at page 100

3. **Learn values Q(s,a) values, no policy learning**
   - Learn a policy by learned values
   - Looking at Chap 6, page 130 — Sarsa, TD algorithms
   - Page 131 — Q-learning — no policy at all

---

## Schools of Thought

1. Learn policy via values
   - a) Learn values only
   - b) Learn policy directly without values
2. Learn policy with the help of values
3. Learn policy via LLM — cheat? Easy way out? Lol

**Policy is easier to learn than values?**
- Some papers mention this
- Theory: learning policy is better... need empirical data

**Non-deterministic:** many claim only policy learning can do it, but if we have learned Q values:

π(s,a) = e^Q(s,a) / Σ_{a' ∈ A} e^Q(s,a')

(Softmax over Q-values — gives a valid non-deterministic policy from learned Q values.)

---

## ★ Start New Material ★ — Chapter 13: Policy Gradient Methods

**Learn policy directly.**

### High-level algorithm

1. Initialize π — randomly or to all very small values
2. Iterate over some experience (episodes):
   - Update π
3. Repeat

---

## Updating of π

Gradient / partial derivatives:

π_new ← π_old + α · ∇J(θ)

- **α** = learning rate
- **J** = function that measures *goodness* of the policy
- **θ** = a vector of parameters

### ANN-based policy

Diagram: ANN takes states s_1, s_2, ..., s_n as inputs → outputs π(s,a)
- Compute J
- Back-propagate −J (or +∇J to ascend)

### Questions raised in lecture

1. Can we use a goodness function to train an ANN instead of a loss or badness function?
2. What kind of data will replace tabular labeled data in supervised learning?
3. How should the update function look?
4. What are possible goodness / measure functions?

---

## Review: Regular ANN (Supervised Learning)

**Setup:**
- Data set with labeled features (or random values y)
- Feed to ANN
- Result comes out: ŷ (predicted value)
- Loss function: L(y, ŷ) → measure of badness

### Gradient Descent in Supervised ML (Training ANNs)

**Assume an ANN with 1 parameter θ:**

| Feature x | Label y |
|-----------|---------|
| 1 | 5 |
| 2 | 3.2 |
| 0 | 10 |
| 2 | 4 |
| ... | ... |
| n | n_2 |

- x →[θ]→ y (value from table = actual value)
- ŷ (value from ANN = predicted value)
- L(y, ŷ) = ½(y − ŷ)² — squared-error loss

### Stochastic Gradient Descent

1. θ_0 ← init
2. i ← 0
3. Repeat:
   - θ_{i+1} ← θ_i + α · (−dL/dθ)  (move against the gradient)
   - i++
4. Until no improvement or N iterations

> **[claude.ai]** Note in margin: "looks like we are 5610 territory here" — i.e., this is where the convex optimization math from CS 5610 overlaps directly with RL.

**Assume the loss function is convex:**
- Plot of L vs θ — bowl shape with minimum at θ*
- θ_0 (initialize / guess a solution)
- i → 0
- Repeat:
  - Update: θ_{i+1} ← θ + (update to θ)
  - Until no improvement possible or stop at a predetermined point
- L = ½(y − f̂(x; θ))²  where  f̂(x; θ) = ŷ

### Multi-parameter ANN (vector θ)

Diagram: ANN with multiple inputs and outputs.

**Init:** θ⃗ = [θ_0, θ_1, ..., θ_n]

3. θ⃗ — init all params randomly
   - Repeat:
     - [θ_0; θ_1; ...; θ_n] ← [θ_0; θ_1; ...; θ_n] − α · [dL/dθ_0; dL/dθ_1; ...; dL/dθ_n]
     - Until ...

### Stochastic GD (diagram)

Plot shows J vs θ (inverted bowl / hill shape in this case — gradient *ascent* for a goodness function).

4. θ⃗ — initialize
   - Repeat:
     - θ⃗ ← θ⃗ − α · ∇L | x⃗
     - Until ...

### RL Update (using goodness function J)

5. **RL update:**
   - θ⃗ ← init
   - Repeat:
     - θ⃗ ← θ⃗ − α · (−∇J) | experience
     - Until ...

(Here −(−∇J) = +∇J, i.e., gradient *ascent* on J — we want to *maximize* the goodness of the policy, not minimize a loss.)

---

## Key Takeaways

- **Policy gradient = direct policy learning** via parameter updates along the gradient of a goodness function J(θ).
- **J, not L:** instead of minimizing a loss, we maximize a goodness function → gradient *ascent*, not descent (flip the sign).
- **θ is a parameter vector** (often the weights of an ANN that represents the policy).
- **Three RL schools** of value/policy learning were reviewed before the pivot to Chap 13.
- **Softmax over Q-values** gives a non-deterministic policy from learned Q values — a bridge between value-based and policy-based methods.
- **Overlap with CS 5610:** the gradient-descent / convex-optimization machinery is the same — RL adapts it to maximizing J(θ) from sampled experience rather than minimizing loss on labeled data.
