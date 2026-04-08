# CS 5080 - Lecture 12
**Date:** Thursday, March 5, 2026

## Neural Networks

### From Tables to Function Approximation

- **Tabular approach:** State input → Q(s,a) table → output (discrete functions)
- **Function approximation:** Convolutional neural net as a generalized format of Q(s,a) table

### Deep Q-Learning

- Throw multiple states into the CNN or Q(s,a) table
- Has a history/memory of prior actions

### ANNs in RL

**Generic policy learning neural net:**
- Agent receives state from environment
- Agent outputs action to environment
- Environment returns reward to agent
- (Standard RL loop diagram)

---

## Biological Neural Networks

**Neurons in the brain:** ~100 billion neurons, each connected to 3,000–4,000 other neurons

- **Dendrite** — fibers connected to soma
- **Synapse** — connecting point between axons
- **Soma** — body of neuron
- **Nucleus** — center of cell
- **Axon** — connection to other neurons
- **Axonal arborization** — branching of axon terminals

**Signals:** Electrochemical process
**Synapses:** Communicate via chemical transmitters; can be **inhibitory** or **excitatory**

---

## Artificial Neural Networks (ANNs)

**McCulloch and Pitts (1943)** — generalized designers of first ANN

### Modeling the Neuron

```
inputs (weighted) → [Σ] → [f] → y (output)
                     linear    activation
                     function  function
                     (performs the weighted sum)
```

### Characterizing an ANN

1. **Activation function** at a single neuron level — step, sign, sigmoid, ...
2. **Architecture** at network level — organized/connected
3. **Learning algorithm** at network level — connections/weights changed/learned over time

### Examples of Activation Functions

> **[From Sutton & Barto, Table 9.1 and common references]**
>
> | Function    | Formula                          | Range     | Notes                                    |
> |-------------|----------------------------------|-----------|------------------------------------------|
> | Step        | f(x) = 1 if x ≥ 0, else 0       | {0, 1}    | Original McCulloch-Pitts                 |
> | Sign        | f(x) = +1 if x ≥ 0, else -1     | {-1, +1}  | Bipolar step                             |
> | Sigmoid     | f(x) = 1/(1 + e^(-x))           | (0, 1)    | Smooth, differentiable                   |
> | Tanh        | f(x) = (e^x - e^(-x))/(e^x + e^(-x)) | (-1, 1) | Zero-centered sigmoid              |
> | ReLU        | f(x) = max(0, x)                | [0, ∞)    | Hinton — Toronto; most widely used today |
> | Leaky ReLU  | f(x) = x if x > 0, else 0.01x   | (-∞, ∞)   | Avoids dying ReLU problem                |
> | Softmax     | f(x_i) = e^(x_i) / Σe^(x_j)    | (0, 1)    | Output layer for classification          |

- **ReLU** — Hinton, Toronto

### Learning — Setting the Weights

- **Supervised training** — labeled data
- **Unsupervised** — has to figure out everything on its own

### Rules

- **Back propagation** — see AI slides from last semester
- **Loss function** — again, see slides from last semester
- **Epoch of training** — hopefully it learns something every epoch

---

## Architecture

### Single Layer: Feed-forward ANN or Perceptron
- Laurene Fausett, 1990
- Classification of letters

### Multi-layer Neural Networks
- **Input layer**
- **Hidden layer(s)**
- **Output layer**

> **Universal Function Approximation Theorem:**
> A feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of R^n, under mild assumptions on the activation function (Cybenko, 1989; Hornik et al., 1989).
>
> In practice, while one hidden layer is *theoretically* sufficient, it is **not practical to compute** — the number of neurons required can be astronomically large. This is why deep networks (multiple hidden layers) are used: they achieve better approximations with far fewer total parameters.

### Receptive Fields of Vision
- N (noted, not elaborated)
