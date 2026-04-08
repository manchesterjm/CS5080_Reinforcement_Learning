# CS 5080 — Lecture 14: Introduction to Neural Networks (Kalita's Slides)
**Date:** Thursday, March 12, 2026

*This lecture used Prof. Kalita's "Introduction to Neural Networks" slide deck. Notes below combine slide content with handwritten annotations.*

## Specifying a Function in a Table

- We can easily specify a function in a table (like Excel)
- Four data examples, each with two features (x₁, x₂) and a computed output

**Linear functions:**

| x₁ | x₂ | f(x₁,x₂) = x₁ + x₂ |
|----|----|-----------------------|
| 2  | 3  | 5                     |
| 5  | 7  | 12                    |
| 2  | 0  | 2                     |
| 9  | 6  | 15                    |

| x₁ | x₂ | f(x₁,x₂) = 2x₁ + 3x₂ + 1 |
|----|----|-----------------------------|
| 2  | 3  | 14                          |
| 5  | 7  | 32                          |
| 2  | 0  | 5                           |
| 9  | 6  | 37                          |

**Non-linear functions:**

| x₁ | x₂ | f(x₁,x₂) = x₁ × x₂ |
|----|----|-----------------------|
| 2  | 3  | 6                     |
| 5  | 7  | 35                    |
| 2  | 0  | 0                     |
| 9  | 6  | 54                    |

| x₁ | x₂ | f(x₁,x₂) = x₁² + x₂² |
|----|----|-------------------------|
| 2  | 3  | 13                      |
| 5  | 7  | 74                      |
| 2  | 0  | 4                       |
| 9  | 6  | 117                     |

## Machine Learning: Empirically Discovering a Function

- Now suppose we are **not given** the function — we need to discover it
- In each top table, there is a linear function to discover
- In each bottom table, there is a non-linear function to discover
- Discovering functions from data is **machine learning**
- The learned function can be used to "predict" values for unseen arguments
- Discovering functions where the resulting value is numeric is called **Regression**

## Machine Learning

- We have a dataset with features (observations, measurements) about data examples
- We assume there is an unknown function of the features hidden in the dataset
- We need to discover or learn it
- The function can be linear or non-linear, usually highly non-linear
- The learned function can then obtain values for new previously unseen data examples
- In real examples, the number of features can be in the tens, hundreds, thousands, millions, or even billions of parameters

### Noisy Measurements

- Features are observations/measurements that can be observed or measured or transcribed incorrectly
- Some measurements can be quite good, others not, and a small number can be quite off or be outliers
- We still need to find the function
- The number of possible functions to discover is infinite — need to find the "best" match according to some criteria

## Machine Learning: Scaling Up

### Boston Housing Dataset

- Derived from U.S. Census Service information concerning housing in Boston, Mass.
- Small dataset: only 506 cases
- Features: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV
- Originally published by Harrison & Rubinfeld, "Hedonic prices and the demand for clean air", J. Environ. Economics & Management, vol.5, 81-102, 1978

### MNIST Database

- Modified National Institute of Standards and Technology database
- Large database of handwritten digits for training image processing systems
- 60,000 training images, 10,000 testing images
- Each sample is a 28×28 image (center of mass of pixels is in center)
- Each digit = 28×28 = 784 numbers between 0 and 255, with a label (the actual digit) as the last column
- Goal: find the function that separates digits from one another as accurately as possible

## Artificial Neural Networks Learn Approximate Functions

- **Classification:** x is an image, and y is label or type of image
  - f*(image) = cat, f*(image) = dog

## ANNs in Reinforcement Learning

- **Classification:** x is an image, and y is the action to perform
- Generic policy learning neural network: State → Deep Neural Network (Policy π) → Action, with Environment providing Reward

## Biological Inspiration

- Computer programs are brittle, break often, cannot solve many problems humans solve easily
- Goal: make computers more robust, intelligent, endow with ability to learn
- Approach: model computer software (and/or hardware) nominally after the brain

## Neurons in the Brain

- ~100 billion neurons in the human brain, each connected to 3-4000 others on average
- Components:
  - **Dendrite** — fibers connected to soma
  - **Synapse** — connecting point between axons
  - **Soma** — body of neuron (cell body)
  - **Nucleus** — center of cell
  - **Axon** — connection to other neurons
  - **Axonal arborization** — branching at end of axon

## Natural Neural Networks

- Signals move via electrochemical pulses (electrical and chemical signals working together)
- A neuron receives input from other neurons (maybe hundreds or thousands) from its synapses, 3-4000 on average
- Synapses release a chemical transmitter — the sum can cause a threshold to be reached — causing the neuron to "fire"
- Inputs are summed by a neuron
  - When input exceeds a threshold, the neuron sends an electrical spike from the body, down the axon, to the next neuron(s)
- Synapses or connections between neurons have different conduction strengths that modify input values (weighted)
- Synapses can be inhibitory or excitatory

## Artificial Neural Networks

- **McCulloch & Pitts (1943)** — generally recognized as designers of the first ANN
- Key ideas still used today:
  - Many simple units ("neurons") combine to give increased computational power
  - Some connections are positively weighted, others negatively
  - They introduced the idea of a **threshold** needed for activation of a neuron

## Modelling a Neuron

- Inputs x₁, x₂, ..., xₙ with weights w₁, w₂, ..., wₙ and bias b (x₀=1)
- Linear function z = Σ(wᵢxᵢ) + b
- Activation function f(z) → output y

## Characterizing an ANN

1. **Activation Function** at single neuron level — Step, sign, sigmoid, ReLU, etc.
2. **Architecture** at network level — how neurons are organized, how they are connected
3. **Learning Algorithm** at network level — how connection weights are changed/learned over time; standardly backpropagation (gradient descent); used with various optimization functions

## Examples of Activation Functions

| Activation Function          | Equation                                          | Example Use                       |
|------------------------------|---------------------------------------------------|-----------------------------------|
| Unit step (Heaviside)        | φ(z) = 0 if z<0, 0.5 if z=0, 1 if z>0            | Perceptron variant                |
| Sign (Signum)                | φ(z) = -1 if z<0, 0 if z=0, 1 if z>0              | Perceptron variant                |
| Linear                       | φ(z) = z                                          | Adaline, linear regression        |
| Piece-wise linear            | φ(z) = 1 if z≥½, z+½ if -½<z<½, 0 if z≤-½        | Support vector machine            |
| Logistic (sigmoid)           | φ(z) = 1/(1+e⁻ᶻ)                                 | Logistic regression, Multi-layer  |
| Hyperbolic tangent           | φ(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ)                        | Multi-layer NN                    |
| Rectifier, ReLU              | φ(z) = max(0, z)                                  | Multi-layer NN                    |
| Rectifier, softplus          | φ(z) = ln(1+eᶻ)                                  | Multi-layer NN                    |

## Learning: Setting the Weights

- **Supervised:** Classification — provide pre-labeled/pre-classified examples. Each example is a pair ⟨input (features), label⟩
- **Unsupervised:** No labeled examples. Group instances based on "inherent similarity". Example: Clustering
