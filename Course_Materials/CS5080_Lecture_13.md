# CS 5080 — Lecture 13: Neural Networks Continued (CNNs)
**Date:** Tuesday, March 10, 2026

## Dense NN Scaling Problem

- 250x250 pixel RGB picture → 250×250×3 = 187,500 input layer nodes
- 120 outputs
- 250K nodes in hidden layer
- 50×10⁹ connections — huge amount
- Dense NN is a universal function approximator, but this doesn't scale

## Convolutional Neural Networks (CNNs)

### Receptive Fields in Vision

- Nodes stacked like a pyramid scheme or management scheme
  - Layers stacked in a hierarchy
- Gets to use a **division of labor**
- Can be better in time, and less computational overhead
- Limited and controlled redundancy, unlike dense systems (fully connected)

### Convolutions in Computer Vision

- Simple box blur convolution — avg effect produces a slight blur
- Convolution ≈ small transformer
- Low-level processing → low-level feature extraction
- High-level processing → high-level feature extraction
- Detecting horizontal lines — only horizontal lines
- Edge detection — all lines no matter the orientation

### Hadamard Product

*[Claude: The Hadamard product (also called element-wise product or Schur product) is the operation that takes two matrices of the same dimensions and produces another matrix where each element is the product of the corresponding elements. For matrices A and B: (A ⊙ B)ᵢⱼ = Aᵢⱼ · Bᵢⱼ. In CNNs, this is used during the convolution operation — the filter/kernel is element-wise multiplied with the corresponding patch of the input, then the results are summed to produce one value in the activation map.]*

- Number of convolutions are **learned**, not given

### Introducing Receptive Fields: Convolutions

- A small filter that is applied at points of the image and slid over the image
- Convolution will be a matrix
- Convolution will produce an **activation map**
- Each filter will produce its own activation map

*[Claude: A tensor is a multi-dimensional array — a generalization of scalars (0D), vectors (1D), and matrices (2D) to arbitrary dimensions. In CNNs, the input image is typically a 3D tensor (height × width × channels), and the output of a convolutional layer is also a 3D tensor (height × width × number of filters). PyTorch and TensorFlow represent all data as tensors.]*

### Dimensions of Input/Output Volumes

- **Stride of 1** — move the filter one pixel at a time

*[Note: "What class did I take that has stride as a term? I think it was operating systems CS4500" — stride is used in both OS (memory stride) and CNNs (filter stride), but the convolution context here is specific to neural networks.]*

- Would need padding maybe
- Example: 32×32×3 → 1×28×28 (with a 5×5 filter, no padding, stride 1)

### Basic CNN Architecture

- Input layer → ... → output layer
- Feature extraction stages → classification at output
- Output: what am I looking at (classification) or regression (what we want the CNN to do)

### Other Layers

- **Dropout layer** — used to prevent overfitting
  - Training phase vs Testing phase
  - Learn less so we can apply the model to other datasets
  - Zeros some nodes out randomly

*[Claude: Dropout works by randomly setting a fraction of neuron outputs to zero during each training step. This forces the network to learn redundant representations and prevents co-adaptation of neurons. During training, each neuron has probability p of being "dropped" (typically p=0.5 for hidden layers, p=0.2 for input). During testing/inference, all neurons are active but outputs are scaled by (1-p) to compensate. This acts as an implicit ensemble of many sub-networks.]*

- **Pooling layer** (downsampling)
  - Max pooling is commonly used — get the largest value in a small sub-region
  - Purpose: reduce the bloat of layers to enhance computation
  - Provides **translation invariance**

### Softmax Layer

- Used as the activation function in the last conv layer or classification layer
- Squashes values to produce probabilities (normalization)
- Gives values that sum to one
- Each output is positive, between [0,1], all outputs sum to 1
- Not real probabilities — pseudo-probabilities
  - But we can think of them as probabilities
- Makes big numbers bigger, small numbers smaller
- Makes the winner stand out — no doubt who wins

### Layer Capture Features

- **LeNet:** First CNN architecture — LeCun, Bottou, Bengio, 1998 (IEEE)

### ImageNet and Competitions

- ImageNet dataset — large dataset — and competitions (ILSVRC)
- Images were classified by humans
- **WordNet:** lexicon database — 145K unique words — organized into 176K synsets
- **AlexNet:** ILSVRC 2012 winner — Hinton 2012, Toronto, Canada
  - Neuro Net revolution — 2012 paper
- **ResNet:** 2015 winner
