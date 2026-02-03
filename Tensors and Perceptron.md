# 📘 Tensor, Perceptron, Single-Layer & Multi-Layer Perceptron (MLP)

A **complete, beginner-to-advanced, technical yet easy-to-understand guide**, written as a **professional GitHub-ready README.md**.

---

## 📑 Table of Contents

1. What is a Tensor?
2. Why Tensors Matter in Machine Learning & Deep Learning
3. Mathematical Representation of Tensors
4. Tensor vs Vector vs Matrix
5. Tensor Operations (with Examples)
6. What is a Perceptron?
7. Mathematical Model of a Perceptron
8. Activation Functions (Detailed)
9. Limitations of a Single Perceptron
10. Single-Layer Perceptron (SLP)
11. Multi-Layer Perceptron (MLP)
12. Forward Propagation in MLP
13. Loss Function
14. Backpropagation (Concept + Math)
15. Training an MLP Step-by-Step
16. Code Examples (NumPy & Scikit-Learn)
17. SLP vs MLP Comparison Table
18. Key Takeaways

---

## 1️⃣ What is a Tensor?

A **tensor** is a **generalized mathematical structure** used to represent data in machine learning and deep learning.

> In simple terms:

* **Scalar** → single number
* **Vector** → list of numbers
* **Matrix** → table of numbers
* **Tensor** → data with **any number of dimensions**

### Examples

| Data              | Representation                     | Tensor Rank |
| ----------------- | ---------------------------------- | ----------- |
| Temperature       | 25                                 | 0 (Scalar)  |
| House prices      | [100, 200, 300]                    | 1 (Vector)  |
| Image (grayscale) | 28×28                              | 2 (Matrix)  |
| Image (RGB)       | 28×28×3                            | 3 (Tensor)  |
| Video             | Frames × Height × Width × Channels | 4 (Tensor)  |

---

## 2️⃣ Why Tensors Matter

Deep learning models **do not work on raw data**.
They operate on **tensors**.

Examples:

* Text → token embeddings → tensor
* Image → pixel values → tensor
* Audio → spectrogram → tensor

Frameworks like **TensorFlow** and **PyTorch** are built entirely around tensors.

---

## 3️⃣ Mathematical Representation of a Tensor

A tensor is represented as:

* Rank-0: `T`
* Rank-1: `T[i]`
* Rank-2: `T[i][j]`
* Rank-3: `T[i][j][k]`

Example:

```
T[batch_size][height][width][channels]
```

---

## 4️⃣ Tensor vs Vector vs Matrix

| Concept | Dimensions | Example                      |
| ------- | ---------- | ---------------------------- |
| Scalar  | 0D         | 5                            |
| Vector  | 1D         | [1, 2, 3]                    |
| Matrix  | 2D         | [[1,2],[3,4]]                |
| Tensor  | nD         | Image, Video, NLP Embeddings |

---

## 5️⃣ Tensor Operations

### Common Operations

* Addition
* Multiplication
* Dot product
* Matrix multiplication
* Broadcasting

### Example (NumPy)

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = np.dot(A, B)
print(C)
```

---

## 6️⃣ What is a Perceptron?

A **perceptron** is the **simplest neural network unit**.

> It mimics a biological neuron.

### Components

* Inputs (features)
* Weights
* Bias
* Activation function

---

## 7️⃣ Mathematical Model of a Perceptron

### Formula

```
Z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

```
Output = Activation(Z)
```

### Step-by-Step

1. Multiply inputs with weights
2. Add bias
3. Apply activation

---

## 8️⃣ Activation Functions

### Why Activation?

Without activation, a neural network becomes **just linear regression**.

### Common Activations

| Function | Formula   | Use Case              |
| -------- | --------- | --------------------- |
| Step     | 0 or 1    | Classic Perceptron    |
| Sigmoid  | 1/(1+e⁻ˣ) | Binary classification |
| ReLU     | max(0, x) | Deep networks         |
| Tanh     | (-1,1)    | Centered data         |

---

## 9️⃣ Limitations of a Single Perceptron

A single perceptron:

* Can only solve **linearly separable problems**
* Cannot solve **XOR problem**

This limitation led to **multi-layer networks**.

---

## 🔟 Single-Layer Perceptron (SLP)

### Definition

An SLP consists of:

* Input layer
* One output layer
* No hidden layer

### Architecture

```
Input → Weights → Output
```

### Use Cases

* Spam detection (simple)
* Binary classification

---

## 1️⃣1️⃣ Multi-Layer Perceptron (MLP)

### Definition

An **MLP** is a neural network with:

* Input layer
* One or more **hidden layers**
* Output layer

### Architecture

```
Input → Hidden Layer(s) → Output
```

### Why Hidden Layers Matter

They allow the network to learn:

* Non-linear relationships
* Complex decision boundaries

---

## 1️⃣2️⃣ Forward Propagation

### Steps

1. Input tensor enters the network
2. Each layer computes weighted sum
3. Activation applied
4. Output generated

### Mathematical Form

```
Z₁ = XW₁ + b₁
A₁ = ReLU(Z₁)
Z₂ = A₁W₂ + b₂
Output = Sigmoid(Z₂)
```

---

## 1️⃣3️⃣ Loss Function

Measures **how wrong the prediction is**.

### Examples

| Loss                 | Use Case              |
| -------------------- | --------------------- |
| MSE                  | Regression            |
| Binary Cross-Entropy | Binary classification |
| Categorical CE       | Multi-class           |

---

## 1️⃣4️⃣ Backpropagation

### What is Backpropagation?

A method to **update weights** using gradients.

### Steps

1. Compute loss
2. Calculate gradient of loss w.r.t weights
3. Update weights using gradient descent

```
w = w - learning_rate × gradient
```

---

## 1️⃣5️⃣ Training an MLP (Step-by-Step)

1. Initialize weights randomly
2. Forward propagation
3. Compute loss
4. Backpropagation
5. Update weights
6. Repeat for epochs

---

## 1️⃣6️⃣ Code Examples

### Single-Layer Perceptron (NumPy)

```python
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

w = np.random.rand(2)
b = 0.0
lr = 0.1

for epoch in range(100):
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = 1 if z >= 0 else 0
        error = y[i] - y_pred
        w += lr * error * X[i]
        b += lr * error

print(w, b)
```

### MLP using Scikit-Learn

```python
from sklearn.neural_network import MLPClassifier

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0]

model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=1000)
model.fit(X, y)

print(model.predict(X))
```

---

## 1️⃣7️⃣ SLP vs MLP Comparison

| Feature       | SLP          | MLP              |
| ------------- | ------------ | ---------------- |
| Hidden Layers | ❌            | ✅                |
| Non-linearity | ❌            | ✅                |
| XOR Problem   | ❌            | ✅                |
| Complexity    | Low          | High             |
| Use Case      | Simple tasks | Complex problems |

---

## 1️⃣8️⃣ Key Takeaways

* **Tensors** are the backbone of deep learning
* **Perceptron** is the basic building block
* **SLP** works only for linear problems
* **MLP** solves complex, real-world tasks
* Hidden layers + activation = power of neural networks

---

## 📌 Final Note

This README is designed to be:

* Beginner friendly
* Technically accurate
* Interview ready
* Production ready

You can directly **download, fork, or extend** this for your GitHub projects.

---

⭐ If you want: CNN, RNN, Backprop math derivation, or interview Q&A — just tell me.
