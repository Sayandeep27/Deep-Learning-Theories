# 🚀 Gradient Descent Optimizers: SGD, SGD with Momentum, AdaGrad

A **professional, GitHub‑ready README** explaining **Stochastic Gradient Descent (SGD)**, **SGD with Momentum**, and **AdaGrad** — from absolute basics to implementation — in **simple, clear language** with **mathematics intuition, diagrams explanation, real examples, and Python code**.

---

## 📌 Table of Contents

1. Introduction to Optimization in Machine Learning
2. What is Gradient Descent?
3. Batch vs Stochastic vs Mini‑Batch Gradient Descent
4. Stochastic Gradient Descent (SGD)

   * Intuition
   * Mathematical Formulation
   * Advantages & Disadvantages
   * Python Implementation (From Scratch)
   * SGD in Scikit‑Learn
5. SGD with Momentum

   * Why Momentum is Needed
   * Physical Intuition
   * Mathematical Formulation
   * Python Implementation (From Scratch)
   * SGD with Momentum in Deep Learning (PyTorch)
6. AdaGrad (Adaptive Gradient Algorithm)

   * Why Adaptive Learning Rates?
   * Mathematical Intuition
   * Python Implementation (From Scratch)
   * AdaGrad in Scikit‑Learn / PyTorch
7. Comparison Table
8. When to Use Which Optimizer?
9. Key Takeaways

---

## 1️⃣ Introduction to Optimization in Machine Learning

In machine learning, **training a model** means **finding the best parameters (weights)** that minimize a **loss function**.

This is done using **optimization algorithms**, also called **optimizers**.

👉 Popular optimizers include:

* Gradient Descent
* SGD
* Momentum
* AdaGrad
* RMSProp
* Adam

This README focuses on:

* **SGD**
* **SGD with Momentum**
* **AdaGrad**

---

## 2️⃣ What is Gradient Descent?

Gradient Descent is an **iterative optimization algorithm** used to minimize a function.

### 🔹 Core Idea

* Compute the **gradient (slope)** of the loss function
* Move **opposite to the gradient**
* Repeat until minimum loss is reached

### 🔹 Update Rule

```
w = w - learning_rate × gradient
```

Where:

* `w` → model parameters
* `learning_rate (η)` → step size
* `gradient` → derivative of loss w.r.t. parameters

---

## 3️⃣ Batch vs Stochastic vs Mini‑Batch Gradient Descent

| Type          | Data Used      | Speed    | Stability   |
| ------------- | -------------- | -------- | ----------- |
| Batch GD      | Entire dataset | Slow     | Very Stable |
| **SGD**       | Single sample  | Fast     | Noisy       |
| Mini‑Batch GD | Small batch    | Balanced | Stable      |

👉 **SGD** is widely used for **large datasets** and **deep learning**.

---

## 4️⃣ Stochastic Gradient Descent (SGD)

### 🔹 What is SGD?

**Stochastic Gradient Descent** updates model parameters using **one data point at a time**.

Instead of computing gradient over full dataset:

```
For each data point:
    compute gradient
    update weights
```

---

### 🔹 Intuition

* Faster updates
* Noisy path toward minimum
* Can escape local minima

Think of SGD like **walking downhill with random small steps**.

---

### 🔹 Mathematical Formulation

For a single training example `(xᵢ, yᵢ)`:

```
w = w - η × ∇L(w; xᵢ, yᵢ)
```

Where:

* `L` → loss function
* `η` → learning rate

---

### 🔹 Advantages

* Faster convergence for large datasets
* Requires less memory
* Can escape saddle points

### 🔹 Disadvantages

* Noisy updates
* Can oscillate near minimum
* Sensitive to learning rate

---

### 🔹 Python Implementation (From Scratch)

```python
import numpy as np

# Sample dataset
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

# Initialize weight and bias
w = 0.0
b = 0.0
lr = 0.01

# SGD training
for epoch in range(100):
    for i in range(len(X)):
        y_pred = w * X[i] + b
        error = y_pred - y[i]

        # gradients
        dw = error * X[i]
        db = error

        # update
        w -= lr * dw
        b -= lr * db

print("Weight:", w)
print("Bias:", b)
```

---

### 🔹 SGD in Scikit‑Learn

```python
from sklearn.linear_model import SGDRegressor

model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
model.fit(X, y)

print(model.coef_, model.intercept_)
```

---

## 5️⃣ SGD with Momentum

### 🔹 Why Momentum?

SGD suffers from:

* Zig‑zag movement
* Slow convergence in narrow valleys

**Momentum** helps accelerate SGD in the correct direction.

---

### 🔹 Physical Intuition

Imagine a **ball rolling downhill**:

* Gains speed
* Resists sudden direction changes

Momentum remembers **past gradients**.

---

### 🔹 Mathematical Formulation

```
v = γv + η∇L(w)
w = w - v
```

Where:

* `v` → velocity
* `γ` → momentum coefficient (usually 0.9)

---

### 🔹 Python Implementation (From Scratch)

```python
w = 0.0
b = 0.0
v_w = 0.0
v_b = 0.0
lr = 0.01
momentum = 0.9

for epoch in range(100):
    for i in range(len(X)):
        y_pred = w * X[i] + b
        error = y_pred - y[i]

        dw = error * X[i]
        db = error

        v_w = momentum * v_w + lr * dw
        v_b = momentum * v_b + lr * db

        w -= v_w
        b -= v_b

print(w, b)
```

---

### 🔹 Momentum in PyTorch

```python
import torch

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 6️⃣ AdaGrad (Adaptive Gradient Algorithm)

### 🔹 Why AdaGrad?

Different parameters may need **different learning rates**.

AdaGrad adapts learning rate **per parameter**.

---

### 🔹 Mathematical Intuition

Parameters with **large gradients** → smaller learning rate
Parameters with **small gradients** → larger learning rate

---

### 🔹 Update Rule

```
G = G + (∇L)^2
w = w - (η / √(G + ε)) × ∇L
```

Where:

* `G` → accumulated squared gradients
* `ε` → small constant for stability

---

### 🔹 Python Implementation (From Scratch)

```python
w = 0.0
b = 0.0
G_w = 0.0
G_b = 0.0
lr = 0.1
epsilon = 1e-8

for epoch in range(100):
    for i in range(len(X)):
        y_pred = w * X[i] + b
        error = y_pred - y[i]

        dw = error * X[i]
        db = error

        G_w += dw ** 2
        G_b += db ** 2

        w -= (lr / (np.sqrt(G_w) + epsilon)) * dw
        b -= (lr / (np.sqrt(G_b) + epsilon)) * db

print(w, b)
```

---

### 🔹 AdaGrad in PyTorch

```python
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
```

---

## 7️⃣ Comparison Table

| Optimizer      | Learning Rate  | Speed    | Stability | Best For        |
| -------------- | -------------- | -------- | --------- | --------------- |
| SGD            | Fixed          | Fast     | Low       | Large datasets  |
| SGD + Momentum | Fixed + memory | Faster   | Medium    | Deep networks   |
| AdaGrad        | Adaptive       | Moderate | High      | Sparse features |

---

## 8️⃣ When to Use Which Optimizer?

* **SGD** → simple problems, large datasets
* **Momentum** → deep learning, faster convergence
* **AdaGrad** → NLP, sparse data, word embeddings

---

## 9️⃣ Key Takeaways

* SGD updates weights per data point
* Momentum accelerates learning using velocity
* AdaGrad adapts learning rate automatically
* Choice of optimizer greatly affects convergence

---

## ⭐ Final Note

This README is **download‑ready**, **GitHub‑friendly**, and **production‑oriented**.

If you want:

* Adam & RMSProp
* Visual loss plots
* Interview questions
* Mathematical proofs

Just tell me — happy to extend 🚀
