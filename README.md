# 🧠 MNIST Digit Classifier – Neural Network from Scratch (NumPy Only)

## 📌 Project Overview

This project demonstrates how to build a simple neural network from scratch using only NumPy to classify handwritten digits from the MNIST dataset. No external deep learning libraries like TensorFlow or PyTorch are used. The network is trained on 28x28 grayscale images to predict digits from 0 to 9.

This is an educational project meant to illustrate the **inner workings of neural networks**—from forward propagation and activation functions to backpropagation and weight updates.

---

## 🗂️ Repository Structure

```
📁 mnist-numpy-nn/
│
├── neural_network_from_scratch.ipynb     # Jupyter notebook with training, evaluation, and plotting
├── mnist_model.npz                       # Saved model weights (optional)
├── README.md                             # Project description and documentation
└── requirements.txt                      # Required libraries (NumPy, matplotlib, etc.)
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook

Open the notebook `neural_network_from_scratch.ipynb` and run all cells. It will:
- Load and preprocess the MNIST dataset
- Build and train a 2-layer neural network
- Plot training accuracy and loss
- Save the model weights

---

## 🧮 Math Behind the Model

### 1. Input and Preprocessing

- Each MNIST image is 28×28 pixels, flattened into a 784-dimensional vector.
- Pixel values are normalized to range [0, 1].

### 2. Architecture

- **Input Layer**: 784 neurons (one per pixel)
- **Hidden Layer**: 64 neurons (ReLU activation)
- **Output Layer**: 10 neurons (one per digit class, softmax activation)

### 3. Forward Propagation

Let:
- \( X \): Input data of shape (m, 784)
- \( W_1 \): Weights from input to hidden layer
- \( b_1 \): Bias for hidden layer
- \( W_2 \): Weights from hidden to output layer
- \( b_2 \): Bias for output layer

Then:

**Hidden layer:**

\[
Z_1 = XW_1 + b_1 \\
A_1 = \text{ReLU}(Z_1) = \max(0, Z_1)
\]

**Output layer:**

\[
Z_2 = A_1W_2 + b_2 \\
\hat{Y} = \text{Softmax}(Z_2)
\]

### 4. Loss Function (Categorical Cross Entropy)

\[
L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{10} Y_{ij} \log(\hat{Y}_{ij})
\]

### 5. Backpropagation

- Derivatives are calculated for each layer:

\[
\frac{\partial L}{\partial Z_2} = \hat{Y} - Y \\
\frac{\partial L}{\partial W_2} = A_1^T \cdot \frac{\partial L}{\partial Z_2} \\
\frac{\partial L}{\partial A_1} = \frac{\partial L}{\partial Z_2} \cdot W_2^T \\
\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \cdot \text{ReLU}'(Z_1)
\]

- Parameters are updated using gradient descent:

\[
W := W - \alpha \cdot \frac{\partial L}{\partial W}
\]

---

## 📊 Training Performance

- **Loss** and **accuracy** are plotted after each epoch.
- Evaluation is done on a validation set (development set).
- Accuracy is expected to reach ~85-90% after tuning.

---

## 💾 Saving and Loading Model

To save the model:

```python
np.savez('mnist_model.npz', W1=W1, b1=b1, W2=W2, b2=b2)
```

To load later:

```python
model = np.load('mnist_model.npz')
W1 = model['W1']
b1 = model['b1']
W2 = model['W2']
b2 = model['b2']
```

---

## 🧠 Why Build From Scratch?

This project helps understand:
- How backpropagation works
- What activation functions do
- How gradients are computed and applied
- Why architectures affect learning performance

---

## 📈 Sample Results

![Training Plot](training_metrics.png)

> 📌 *You can save the loss/accuracy plot as PNG inside the notebook using `plt.savefig('training_accuracy_loss_plot.png')`.*

---

## ✅ To-Do / Future Work

- Add more hidden layers (deep network)
- Add dropout or batch normalization
- Convert to object-oriented code (NeuralNetwork class)
- Try other datasets (e.g., Fashion MNIST)

---

## 🙌 Acknowledgements

- MNIST dataset from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)
- Inspired by educational deep learning tutorials
