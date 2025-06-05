# Neural Network from Scratch with NumPy

## Introduction

This notebook demonstrates how to build a simple neural network from scratch using only NumPy, without relying on high-level deep learning libraries such as TensorFlow or Keras. The goal is to gain a deeper understanding of the core concepts behind neural networks by implementing each component manually.

We will apply our neural network to the MNIST dataset, a classic benchmark for handwritten digit recognition. By the end of this notebook, you will understand how neural networks learn from data and how to implement them from first principles.

---

## Plan of Action

1. **Import Libraries**  
   Load all necessary Python libraries (NumPy, Pandas, Matplotlib, etc.).

2. **Data Preparation**  
   Load the MNIST dataset, shuffle, and split into training, validation, and test sets.

3. **Network Initialization**  
   Define functions to initialize weights and biases for the neural network.

4. **Activation Functions**  
   Implement ReLU and softmax activation functions and their derivatives.

5. **Forward and Backward Propagation**  
   Build functions for forward pass, backward pass, and parameter updates.

6. **Utilities**  
   Add helper functions for predictions, accuracy, and loss calculation.

7. **Training Loop**  
   Implement the main training loop with loss and accuracy tracking.

8. **Hyperparameter Tuning**  
   Experiment with different hidden layer sizes and regularization strengths.

9. **Evaluation**  
   Evaluate the trained model on validation and test sets.

10. **Visualization**  
    Plot training metrics and visualize predictions.

11. **Model Saving/Loading**  
    Save and reload the trained model using pickle.
