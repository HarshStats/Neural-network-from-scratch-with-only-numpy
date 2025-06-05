# Neural Network from Scratch with NumPy

## Overview

This project demonstrates how to build and train a simple neural network from scratch using only NumPy, without relying on high-level deep learning libraries such as TensorFlow or Keras. The network is applied to the MNIST dataset, a widely used benchmark for handwritten digit recognition. The goal is to provide a clear, educational implementation that covers all the essential steps of building, training, and evaluating a neural network.

---

## Project Structure

- **Data Preparation:**  
  The MNIST dataset is loaded, shuffled, and split into training, validation (dev), and test sets.

- **Network Architecture:**  
  The neural network consists of one hidden layer with a tunable number of neurons (hidden size), using ReLU activation, and an output layer with softmax activation for multi-class classification.

- **Training:**  
  The model is trained using gradient descent, with support for L2 regularization and hyperparameter tuning (hidden layer size and regularization strength).

- **Evaluation:**  
  The model's performance is evaluated on both the validation and test sets, and training metrics are visualized.

- **Model Saving/Loading:**  
  The trained model parameters are saved and can be reloaded for future use.

---

## Results

### Training Metrics

The following plot shows the training accuracy and loss over the course of training:

![Training Metrics](training_metrics.png)

- **Left:** Training accuracy vs. iterations  
- **Right:** Training loss vs. iterations

### Model Performance

After hyperparameter tuning, the best hidden layer size and regularization strength were selected. The final model was evaluated as follows:

- **Validation (Dev) Accuracy:**  
  _e.g.,_ `0.92` (replace with your actual result)

- **Test Accuracy:**  
  _e.g.,_ `0.91` (replace with your actual result)

Sample predictions and their corresponding true labels are printed in the notebook for qualitative assessment.

---

## How to Run

1. Place the MNIST `train.csv` file in the project directory.
2. Run the notebook cells sequentially.
3. The training metrics plot will be saved as `training_metrics.png`.
4. The trained model will be saved as `mnist_model.pkl`.

---

## Author

**Harsh**
