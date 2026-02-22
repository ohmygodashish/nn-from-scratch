"""
Simple neural network from scratch for MNIST digit classification.
Uses a 2-layer fully connected network with ReLU and softmax activations.
"""
import argparse
import numpy as np
import fireducks.pandas as pd

from utils.activations import d_ReLU, ReLU, softmax
from utils.encoder import one_hot
from utils.loader import data_loader
from utils.weights import init_params, update_params

def forward_prop(W1, b1, W2, b2, X):
    """
    Perform forward propagation through the network.
    
    Args:
        W1, b1: First layer weights and biases
        W2, b2: Second layer weights and biases
        X: Input data matrix
    """
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    """
    Compute gradients via backpropagation.
    
    Args:
        Z1, A1: First layer pre-activation and activation
        Z2, A2: Second layer pre-activation and activation
        W1, W2: Layer weights
        X, Y: Input data and labels
    """
    one_hot_Y = one_hot(Y)
    m_train = Y.size

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m_train * dZ2.dot(A1.T)
    db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * d_ReLU(Z1)
    dW1 = 1 / m_train * dZ1.dot(X.T)
    db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

def get_predictions(A2):
    """
    Extract predicted class labels from output probabilities.
    
    Args:
        A2: Output activation matrix (class probabilities)
    """
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Predicted class labels
        Y: True labels
    """
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    """
    Train network parameters using gradient descent.
    
    Args:
        X: Training input data
        Y: Training labels
        alpha: Learning rate
        iterations: Number of training iterations
    """
    W1, b1, W2, b2 = init_params()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 10 == 0 or i == iterations:
            print("Iteration number: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))

    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    """
    Generate predictions for input data using trained weights.
    
    Args:
        X: Input data matrix
        W1, b1, W2, b2: Trained network parameters
    """
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def main():
    """Load data, train the network, and evaluate on test set."""
    parser = argparse.ArgumentParser(description="Train a neural network on MNIST data")
    parser.add_argument("--dataset_path", required=True, help="Path to the training CSV file")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument("--iterations", type=int, default=750, help="Number of training iterations (default: 750)")
    
    args = parser.parse_args()
    
    data = pd.read_csv(args.dataset_path)
    X_train, Y_train, X_test, Y_test, m, n = data_loader(data)
    
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, args.learning_rate, args.iterations)
    
    test_predictions = make_predictions(X_test, W1, b1, W2, b2)
    print("\nTest accuracy (in %) is: ", 100 * get_accuracy(test_predictions, Y_test))


if __name__ == "__main__":
    main()
