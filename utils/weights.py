import numpy as np

def init_params():
    """
    Initialize the weights and biases for a two-layer neural network.

    This function creates random initial parameters for a neural network with:
    - Input layer: 784 neurons (28x28 image pixels)
    - Hidden layer: 10 neurons
    - Output layer: 10 neurons (for 10 digit classes)
    """
    W1 = np.random.rand(10, 784) * np.sqrt(1/784)
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10) * np.sqrt(1/10)
    b2 = np.random.rand(10, 1)

    return W1, b1, W2, b2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Update the neural network parameters using gradient descent.
    """
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2
