import numpy as np

def one_hot(Y):
    """
    Convert class labels to one-hot encoded matrix.
    
    Args:
        Y (np.ndarray): 1D array of class labels (integers from 0 to num_classes-1).
    Returns:
        np.ndarray: One-hot encoded matrix of shape (num_classes, num_samples)
                   where each column represents a sample and contains a 1 at the
                   index corresponding to its class label, with 0s elsewhere.
    """
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T

    return one_hot_Y
