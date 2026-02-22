import numpy as np

def ReLU(Z):
    return np.maximum(Z, 0)

def d_ReLU(Z):
    return Z > 0

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    A = np.exp(Z_shifted) / np.sum(np.exp(Z_shifted), axis=0, keepdims=True)
    return A
