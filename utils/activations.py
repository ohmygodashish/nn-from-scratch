import numpy as np

def ReLU(Z):
    return np.maximum(Z, 0)

def d_ReLU(Z):
    return Z > 0

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
