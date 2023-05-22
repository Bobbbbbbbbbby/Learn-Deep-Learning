import os
import sys
import numpy as np
import pickle
from my_mnist import load_mnist

class activation_functions:
    def __init__(self):
        return
    
    def step_function(self, x: np.float64) -> np.float64 :
        y = x > 0
        ret = y.astype(np.float64)
        return ret
    
    def sigmoid(self, x: np.float64) -> np.float64 :
        y = 1.0 / (1.0 + np.exp(-x))
        return y
    
    def ReLU(self, x: np.float64) -> np.float64:
        y = np.maximum(0, x)
        return y

    def softmax(self, x: np.ndarray) -> np.float64:
        a = np.exp(x - np.max(x))
        b = np.sum(np.exp(x - np.max(x)))
        y = a / b
        return y


# loss_function only support one-hot expression of the training data
class loss_functions:
    def mean_squared_error(self, y: np.ndarray, t:np.ndarray) -> np.float64:
        diff = y - t
        diff *= diff
        return np.sum(diff) / 2
    
    def cross_entrophy_error(self, y: np.ndarray, t:np.ndarray) -> np.float64:
        y += 1e-7
        y = np.log(y)
        sum: np.float64 = np.sum(t * y)
        return -sum
    
def numerical_diff(f, x: np.ndarray) -> np.float64:
    h = np.float64(1e-4)
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_grad(f, x:np.ndarray) -> np.ndarray:
    h = np.float64(1e-4)
    grad = np.zeros_like(x)

    for idx in range(len(x)):
        temp = x[idx]
        x[idx] = temp + h
        forward = f(x)
        x[idx] = temp - h
        backward = f(x)

        grad[idx] = (forward - backward) / (2 * h)
        x[idx] = temp
    return grad
