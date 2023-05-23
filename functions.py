import os
import sys
import numpy as np
import pickle
from my_mnist import load_mnist

class activation_functions:
    def __init__(self):
        return
    
    def step_function(self, x: np.ndarray) -> np.ndarray :
        y = x > 0
        ret = y.astype(np.float64)
        return ret
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray :
        y = 1.0 / (1.0 + np.exp(-x))
        return y
    
    def ReLU(self, x: np.ndarray) -> np.ndarray:
        y = np.maximum(0, x)
        return y

    def softmax(self, x: np.ndarray) -> np.ndarray:
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


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size) :
        self.params = {}
        self.params['W1'] = 0.01 * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = 0.01 * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x) :
        func = activation_functions()

        W1 = self.params['W1']
        W2 = self.params['W2']
        b1 = self.params['b1']
        b2 = self.params['b2']

        a1 = b1 + np.dot(x, W1)
        z1 = func.sigmoid(a1)

        a2 = b2 + np.dot(z1, W2)

        y = func.softmax(a2)

        return y
    
    # x is the input data, t is the supervision data
    def loss(self, x, t) :
        func = loss_functions()
        y = self.predict(x)
        return func.cross_entrophy_error(y, t)
    
    def accuracy(self, x, t) :
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(len(x))

        return accuracy
    
    
    def numerical_gradient(self, x, t) :
        # use this lambda expression can pass the param of numerical_gradient to the loss function, which can not be achieved by 'def'
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_grad(loss_W, self.params['W1'])
        grads['W2'] = numerical_grad(loss_W, self.params['W2'])
        grads['b1'] = numerical_grad(loss_W, self.params['b1'])
        grads['b2'] = numerical_grad(loss_W, self.params['b2'])

        return grads


def numerical_diff(f, x: np.ndarray) -> np.float64:
    h = np.float64(1e-4)
    return (f(x + h) - f(x - h)) / (2 * h)

 #def numerical_grad(f, x:np.ndarray) -> np.ndarray:
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

# note that the numerical gradient given in the textbook can not directly use here, because now we need a 2d iteration
# and this method is super slow
def numerical_grad(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad