import os
import sys
import numpy as np
import pickle
from my_mnist import load_mnist


def step_function(x):
    y = x > 0
    return y.astype(np.int32)


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def ReLU(x):
    return np.maximum(0, x)


def softmax(x):
    a = np.exp(x)
    b = np.sum(np.exp(x))
    y = a / b
    return y


def mod_softmax(x):
    a = np.exp(x - np.max(x))
    b = np.sum(np.exp(x - np.max(x)))
    y = a / b
    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist()
    return x_test, t_test


def init_network():
    with open(os.path.dirname(os.path.abspath(__file__)) + "/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a = np.dot(x, W1) + b1
    a = sigmoid(a)
    a = np.dot(a, W2) + b2
    a = sigmoid(a)
    a = np.dot(a, W3) + b3
    y = softmax(a)

    return y
