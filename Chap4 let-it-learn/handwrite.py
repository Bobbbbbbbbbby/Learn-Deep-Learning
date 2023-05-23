import os, sys
import numpy as np
sys.path.append("..")
from my_mnist import load_mnist
from functions import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)

train_loss_list = []

iters_num = 10000
train_size = len(x_train)
batch_size = 100
learning_rate = 0.01

network = TwoLayerNet(784, 50, 10)

for i in range(iters_num) :
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train(batch_mask)
    t_batch = t_train(batch_mask)

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'W2', 'b1', 'b2') :
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    print(loss, end='\t')
    train_loss_list.append(loss)