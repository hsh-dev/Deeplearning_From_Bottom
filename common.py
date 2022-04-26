#### COMMON TOOLS ####

import os
import numpy as np
import warnings
from dataset.mnist import load_mnist

warnings.filterwarnings('ignore')

py_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = py_dir + "/" + "dataset"

def get_train_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=False)
    return x_train, t_train

def get_test_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=False)
    return x_test, t_test  

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # ### one_hot vector case
    # return -np.sum(t * np.log(y + 1e-7)) / batch_size
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    
def soft_max(a):
    max = np.max(a)
    exp_a = np.exp(a - max)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def identity_function(x):
    return x


class Momentum:
    def __init__(self, lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
            
class AdaGrad:
    def __init__(self, lr = 0.01):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            
