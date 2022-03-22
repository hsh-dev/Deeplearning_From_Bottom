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

