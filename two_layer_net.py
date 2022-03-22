import os
import numpy as np
from common import *
#### SIMPLE NET IMPLEMENT ####

## input -> hidden -> output
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        
        a2 = np.dot(z1, W2) + b2
        y = soft_max(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        loss = cross_entropy_error(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy


    def gradient(self, f, x, t, param):
        h = 1e-4  # 0.0001
        grad = np.zeros_like(param)

        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = param[idx]
            param[idx] = tmp_val + h
            fxh1 = f(x, t)  # f(x+h)

            param[idx] = tmp_val - h
            fxh2 = f(x, t)  # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)

            param[idx] = tmp_val  
            it.iternext()

        return grad
    
    def numerical_gradient(self, x, t):
        # grads = {}
        # grads['W1'] = self.params['W1'] * 0.1
        # grads['W2'] = self.params['W2'] * 0.1
        # grads['b1'] = self.params['b1'] * 0.1
        # grads['b2'] = self.params['b2'] * 0.1

        # loss_W = lambda w: self.loss(x, t)
        
        grads = {}
        grads['W1'] = self.gradient(self.loss, x, t, self.params['W1'])
        grads['b1'] = self.gradient(self.loss, x, t, self.params['b1'])
        grads['W2'] = self.gradient(self.loss, x, t, self.params['W2'])
        grads['b2'] = self.gradient(self.loss, x, t, self.params['b2'])
        
        return grads
