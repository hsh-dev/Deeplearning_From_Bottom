from typing import OrderedDict
import numpy as np
from layer import Affine, Relu, SoftmaxWithLoss
from collections import OrderedDict

class network:
    def __init__(self, input_size, layer_1, layer_2, layer_3, layer_4, output_size, weight_init_std = 0.01):
        
        ## Parameters
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, layer_1) * np.sqrt(2/(input_size+layer_1))
        self.params['b1'] = np.zeros(layer_1)
        self.params['W2'] = np.random.randn(layer_1, layer_2) * np.sqrt(2/(layer_1+ layer_2))
        self.params['b2'] = np.zeros(layer_2)
        self.params['W3'] = np.random.randn(layer_2, layer_3) * np.sqrt(2/(layer_2+ layer_3))
        self.params['b3'] = np.zeros(layer_3)
        self.params['W4'] = np.random.randn(layer_3, layer_4) * np.sqrt(2/(layer_3 + layer_4))
        self.params['b4'] = np.zeros(layer_4)
        self.params['W5'] = np.random.randn(layer_4, output_size) * np.sqrt(2/(layer_4 + output_size))
        self.params['b5'] = np.zeros(output_size)
        
        ## Layers
        self.layers = OrderedDict()     # Dictionary with order
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['Relu4'] = Relu()
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])
        
        self.lastlayer = SoftmaxWithLoss()
        
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y,t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        
        ## if t is not Nx1 matrix
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        
        return accuracy
    
    def gradient(self, x, t):
        self.loss(x, t)
        
        dout = 1
        dout = self.lastlayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dW
        grads['b4'] = self.layers['Affine4'].db
        grads['W5'] = self.layers['Affine5'].dW
        grads['b5'] = self.layers['Affine5'].db
        

        return grads
