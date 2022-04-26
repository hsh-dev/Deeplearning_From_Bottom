import sys, os
sys.path.append(os.pardir)

from cmath import nan
import numpy as np
from network import network as nn
from dataset.mnist import load_mnist

from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    network = nn(input_size = 784, layer_1 = 300, layer_2 = 150, layer_3 = 100, layer_4 = 50, output_size = 10)
    
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    
    
    # img = x_train[0].reshape(28,-1)
    # plt.imshow(img)
    # plt.show()
    
    # print("Train : ",t_train[0])
    
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size , 1)
    
    
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        
        grad = network.gradient(x_batch, t_batch)
        
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5'):
            network.params[key] -= learning_rate * grad[key]
        
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        
        if loss == nan:
            break
        
        ## Save
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("Train ACC : ", train_acc, " Test ACC : ", test_acc)
    
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
