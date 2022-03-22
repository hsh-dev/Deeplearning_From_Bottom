import numpy as np
from two_layer_net import TwoLayerNet
from common import *
import matplotlib.pyplot as plt

network = TwoLayerNet(input_size=784, hidden_size=50,
                      output_size=10, weight_init_std=0.01)

train_loss_list = []

x_train, t_train = get_train_data()
iters_num = 10
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


for i in range(iters_num):
    ## MINI BATCHs
    print("###BATCH ", i, " RUN###")
    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기
    grad_ = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad_[key]

    loss = network.loss(x_batch, t_batch)
    print("###LOSS : " , loss , " ###")
    train_loss_list.append(loss)

plt.plot(train_loss_list)
plt.show()
