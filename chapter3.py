from concurrent.futures import thread
from copyreg import pickle
import sys, os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle
import warnings
import time
from multiprocessing import Process, Queue
from tqdm import trange

warnings.filterwarnings('ignore')

py_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = py_dir + "/" + "dataset"


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def identity_function(x):
    return x

def soft_max(a):
    max = np.max(a)
    exp_a = np.exp(a - max)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=False)
    return x_test, t_test

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def init_network():
    pic_dir = dataset_dir + "/sample_weight.pkl"
    with open(pic_dir,'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    # print("_____1st LAYER______")
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    # print("_____2nd LAYER______")
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    # print("_____3rd LAYER______")
    a3 = np.dot(z2, W3) + b3
    y = soft_max(a3)
    
    return y

def operation(number, que):
    print("PROCESS ID : ", os.getpid(), "____")
    
    start = number[0]
    end = number[1]
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in trange(start, end+1):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    que.put(accuracy_cnt)

def main():
    start_time = time.time()
    print("____MULTI PROCESSING BEGIN____")
    
    que = Queue()

    process_cnt = 2
    instruct = []
    for i in range(process_cnt):
        index_len = int(10000/process_cnt)
        instruct.append([i*index_len, (i+1)*index_len-1])

    procs = []
    
    for index, number in enumerate(instruct):
        proc = Process(target = operation, args=(number,que))
        procs.append(proc)
        proc.start()
    
    for proc in procs:
        proc.join()
    
    print("#####COMPLETE!!#####")
    print("##### %s seconds" % (time.time() - start_time))
    
    total = 0
    que.put('exit')
    while True:
        tmp = que.get()
        if tmp == 'exit':
            break
        else:
            total += tmp
    
    print("RESULT : ", str(float(total)/10000))
    

if __name__ == '__main__':
    main()
    
    
