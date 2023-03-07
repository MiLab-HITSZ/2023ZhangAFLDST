import math
import os
import datetime
import time
import threading
from multiprocessing import Queue
import copy
import random
import numpy
import scipy.stats as stats
import matplotlib.pyplot as plt

import tensorflow as tf

import AsyncClient
import utils.Tools as Tools
import utils.ResultManager as ResultManager
from Server import Server
from Server import Server
import Datasets.MNIST as MNIST
import Datasets.FashionMNIST as FashionMNIST
import Datasets.CIFAR10 as CIFAR10
import Models.FC3 as FC3
import Models.VGG13 as VGG13
import DensityPeaks.ClusterWithDensityPeaks as ClusterWithDensityPeaks

Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

DATA_TYPE = "MNIST"
MODEL_NAME = "FC3"

RESULT_FILE_NAME = "fc3_mnist_test"

EPOCHS = 20

CLIENT_NUMBER = 500

USE_IID_CLIENTS = True

BATCH_SIZE = 50
CLIENT_RATIO = 0.1
E = 1

if MODEL_NAME == "FC3":
    global_network = FC3.FC3(DATA_TYPE)
elif MODEL_NAME == "VGG13":
    global_network = VGG13.VGG13(DATA_TYPE)
else:
    print("Unexpected dataset name!")
    global_network = FC3.FC3(DATA_TYPE)

init_weights = copy.deepcopy(global_network.get_init_weights())

if DATA_TYPE == "MNIST" or "FashionMNIST" or "CIFAR10" or "CIFAR100":
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)
else:
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)

if __name__ == '__main__':
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    b = tf.constant([[3, 5], [7, 9]], dtype=tf.float32)
    c = abs(a - b)
    print(a, b, c, c * c)
    print("------------------------------")
    cq = tf.math.square(a - b)
    print(cq)
    diff = tf.math.reduce_sum(cq)
    print(diff)

    dd = {1: '张三', 2: '李四', 3: '王五', 4: '赵六', 5: '王麻子', 6: '包子', 7: '豆浆'}
    r = random.sample(dd.keys(), 5)
    print(r)
    print(44 in dd)

    for i in range(100):
        random3 = numpy.random.randn(1)
        print(random3)

    lower, upper = 2, 128
    mu, sigma = 63, 40
    # X表示含有最大最小值约束的正态分布
    # N表示不含最大最小值约束的正态分布
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数
    N = stats.norm(loc=mu, scale=sigma)  # 无区间限制的随机数
    a = X.rvs(100)  # 取其中的5个数，赋值给a；a为array类型
    print(a)

    fig, ax = plt.subplots(2)
    ax[0].hist(X.rvs(10000), density=True)
    ax[1].hist(N.rvs(10000), density=True)
    plt.show()

    clustering_result = [1, 2, 3, 4, 5]
    print(clustering_result)
    l = [0]
    l.extend(clustering_result)
    print(l)

    ll = [[1, 2, 3], [4, 5, 6]]
    lla = numpy.array(ll)
    wla = Tools.weight_nd_array_list([lla, lla], 0.2213)
    print(wla)

    dd = {1: 20, 2: 30, 3: 40, 4: 50}
    print(sum(dd.values()))

    vv = [66, 56, 18, 10, 13, 25, 8, 36, 0, 9, 23]
    vvd = {1: 66, 2: 56, 3: 18, 4: 10, 5: 13, 6: 25, 7: 8, 8: 36, 9: 0, 10: 9, 11: 23}
    vdl = list(vvd.values())
    # vv = [1, 4, 2, 6, 3, 0, 6, 2, 6, 9]
    print("var =", numpy.var(vv))
    print("var =", numpy.var(vdl))
    for i in range(50):
        for v in range(len(vv)):
            if vv[v] > 5:
                vv[v] -= 1
        print("var =", numpy.var(vv), vv)

    start_time = datetime.datetime.now()

    c_dataset, x, y, x_valid, y_valid, x_test, y_test = Tools.generate_data(DATA_TYPE)

    client_list = Tools.generate_clients(c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE, BATCH_SIZE,
                                         USE_IID_CLIENTS, x, y)

    client = client_list[0]

    accuracy_lists = []
    loss_lists = []

    accuracy_list = []
    loss_list = []
    for e in range(EPOCHS):
        client.model_train_one_epoch(BATCH_SIZE, E, use_weight_regularization=False, rho=0)
        accuracy, loss = client.network.evaluate_network(x_test, y_test)
        accuracy_list.append(accuracy)
        loss_list.append(loss)
    accuracy_lists.append(accuracy_list)
    loss_lists.append(loss_list)

    print("---------------------------------------------------------------------------------------------------------")
    client.network.set_weights(init_weights)

    accuracy_list = []
    loss_list = []
    for e in range(EPOCHS):
        client.model_train_one_epoch(BATCH_SIZE, E, use_weight_regularization=True, rho=10)
        accuracy, loss = client.network.evaluate_network(x_test, y_test)
        accuracy_list.append(accuracy)
        loss_list.append(loss)
    accuracy_lists.append(accuracy_list)
    loss_lists.append(loss_list)

    curve_name_list = ["MNIST1", "MNIST2"]

    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")
    ResultManager.handle_result("test", EPOCHS, 2, curve_name_list, accuracy_lists, loss_lists)
