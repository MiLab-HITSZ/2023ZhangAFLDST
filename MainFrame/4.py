import math
import os
import datetime
import time
import threading
from multiprocessing import Queue
import copy
import random
import numpy

import tensorflow as tf

import FedAsync.AsyncClient as AsyncClient
# import AsyncClient
import utils.Tools as Tools
import utils.ResultManager as ResultManager
from Server import Server
from Server import Server
import Datasets.MNIST as MNIST
import Datasets.FashionMNIST as FashionMNIST
import Datasets.CIFAR10 as CIFAR10
import Models.FC3 as FC3
import Models.CNN as CNN
import Models.VGG13 as VGG13
import DensityPeaks.ClusterWithDensityPeaks as ClusterWithDensityPeaks

Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

# DATA_TYPE = "MNIST"
# MODEL_NAME = "FC3"
DATA_TYPE = "CIFAR10"
MODEL_NAME = "CNN"

CLIENT_NUMBER = 1000
USE_CLIENT_NUMBER = 1000
CLIENT_SIZE = int(50000 / CLIENT_NUMBER)

USE_IID_CLIENTS = False

BATCH_SIZE = 50
CLIENT_RATIO = 0.1
E = 5
SCHEDULER_INTERVAL = 30

AUTO = True
FULL_USE = 0
TRUNCATION = 5
DC_MIN, DC_MAX, DC_STEP = -0.995, -0.005, 0.005
T_MIN, T_MAX, T_STEP = 4.1, 4.1, 1

# global_network = FC3.FC3(DATA_TYPE)
global_network = CNN.CNN(DATA_TYPE)
init_weights = copy.deepcopy(global_network.get_init_weights())

queue = Queue()
stop_event = threading.Event()
stop_event.clear()

client_pre_weights_dict = {}


if __name__ == '__main__':
    al = [[2.456, 3.657, 4.13463547], [5.56867, 6.8123475, 7.9057805]]
    bl = [[1.1, 2.2, 3.01], [4.1, 5.4, 6.81]]
    a = numpy.array(al)
    b = numpy.array(bl)
    c = a - b
    print(type(c))
    print(c)

    xyz = numpy.array([1, 1, 1])
    td = xyz[0]
    for i in range(len(xyz)):
        if i > 0:
            td = math.hypot(td, xyz[i])
    distance = td
    print(distance)

    re = [0.20405000000000725, 3.5, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    rr = []
    print(re[2:])
    if re[2:] == [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]:
        rr.append(re)
    print(rr)

    c_dataset, x, y, x_valid, y_valid, x_test, y_test = Tools.generate_data(DATA_TYPE)

    client_list = Tools.generate_clients(c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE, BATCH_SIZE,
                                         USE_IID_CLIENTS, x, y)

    non_iid_client_index = []
    for i in range(USE_CLIENT_NUMBER):
        # non_iid_client_index.append(i)
        if i < FULL_USE:
            non_iid_client_index.append(i)
        else:
            if i % 100 < TRUNCATION:
                non_iid_client_index.append(i)
        if i % 100 == 0:
            TRUNCATION += 10
    print(non_iid_client_index)

    non_iid_client_list = []
    for n_i_c_i in non_iid_client_index:
        non_iid_client_list.append(client_list[n_i_c_i])

    # server1 = Server(DATA_TYPE, MODEL_NAME, 1, non_iid_client_list, init_weights,
    #                  [x_valid, y_valid], [x_test, y_test], BATCH_SIZE, E)
    #
    # # 预先正常进行一个Epoch
    # print("Pre-train:")
    # server1.set_e(E)
    # p_accuracy, p_loss = server1.train_clients_one_epoch(0)
    # server1.set_e(E)
    # print("---------------", x_valid.shape)

    # for c_id in range(len(non_iid_client_list)):
    #     client_pre_weights_dict[c_id] = non_iid_client_list[c_id].get_client_weights()

    server_thread_lock = threading.Lock()
    pre_train_queue = Queue()

    client_thread_list = []
    for i in range(len(non_iid_client_list)):
        client_delay = random.uniform(0, 0.001)
        client_thread_list.append(AsyncClient.AsyncClient(i, non_iid_client_list[i], queue, pre_train_queue, BATCH_SIZE,
                                                          E, 0, stop_event, client_delay, False))

    # 启动clients
    print("Start clients:")
    for client_thread in client_thread_list:
        client_thread.start()

    for s_client_thread in client_thread_list:
        # 将server的初始模型参数发给client
        s_client_thread.set_client_weight(init_weights)

        # 启动一次client线程
        s_client_thread.set_event()

    # 接收各个client发回的模型参数并存储
    for c in range(len(client_thread_list)):
        while True:
            # 接收一个client发回的模型参数和时间戳
            if not queue.empty():
                (c_id, client_weights, time_stamp) = queue.get()
                # print("Received data from client", c_id, time_stamp, "| queue size = ", self.queue.qsize())
                client_pre_weights_dict[c_id] = client_weights
                print(type(client_weights), len(client_weights))
                break
            else:
                time.sleep(0.01)

    cluster_with_density_peaks = ClusterWithDensityPeaks.ClusterWithDensityPeaks(client_pre_weights_dict, init_weights,
                                                                                 "multidimensional_point",
                                                                                 6, -0.5, 4.1)
    distance_matrix = cluster_with_density_peaks.get_distance_matrix()

    clusters = cluster_with_density_peaks.clustering()
    cluster_with_density_peaks.show_clusters()

    result_list = []
    correct_results = []

    if AUTO:
        DC = DC_MIN
        while DC <= DC_MAX:
            T = T_MIN
            while T <= T_MAX:
                print("DC =", DC, " | T =", T)
                cluster_with_density_peaks.set_distance_matrix(distance_matrix)
                cluster_with_density_peaks.set_dc(DC)
                cluster_with_density_peaks.set_t(T)

                clusters = cluster_with_density_peaks.clustering()
                cluster_with_density_peaks.show_clusters()

                result = [DC, T]
                for c in range(len(clusters)):
                    # print("cluster", c, ":", len(clusters[c]))
                    result.append(len(clusters[c]))
                result_list.append(result)

                cluster_with_density_peaks.reset()

                T += T_STEP
            DC += DC_STEP
    else:
        DC = 0
        end = 1

        while end >= 0:
            DC = float(input('请输入 DC：'))
            T = float(input('请输入 T：'))
            cluster_with_density_peaks.set_distance_matrix(distance_matrix)
            cluster_with_density_peaks.set_dc(DC)
            cluster_with_density_peaks.set_t(T)

            clusters = cluster_with_density_peaks.clustering()

            cluster_with_density_peaks.show_clusters()

            result = [DC]
            for c in range(len(clusters)):
                print("cluster", c, ":", len(clusters[c]))
                result.append(len(clusters[c]))
            result_list.append(result)

            cluster_with_density_peaks.reset()

            end = float(input('是否结束：'))

    print("----------------------------------------------------------------------------------------------------------")
    for r in result_list:
        print(r, end="\t | \t")
        c100 = 0
        c50 = 0
        c_else = 0
        for i in range(2, len(r)):
            if r[i] == 100:
                c100 += 1
            elif r[i] == TRUNCATION:
                c50 += 1
            else:
                c_else += 1
        print("100:", c100, " | 50:", c50, " | else:", c_else)
        if c100 == 0 and c50 == 10:
            correct_results.append(r)

    print("----------------------------------------------------------------------------------------------------------")
    for c_r in correct_results:
        print(c_r)
