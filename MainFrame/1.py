import os
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import random

import utils.ResultManager as ResultManager
from Server import Server
import Datasets.MNIST as MNIST
import Datasets.FashionMNIST as FashionMNIST
import Datasets.CIFAR10 as CIFAR10
import Models.FC3 as FC3
import Models.VGG13 as VGG13
import utils.Tools as Tools
import DensityPeaks.Node as Node
import DensityPeaks.NodesManager as NodesManager


Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

DATA_TYPE = "MNIST"
MODEL_NAME = "FC3"
RESULT_FILE_NAME = "fc3_mnist_cc3_test"

NODE_TYPE = "multidimensional_point"
# NODE_TYPE = "3_dimensional_point"

# DC_MIN, DC_MAX, DC_STEP = 0.1709, 0.1714, 0.00005  # 0.166, 0.178, 0.002
# T_MIN, T_MAX, T_STEP = 3.6, 5.4, 0.2
DC_MIN, DC_MAX, DC_STEP = 0.0516, 1, 10
T_MIN, T_MAX, T_STEP = 4, 5, 10

USE_WEIGHT_LAYER = 4

CLIENT_NUMBER = 1000
USE_CLIENT_NUMBER = 1000
CLIENT_SIZE = int(50000 / CLIENT_NUMBER)

USE_IID_CLIENTS = False

BATCH_SIZE = 50
E = 1
pre_E = 5

global_network = FC3.FC3(DATA_TYPE)
init_weights = global_network.get_init_weights()

if __name__ == "__main__":
    c_dataset, x, y, x_valid, y_valid, x_test, y_test = Tools.generate_data(DATA_TYPE)
    client_list = Tools.generate_clients(c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE,
                                         BATCH_SIZE, USE_IID_CLIENTS, x, y)

    non_iid_client_index = []
    for i in range(USE_CLIENT_NUMBER):
        non_iid_client_index.append(i)

    print(non_iid_client_index)
    # random.shuffle(non_iid_client_index)
    print(non_iid_client_index)
    non_iid_client_list = []
    for n_i_c_i in non_iid_client_index:
        non_iid_client_list.append(client_list[n_i_c_i])
    print(non_iid_client_list[0].get_client_weights()[0].shape)
    print(non_iid_client_list[0].get_client_weights()[1].shape)
    print(non_iid_client_list[0].get_client_weights()[2].shape)
    print(non_iid_client_list[0].get_client_weights()[3].shape)
    print(non_iid_client_list[0].get_client_weights()[4].shape)
    print(non_iid_client_list[0].get_client_weights()[5].shape)

    # 生成server
    server1 = Server(DATA_TYPE, MODEL_NAME, 1, non_iid_client_list, init_weights,
                     [x_valid, y_valid], [x_test, y_test], BATCH_SIZE, E)

    # 预先正常进行一个Epoch
    print("Pre-train:")
    server1.set_e(pre_E)
    p_accuracy, p_loss = server1.train_clients_one_epoch(0)
    server1.set_e(E)
    print("---------------", x_valid.shape)

    weight_list = []
    coordinates = []
    nodes = []
    for client in non_iid_client_list:
        weight_list.append(client.get_client_weights())
        coordinates.append(client.get_client_weights()[0])

        # coordinate = []
        # a = np.linalg.norm(np.linalg.norm(client.get_client_weights()[0], axis=1, keepdims=True), axis=0, keepdims=True)
        # b = np.linalg.norm(np.linalg.norm(client.get_client_weights()[2], axis=1, keepdims=True), axis=0, keepdims=True)
        # c = np.linalg.norm(np.linalg.norm(client.get_client_weights()[4], axis=1, keepdims=True), axis=0, keepdims=True)

        nodes.append(Node.Node(NODE_TYPE, client.get_client_weights()[USE_WEIGHT_LAYER]))
        # nodes.append(Node.Node(NODE_TYPE, np.array([a, b, c]).reshape([3])))

    nodes_manager = NodesManager.NodesManager(NODE_TYPE, nodes)
    distance_matrix = nodes_manager.get_distance_matrix()
    print("------------------------------------------------------------------------------------")

    result_list = []

    DC = DC_MIN
    while DC <= DC_MAX:
        T = T_MIN
        while T <= T_MAX:
            print("DC =", DC, " | T =", T)

            nodes_manager.set_distance_matrix(distance_matrix)
            # nodes_manager.show_distance_matrix()
            nodes_manager.calculate_nodes_rho(DC)
            nodes_manager.calculate_nodes_delta()
            print("average_rho:", nodes_manager.calculate_average_rho())
            nodes_manager.find_cluster_center(T)
            nodes_manager.clustering()

            print("\n----------")

            result = [DC, T]
            for c in range(len(nodes_manager.clusters)):
                print("cluster", c, ":", len(nodes_manager.clusters[c]), nodes_manager.cluster_centers[c].get_gama())
                result.append(len(nodes_manager.clusters[c]))

            nodes_manager.show_nodes_rho_and_delta()
            result_list.append(result)

            nodes_manager.reset()
            nodes_manager.reset_nodes()

            T += T_STEP
        DC += DC_STEP

    print("----------------------------------------------------------------------------------------------------------")
    for r in result_list:
        print(r)
