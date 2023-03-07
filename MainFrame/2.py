import os
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import random

import utils.Tools as Tools
import DensityPeaks.Node as Node
import DensityPeaks.NodesManager as NodesManager


Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

NODE_TYPE = "2_dimensional_point"
DC = 5


if __name__ == "__main__":
    # coordinates = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [2, 2], [3, 2], [4, 2], [2, 5],
    #                [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 7], [7, 9], [8, 7], [8, 8], [8, 9]]
    x = [[3, 4], [30, 40]]
    # x = tf.convert_to_tensor(x)
    print(type(x))
    y = np.linalg.norm(x, axis=1, keepdims=True)
    print(type(y), y)
    y = np.linalg.norm(y, axis=0, keepdims=True)
    # y = tf.norm(x, ord=2)
    print(type(y), y)
    z = float(y)
    print(type(z), z)
    coordinates = []
    for i in range(20):
        x = random.randint(0, 50)
        y = random.randint(0, 50)
        # coordinates.append([x, y])
        coordinates.append(np.array([x, y]))
    for i in range(20):
        x = random.randint(50, 100)
        y = random.randint(50, 100)
        # coordinates.append([x, y])
        coordinates.append(np.array([x, y]))
    nodes = []
    for c in coordinates:
        c_np = np.array(c)
        print("::", type(c_np), c_np.shape)
        nodes.append(Node.Node(NODE_TYPE, c_np))
    nodes_manager = NodesManager.NodesManager(NODE_TYPE, nodes)
    nodes_manager.show_distance_matrix()
    nodes_manager.calculate_nodes_rho(DC)
    nodes_manager.calculate_nodes_delta()
    print("average_rho:", nodes_manager.calculate_average_rho())
    nodes_manager.find_cluster_center(2)
    nodes_manager.clustering()

    for node in nodes:
        print(round(node.get_rho() * node.get_delta(), 4), end=" ")
    print("\n----------")

    for c in range(len(nodes_manager.clusters)):
        print("cluster", c, ":", len(nodes_manager.clusters[c]), nodes_manager.cluster_centers[c].get_node_c(),
              nodes_manager.cluster_centers[c].get_gama())
        x_list = []
        y_list = []
        for cn in nodes_manager.clusters[c]:
            x_list.append(cn.get_node_c()[0])
            y_list.append(cn.get_node_c()[1])
        plt.plot(x_list, y_list, 'o')

    plt.show()
    nodes_manager.show_nodes_distribution()
    nodes_manager.show_nodes_rho_and_delta()
