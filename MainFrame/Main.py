import os
import datetime

import tensorflow as tf

from Client import Client
from ServerManage import ServerManager

import utils.Tools as Tools
import utils.ResultManager as ResultManager
import Datasets.MNIST as MNIST
import Datasets.FashionMNIST as FashionMNIST
import Datasets.CIFAR10 as CIFAR10

Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

# DATA_TYPE = "MNIST"
# MODEL_NAME = "FC3"
# DATA_TYPE = "FashionMNIST"
# MODEL_NAME = "FC3"
# DATA_TYPE = "CIFAR10"
# MODEL_NAME = "VGG13"
DATA_TYPE = "CIFAR10"
MODEL_NAME = "CNN"

RESULT_FILE_NAME = "m_t"
# RESULT_FILE_NAME = "fc3_fmnist_test"
# RESULT_FILE_NAME = "vgg13_cifar10_iid_1_1"
# RESULT_FILE_NAME = "cnn_mnist_n_iid_1_1"

EPOCHS = 2000
MERGE_CYCLE_LIST = [1]

CLIENT_NUMBER = 100
GROUP_NUMBER = 10

BEST_SERVER_WEIGHT = 1
USE_IID_CLIENTS = False

BATCH_SIZE = 64
CLIENT_RATIO = 0.1
E = 1

if DATA_TYPE == "MNIST" or "FashionMNIST" or "CIFAR10" or "CIFAR100":
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)
else:
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)

if __name__ == "__main__":
    start_time = datetime.datetime.now()

    accuracy_lists = []
    loss_lists = []

    c_dataset, x, y, x_valid, y_valid, x_test, y_test = Tools.generate_data(DATA_TYPE)

    client_list = Tools.generate_clients(c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE,
                                         BATCH_SIZE, USE_IID_CLIENTS, x, y)

    for grouping_cycle in MERGE_CYCLE_LIST:
        print("Start training server_manager, total epoch =", EPOCHS, "Model =", MODEL_NAME, "Dataset =", DATA_TYPE,
              "\nGrouping cycle =", grouping_cycle, "E =", E, "C =", CLIENT_RATIO, "file_name =", RESULT_FILE_NAME)

        server_manager = ServerManager(DATA_TYPE, MODEL_NAME, EPOCHS, BATCH_SIZE, E, grouping_cycle, GROUP_NUMBER,
                                       CLIENT_RATIO, BEST_SERVER_WEIGHT, client_list, [x_valid, y_valid],
                                       [x_test, y_test])
        server_manager.train_servers()
        accuracy_list, loss_list = server_manager.get_history()
        accuracy_lists.append(accuracy_list)
        loss_lists.append(loss_list)

        del server_manager

    print("Start training server_manager, total epoch =", EPOCHS, "Model =", MODEL_NAME, "Dataset =", DATA_TYPE,
          "\nE =", E, "C =", CLIENT_RATIO, "file_name =", RESULT_FILE_NAME)

    end_time = datetime.datetime.now()
    print("Time used:")
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    # 处理结果
    ResultManager.handle_result(RESULT_FILE_NAME, EPOCHS, len(MERGE_CYCLE_LIST), MERGE_CYCLE_LIST, accuracy_lists,
                                loss_lists)
