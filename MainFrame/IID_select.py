import os
import datetime
import random

import tensorflow as tf

from Client import Client
from Server import Server

import utils.Tools as Tools
import utils.ResultManager as ResultManager
import Datasets.MNIST as MNIST
import Datasets.FashionMNIST as FashionMNIST
import Datasets.CIFAR10 as CIFAR10
import Models.FC3 as FC3
import Models.VGG13 as VGG13

Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

# DATA_TYPE = "MNIST"
# MODEL_NAME = "FC3"
# DATA_TYPE = "FashionMNIST"
# MODEL_NAME = "FC3"
DATA_TYPE = "CIFAR10"
MODEL_NAME = "VGG13"
# DATA_TYPE = "CIFAR100"
# MODEL_NAME = "VGG13"

# RESULT_FILE_NAME = "fc3_mnist_cc3_test"
# RESULT_FILE_NAME = "fc3_fmnist_test"
RESULT_FILE_NAME = "vgg13_cifar10_n_iid_t"
# RESULT_FILE_NAME = "vgg13_cifar100_test"

EPOCHS = 5000

CLIENT_NUMBER = 100
CLIENT_RATIO = 1

USE_IID_CLIENTS = False
USE_ALL_CLIENTS = False

BATCH_SIZE = 128
E = 1
pre_E = 5

if MODEL_NAME == "FC3":
    global_network = FC3.FC3(DATA_TYPE)
elif MODEL_NAME == "VGG13":
    global_network = VGG13.VGG13(DATA_TYPE)
else:
    print("Unexpected dataset name!")
    global_network = FC3.FC3(DATA_TYPE)
init_weights = global_network.get_init_weights()

if DATA_TYPE == "MNIST" or "FashionMNIST" or "CIFAR10" or "CIFAR100":
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)
else:
    print("Unexpected dataset name!")
    CLIENT_SIZE = int(50000 / CLIENT_NUMBER)

if __name__ == "__main__":
    start_time = datetime.datetime.now()

    accuracy_lists = []
    loss_lists = []

    print("Epoch =", EPOCHS, "client_num =", CLIENT_NUMBER, "batch_size =", BATCH_SIZE,
          "\npre_E =", pre_E, "E =", E, "C =", CLIENT_RATIO, "non_iid_client_size =", CLIENT_SIZE,
          "\nfile_name =", RESULT_FILE_NAME)

    # 生成数据集与非IID的clients
    c_dataset, x, y, x_valid, y_valid, x_test, y_test = Tools.generate_data(DATA_TYPE)
    client_list = Tools.generate_clients(c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE,
                                         BATCH_SIZE, USE_IID_CLIENTS, x, y)
    # non_iid_client_index = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9]
    non_iid_client_index = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90,
                            1, 11, 21, 31, 41,
                            2, 12, 22, 32,  # 42,
                            3, 13, 23, 33,  # 43,
                            4, 14, 24,  # 34, 44,
                            5, 15, 25,  # 35, 45,
                            6, 16,  # 26, 36, 46,
                            7, 17,  # 27, 37, 47,
                            8,  # 18, 28, 38, 48,
                            9,  # 19, 29, 39, 49
                            ]
    # non_iid_client_index = []
    # for i in range(100):
    #     non_iid_client_index.append(int(i * 2))
    # for i in range(50):
    #     non_iid_client_index.append(int(i * 2 + 1))

    print(non_iid_client_index)
    random.shuffle(non_iid_client_index)
    print(non_iid_client_index)
    non_iid_client_list = []
    for n_i_c_i in non_iid_client_index:
        non_iid_client_list.append(client_list[n_i_c_i])

    # 生成server
    server1 = Server(DATA_TYPE, MODEL_NAME, 1, non_iid_client_list, init_weights,
                     [x_valid, y_valid], [x_test, y_test], BATCH_SIZE, E)
    server2 = Server(DATA_TYPE, MODEL_NAME, 1, non_iid_client_list, init_weights,
                     [x_valid, y_valid], [x_test, y_test], BATCH_SIZE, E)
    server3 = Server(DATA_TYPE, MODEL_NAME, 1, non_iid_client_list, init_weights,
                     [x_valid, y_valid], [x_test, y_test], BATCH_SIZE, E)

    for i in range(3):
        accuracy_lists.append([])
        loss_lists.append([])

    # 预先正常进行一个Epoch
    print("Pre-train:")
    server1.set_e(pre_E)
    p_accuracy, p_loss = server1.train_clients_one_epoch(0)
    server1.set_e(E)
    accuracy_lists[0].append(p_accuracy)
    loss_lists[0].append(p_loss)
    server2.set_e(pre_E)
    p_accuracy, p_loss = server2.train_clients_one_epoch(0)
    server2.set_e(E)
    accuracy_lists[1].append(p_accuracy)
    loss_lists[1].append(p_loss)
    print("---------------", x_valid.shape)
    x_pre_valid, y_pre_valid = c_dataset.get_pre_valid_data()
    iid_clients_list, iid_clients_index_list = Tools.select_iid_clients(global_network, non_iid_client_list,
                                                                        non_iid_client_index, x_pre_valid, y_pre_valid)
    for clients in iid_clients_index_list:
        print(clients, end=" ")
    print("\n", len(iid_clients_list))
    for clients in iid_clients_list:
        for client in clients:
            print(client.get_data()[1][0], end=" ")
            # FashionMNIST.draw_pictures([client.get_data()[0][0], client.get_data()[0][487]])
        print("")

    # 按照均匀且选择代表的clients进行训练
    for e in range(EPOCHS):
        print("\nEpoch:", e + 1)
        if USE_ALL_CLIENTS:
            use_clients_list = Tools.random_select_from_lists_list(iid_clients_list, CLIENT_RATIO)
        else:
            random_selected_iid_clients_list = Tools.random_select_from_lists_list(iid_clients_list, CLIENT_RATIO)
            random_iid_client_list = []
            for rs_iid_clients in random_selected_iid_clients_list:
                if len(rs_iid_clients) > 0:
                    random_index = random.randint(0, len(rs_iid_clients) - 1)
                    print(random_index, len(rs_iid_clients), end=" | ")
                    random_iid_client_list.append([rs_iid_clients[random_index]])
                else:
                    print("non", end=" | ")
                    random_iid_client_list.append([])
            use_clients_list = random_iid_client_list
            print()
            for random_iid_client in random_iid_client_list:
                if len(random_iid_client) > 0:
                    print(random_iid_client[0].get_data()[1][0], end=" , ")
                else:
                    print("non", end=" , ")
            print()

        server1.train_clients_one_epoch_with_iid_selection(0, use_clients_list, accuracy_lists, loss_lists)

    # 按照均匀的clients进行训练
    for e in range(EPOCHS):
        print("Epoch:", e)
        use_clients_list = Tools.random_select_from_lists_list(iid_clients_list, CLIENT_RATIO)
        server2.train_clients_one_epoch_with_iid_selection(1, use_clients_list, accuracy_lists, loss_lists)

    # 对照实验
    server3.set_client_ratio(CLIENT_RATIO)
    for e in range(EPOCHS + 1):
        print("Epoch:", e)
        accuracy, loss = server3.train_clients_one_epoch(1)
        accuracy_lists[2].append(accuracy)
        loss_lists[2].append(loss)

    print("Epoch =", EPOCHS, "client_num =", CLIENT_NUMBER, "batch_size =", BATCH_SIZE,
          "\npre_E =", pre_E, "E =", E, "C =", CLIENT_RATIO, "non_iid_client_size =", CLIENT_SIZE,
          "\nfile_name =", RESULT_FILE_NAME)
    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    name_list = ["client classification 1", "client classification all", "non_iid"]
    ResultManager.handle_result(RESULT_FILE_NAME, EPOCHS + 1, 3, name_list, accuracy_lists, loss_lists)
