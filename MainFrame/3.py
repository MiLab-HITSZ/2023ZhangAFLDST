import os
import datetime
import time
import threading
import copy

import tensorflow as tf

import utils.Tools as Tools
import utils.ResultManager as ResultManager
import Datasets.MNIST as MNIST
import Datasets.FashionMNIST as FashionMNIST
import Datasets.CIFAR10 as CIFAR10
import Models.FC3 as FC3
import Models.CNN as CNN
import Models.VGG13 as VGG13
# import AsyncClientManager
# import AsyncServer


Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

DATA_TYPE = "CIFAR10"
MODEL_NAME = "VGG13"

RESULT_FILE_NAME = "ECFA_CIFAR10_VGG13_T"

EPOCHS = 5000

CLIENT_NUMBER = 10
USE_CLIENT_NUMBER = 10

USE_IID_CLIENTS = False

BATCH_SIZE = 128  # 50
CLIENT_RATIO = 0.1
E = 10

CLIENT_STALENESS_SETTING = [2, 128, 63, 40]  # lower, upper, mu, sigma

if MODEL_NAME == "FC3":
    global_network = FC3.FC3(DATA_TYPE)
elif MODEL_NAME == "CNN":
    global_network = CNN.CNN(DATA_TYPE)
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


if __name__ == "__main__":
    start_time = datetime.datetime.now()

    c_dataset, x, y, x_valid, y_valid, x_test, y_test = Tools.generate_data(DATA_TYPE)

    client_list = Tools.generate_clients(c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE, BATCH_SIZE,
                                         USE_IID_CLIENTS, x, y)

    client_staleness_list = Tools.generate_normal_distribution_list(CLIENT_STALENESS_SETTING[0],
                                                                    CLIENT_STALENESS_SETTING[1],
                                                                    CLIENT_STALENESS_SETTING[2],
                                                                    CLIENT_STALENESS_SETTING[3], USE_CLIENT_NUMBER)

    non_iid_client_index = []
    for i in range(USE_CLIENT_NUMBER):
        non_iid_client_index.append(i)
    print(non_iid_client_index)

    accuracy_lists = []
    loss_lists = []

    alpha_list = [1]
    s_setting_list = [("Constant", 0, 0)]
    protocol_list = ["FedAsync"]
    curve_name_list = ["FedAsync"]
    rho_list = [0]

    for i in range(len(alpha_list)):
        global_network.set_weights(copy.deepcopy(init_weights))
        for client in client_list:
            client.set_client_weights(copy.deepcopy(init_weights))

        non_iid_client_list = []
        for n_i_c_i in non_iid_client_index:
            non_iid_client_list.append(client_list[n_i_c_i])

        s_setting = s_setting_list[i]
        protocol = protocol_list[i]
        print(protocol, ":")

        for e in range(EPOCHS):
            client_weights_list = []
            for client in non_iid_client_list:
                client.set_client_weights(global_network.get_weights())
                client.model_train_one_epoch(BATCH_SIZE, E)
                client_weights_list.append(client.get_client_weights())
            sum_weights = Tools.sum_nd_array_lists(client_weights_list)
            merged_weights = Tools.avg_nd_array_list(sum_weights, len(client_weights_list))
            global_network.set_weights(merged_weights)
            accuracy, loss = global_network.evaluate_network(x_test, y_test)
            accuracy_lists.append(accuracy)
            loss_lists.append(loss)
            print("Epoch", e, ": accuracy =", accuracy, "loss =", loss)

        del non_iid_client_list

    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    ResultManager.handle_result(RESULT_FILE_NAME, EPOCHS, len(alpha_list), curve_name_list, accuracy_lists, loss_lists)