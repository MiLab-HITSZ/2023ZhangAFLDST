import os
import datetime
import copy

import tensorflow as tf

import utils.Tools as Tools
import utils.ResultManager as ResultManager
import Models.ModelGenerator as ModelGenerator

Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

# CUDA_VISIBLE_DEVICES=

DATA_TYPE = "MNIST"
MODEL_NAME = "FC3"
# DATA_TYPE = "FashionMNIST"
# MODEL_NAME = "FC3"
# DATA_TYPE = "CIFAR10"
# MODEL_NAME = "CNN"
# DATA_TYPE = "CIFAR10"
# MODEL_NAME = "VGG13"

EPOCHS = 10

CLIENT_NUMBER = 10
CLIENT_SIZE = int(50000 / CLIENT_NUMBER)

BATCH_SIZE = 50
CLIENT_RATIO = 0.1
LOCAL_EPOCH = 1

USE_IID_CLIENTS = 0

SI = 0.3
SP = 1

USE_GPU_ID = [1]

# RESULT_FILE_NAME = "CE_" + str(DATA_TYPE) + "_SI" + str(SI) + "_mix"
RESULT_FILE_NAME = "CE_" + str(DATA_TYPE) + "_all-layer_SI1_SP-1-08-05-03"


def show_differences(weights_differences, layer_number, curves_name_list, result_file_num):
    difference_lists1 = [[] for _ in range(layer_number)]
    difference_lists2 = [[] for _ in range(layer_number)]

    for c_e in range(len(weights_differences[0])):
        for l0 in range(layer_number):
            difference_lists1[l0].append(weights_differences[0][c_e][l0])
        print("0 -", c_e, ":", weights_differences[0][c_e])

    for c_e in range(len(weights_differences[1])):
        for l1 in range(layer_number):
            difference_lists2[l1].append(weights_differences[1][c_e][l1])
        print("1 -", c_e, ":", weights_differences[1][c_e])

    ResultManager.handle_result(result_file_num, len(weights_differences[0]), len(curves_name_list), curves_name_list,
                                difference_lists1, difference_lists2)


def federated_learning_train(global_model, clients_list, epoch, select_list, select_interval, select_percentage):
    accuracy_list = []
    loss_list = []
    clients_weights_difference = \
        [[[] for _ in range(epoch)] for _ in range(len(clients_list))]  # 每个客户端在每个 Epoch 中各个层的参数变化程度
    all_list = [True] * len(select_list)
    clients_select_lists = [[] for _ in range(len(clients_list))]  # 每个 client 的较大参数选择情况
    for e in range(epoch):
        global_weights = copy.deepcopy(global_model.get_weights())
        clients_weights_list = []
        for c in range(len(clients_list)):
            print(c, end=" ")
            # 根据该 client 上一轮的较大参数选择情况设置其本轮的初始参数
            if e == 0 or select_percentage < 0:  # clients_select_lists 尚未初始化
                clients_list[c].set_client_weights(copy.deepcopy(global_weights))
            else:
                client_select_np_array_list = clients_select_lists[c]
                print("?? client_select_np_array_list:", client_select_np_array_list[0].shape, client_select_np_array_list[0])
                client_old_weights = copy.deepcopy(clients_list[c].get_client_weights())
                partially_updated_client_weights = \
                    Tools.partially_update_client_weights(client_old_weights, global_weights,
                                                          client_select_np_array_list)
                clients_list[c].set_client_weights(partially_updated_client_weights)

            # 训练 client 并获取训练后的参数
            clients_list[c].client_train_one_epoch(BATCH_SIZE, LOCAL_EPOCH)
            client_weights = copy.deepcopy(clients_list[c].get_client_weights())

            # 根据要求进行选择性传输参数
            if select_percentage == 1:
                select_percentage_list = [1, 1, 0.6, 0.6, 0.2, 0.2]
            elif select_percentage == 0.8:
                select_percentage_list = [0.8, 0.8, 0.5, 0.5, 0.2, 0.2]
            elif select_percentage == 0.5:
                select_percentage_list = [0.5, 0.5, 0.3, 0.3, 0.1, 0.1]
            elif select_percentage == 0.3:
                select_percentage_list = [0.3, 0.3, 0.2, 0.2, 0.1, 0.1]
            else:
                print("Error select_percentage!!")
                select_percentage_list = []
            # part_client_weights, select_np_array_list = \
            #     Tools.largest_part_transmission_of_weights_list(client_weights, global_weights, abs(select_percentage))
            part_client_weights, select_np_array_list = \
                Tools.largest_part_transmission_of_weights_list(client_weights, global_weights, select_percentage_list)
            if select_percentage < 0:
                part_client_weights = client_weights
            if e % select_interval == 0:
                clients_weights_list.append(Tools.selective_transmission(part_client_weights, global_weights,
                                                                         all_list))
            else:
                clients_weights_list.append(Tools.selective_transmission(part_client_weights, global_weights,
                                                                         select_list))

            # 记录每个 client 的较大参数选择情况
            clients_select_lists[c] = select_np_array_list

            # 记录参数变化程度
            weights_difference = []
            for i in range(len(client_weights)):
                # 计算每层参数与全局模型参数的差异并求 l2 范数
                weights_difference.append(Tools.l2_regularization_of_array(global_weights[i] - client_weights[i]))
            clients_weights_difference[c][e] = copy.deepcopy(weights_difference)
        print()

        # 平均各个 client 的参数并把全局模型换为平均模型
        sum_clients_weights_list = Tools.sum_nd_array_lists(clients_weights_list)
        averaged_clients_weights_list = Tools.avg_nd_array_list(sum_clients_weights_list, len(clients_list))
        global_model.set_weights(copy.deepcopy(averaged_clients_weights_list))

        # 记录 accuracy 与 loss
        accuracy, loss = global_model.evaluate_network(x_test, y_test)
        accuracy_list.append(accuracy)
        loss_list.append(loss)
        print("Train Epoch", e, ":  accuracy =", accuracy, ", loss =", loss,
              "|", select_list, "|", select_interval, "|", select_percentage)
    return accuracy_list, loss_list, clients_weights_difference


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # 生成全局模型
    modelGenerator = ModelGenerator.ModelGenerator(MODEL_NAME, DATA_TYPE)  # 模型生成器
    global_network = modelGenerator.generate_model()  # 全局模型
    init_weights = global_network.get_init_weights()  # 全局模型的初始参数
    print("init_weights:", type(init_weights), len(init_weights))
    print("init_weights[0]:", type(init_weights[0]), init_weights[0].shape)

    # 生成 dataset 和 clients
    c_dataset, x, y, x_valid, y_valid, x_test, y_test = Tools.generate_data(DATA_TYPE)
    client_list = Tools.generate_clients(USE_GPU_ID, c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE,
                                         BATCH_SIZE, USE_IID_CLIENTS, x, y)

    accuracy_lists = []
    loss_lists = []

    # 全部层传输 + 全部参数传输（真）
    # layer_select_list = [True, True, True, True, True,
    #                      True, True, True, True, True,
    #                      True, True, True, True, True,
    #                      True, True, True, True, True,
    #                      True, True, True, True, True, True]
    layer_select_list = [True, True, True, True, True, True]
    accuracy_list1, loss_list1, clients_weights_difference1 = \
        federated_learning_train(global_network, client_list, EPOCHS, layer_select_list, SI, 1)
    accuracy_lists.append(accuracy_list1)
    loss_lists.append(loss_list1)

    global_network = modelGenerator.generate_model()
    client_list = Tools.generate_clients(USE_GPU_ID, c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE,
                                         BATCH_SIZE, USE_IID_CLIENTS, x, y)

    # 部分层传输 + 全部参数传输
    layer_select_list = [True, True, True, True, True, True]
    accuracy_list2, loss_list2, clients_weights_difference2 = \
        federated_learning_train(global_network, client_list, EPOCHS, layer_select_list, SI, 0.8)
    accuracy_lists.append(accuracy_list2)
    loss_lists.append(loss_list2)

    global_network = modelGenerator.generate_model()
    client_list = Tools.generate_clients(USE_GPU_ID, c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE,
                                         BATCH_SIZE, USE_IID_CLIENTS, x, y)

    # 全部层传输 + 50% 参数传输
    layer_select_list = [True, True, True, True, True, True]
    accuracy_list3, loss_list3, clients_weights_difference3 = \
        federated_learning_train(global_network, client_list, EPOCHS, layer_select_list, SI, 0.5)
    accuracy_lists.append(accuracy_list3)
    loss_lists.append(loss_list3)

    global_network = modelGenerator.generate_model()
    client_list = Tools.generate_clients(USE_GPU_ID, c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE,
                                         BATCH_SIZE, USE_IID_CLIENTS, x, y)

    # 部分层传输 + 50% 参数传输
    layer_select_list = [True, True, True, True, True, True]
    accuracy_list4, loss_list4, clients_weights_difference4 = \
        federated_learning_train(global_network, client_list, EPOCHS, layer_select_list, SI, 0.3)
    accuracy_lists.append(accuracy_list4)
    loss_lists.append(loss_list4)

    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")

    # curve_name_list = ["1", "2", "3", "4", "5",
    #                    "6", "7", "8", "9", "10",
    #                    "11", "12", "13", "14", "15",
    #                    "16", "17", "18", "19", "20",
    #                    "21", "22", "23", "24", "25", "26"]
    # show_differences(clients_weights_difference1, 26, curve_name_list, "VGG13_CIFAR10_difference")

    # curve_name_list = ["all_layer",
    #                    "layer-1-2-3", "layer-1-2", "layer-1"]
    curve_name_list = ["all-layer_SI1_SP-1-0.6-0.2", "all-layer_SI1_SP-0.8-0.5-0.2",
                       "all-layer_SI1_SP-0.5-0.3-0.1", "all-layer_SI1_SP-0.3-0.2-0.1"]
    ResultManager.handle_result(RESULT_FILE_NAME, EPOCHS, len(curve_name_list), curve_name_list,
                                accuracy_lists, loss_lists)
