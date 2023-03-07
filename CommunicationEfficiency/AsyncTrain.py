import os
import datetime

import tensorflow as tf

import utils.Tools as Tools
import utils.ResultManager as ResultManager
import Models.ModelGenerator as ModelGenerator
import MainFrame.AsyncServer as AsyncServer

Tools.set_gpu_with_increasing_occupancy_mode()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)

# DATA_TYPE = "MNIST"
# MODEL_NAME = "FC3"
DATA_TYPE = "FashionMNIST"
MODEL_NAME = "FC3"
# DATA_TYPE = "CIFAR10"
# MODEL_NAME = "CNN"
# DATA_TYPE = "CIFAR10"
# MODEL_NAME = "VGG13"

EPOCHS = 40

CLIENT_NUMBER = 100
CLIENT_SIZE = int(50000 / CLIENT_NUMBER)

BATCH_SIZE = 50
CLIENT_RATIO = 0.1
LOCAL_EPOCH = 1

USE_IID_CLIENTS = 0

USE_GPU_ID = [0]

RESULT_FILE_NAME = "2022_12_01_2"

accuracy_lists = []
loss_lists = []

select_percentage_history_list = []
final_accuracy_list = []
return_count_list = []


def align_result():
    # 以最短的为基准
    min_communication_round = len(accuracy_lists[0])
    for acc_list in accuracy_lists:
        if len(acc_list) < min_communication_round:
            min_communication_round = len(acc_list)

    # 将所有结果按 min_communication_round 对齐
    for i in range(len(accuracy_lists)):
        aligned_accuracy_list = []
        aligned_loss_list = []
        for r in range(min_communication_round):
            aligned_accuracy_list.append(accuracy_lists[i][r])
            aligned_loss_list.append(loss_lists[i][r])
        accuracy_lists[i] = aligned_accuracy_list
        loss_lists[i] = aligned_loss_list

    return min_communication_round


def async_train(select_list, select_percentage_list, adjust, staleness_penalty, decay_rate):
    # 生成数据集
    c_dataset, x, y, x_valid, y_valid, x_test, y_test = Tools.generate_data(DATA_TYPE)

    # 生成客户端
    client_list = Tools.generate_clients(USE_GPU_ID, c_dataset, DATA_TYPE, MODEL_NAME, CLIENT_NUMBER, CLIENT_SIZE,
                                         BATCH_SIZE, USE_IID_CLIENTS, x, y)

    # 设定客户端是否进行自适应调节
    for c in range(len(client_list)):
        client_list[c].set_adjust(adjust)
        client_list[c].set_select(select_list, select_percentage_list, staleness_penalty, decay_rate)

    # 生成全局模型
    model_generator = ModelGenerator.ModelGenerator(MODEL_NAME, DATA_TYPE)  # 模型生成器
    global_network = model_generator.generate_model()  # 全局模型
    init_weights = global_network.get_init_weights()  # 全局模型的初始参数

    # 生成服务器
    server = AsyncServer.AsyncServer(global_network, CLIENT_RATIO, client_list, init_weights,
                                     [x_valid, y_valid], [x_test, y_test], BATCH_SIZE,
                                     select_list)

    # 进行实验，并记录结果
    server.asynchronous_train(EPOCHS)
    accuracy_list, loss_list = server.get_accuracy_and_loss_result()
    accuracy_lists.append(accuracy_list)
    loss_lists.append(loss_list)

    # 计算最终准确率
    sum_accuracy = 0.0
    for i in range(10):
        sum_accuracy += accuracy_list[-1 - i]
    final_accuracy_list.append(sum_accuracy / 10)

    # 展示每个客户端各进行了多少次训练
    client_sp_average_list = []
    client_return_count_list = []
    for c in range(len(client_list)):
        sp_list, sp_average = client_list[c].get_select_percentage_history()
        if sp_average != -1:
            client_sp_average_list.append(sp_average)
            client_return_count_list.append(client_list[c].get_return_count())
        print(client_list[c].get_return_count(), "(", sp_average, ")", end=" ")
        if (c + 1) % 10 == 0:
            print()

    select_percentage_history_list.append(sum(client_sp_average_list) / len(client_sp_average_list))
    return_count_list.append(sum(client_return_count_list) / len(client_return_count_list))


if __name__ == '__main__':
    start_time = datetime.datetime.now()

    # 进行对照实验
    s_list = [True, True, True, True, True, True]
    s_p_list = [1, 1, 1, 1, 1, 1]
    async_train(s_list, s_p_list, False, 0, 0)

    # s_list = [True, True, True, True, True, True]
    # s_p_list = [1, 1, 1, 1, 1, 1]
    # async_train(s_list, s_p_list, True, 0, 0)

    s_list = [True, True, True, True, True, True]
    s_p_list = [1, 1, 1, 1, 1, 1]
    async_train(s_list, s_p_list, True, 0.125, 0.125)

    s_list = [True, True, True, True, True, True]
    s_p_list = [1, 1, 1, 1, 1, 1]
    async_train(s_list, s_p_list, True, 0.25, 0.25)

    s_list = [True, True, True, True, True, True]
    s_p_list = [1, 1, 1, 1, 1, 1]
    async_train(s_list, s_p_list, True, 0.5, 0.5)

    s_list = [True, True, True, True, True, True]
    s_p_list = [1, 1, 1, 1, 1, 1]
    async_train(s_list, s_p_list, True, 1, 1)

    s_list = [True, True, True, True, True, True]
    s_p_list = [1, 1, 1, 1, 1, 1]
    async_train(s_list, s_p_list, True, 2, 2)

    s_list = [True, True, True, True, True, True]
    s_p_list = [1, 1, 1, 1, 1, 1]
    async_train(s_list, s_p_list, True, 4, 4)

    s_list = [True, True, True, True, True, True]
    s_p_list = [1, 1, 1, 1, 1, 1]
    async_train(s_list, s_p_list, True, 8, 8)

    s_list = [True, True, True, True, True, True]
    s_p_list = [1, 1, 1, 1, 1, 1]
    async_train(s_list, s_p_list, True, 16, 16)

    s_list = [True, True, True, True, True, True]
    s_p_list = [1, 1, 1, 1, 1, 1]
    async_train(s_list, s_p_list, True, 32, 32)

    # s_list = [True, True, True, True, True, True]
    # s_p_list = [1, 1, 1, 1, 1, 1]
    # async_train(s_list, s_p_list, True, 0.125, 0)
    #
    # s_list = [True, True, True, True, True, True]
    # s_p_list = [1, 1, 1, 1, 1, 1]
    # async_train(s_list, s_p_list, True, 0.25, 0)
    #
    # s_list = [True, True, True, True, True, True]
    # s_p_list = [1, 1, 1, 1, 1, 1]
    # async_train(s_list, s_p_list, True, 0.5, 0)
    #
    # s_list = [True, True, True, True, True, True]
    # s_p_list = [1, 1, 1, 1, 1, 1]
    # async_train(s_list, s_p_list, True, 1, 0)
    #
    # s_list = [True, True, True, True, True, True]
    # s_p_list = [1, 1, 1, 1, 1, 1]
    # async_train(s_list, s_p_list, True, 2, 0)
    #
    # s_list = [True, True, True, True, True, True]
    # s_p_list = [1, 1, 1, 1, 1, 1]
    # async_train(s_list, s_p_list, True, 4, 0)
    #
    # s_list = [True, True, True, True, True, True]
    # s_p_list = [1, 1, 1, 1, 1, 1]
    # async_train(s_list, s_p_list, True, 8, 0)
    #
    # s_list = [True, True, True, True, True, True]
    # s_p_list = [1, 1, 1, 1, 1, 1]
    # async_train(s_list, s_p_list, True, 16, 0)
    #
    # s_list = [True, True, True, True, True, True]
    # s_p_list = [1, 1, 1, 1, 1, 1]
    # async_train(s_list, s_p_list, True, 32, 0)

    print("select_percentage_history_list: ", select_percentage_history_list)
    print("final_accuracy_list: ", final_accuracy_list)
    print("return_count_list: ", return_count_list)

    # 保存结果
    # curve_name_list = ["all layer transmission", "partial layer transmission"]
    # curve_name_list = ["$\lambda_k = 0$",
    #                    "$\lambda_k = 0.125$", "$\lambda_k = 0.25$", "$\lambda_k = 0.5$",
    #                    "$\lambda_k = 1$", "$\lambda_k = 2$", "$\lambda_k = 4$",
    #                    "$\lambda_k = 8$", "$\lambda_k = 16$", "$\lambda_k = 32$"]
    # curve_name_list = ["$\lambda_s = 0$",
    #                    "$\lambda_s = 0.125$", "$\lambda_s = 0.25$", "$\lambda_s = 0.5$",
    #                    "$\lambda_s = 1$", "$\lambda_s = 2$", "$\lambda_s = 4$",
    #                    "$\lambda_s = 8$", "$\lambda_s = 16$", "$\lambda_s = 32$"]
    curve_name_list = ["$\lambda_k = 0, \lambda_s = 0$",
                       "$\lambda_k = 0.125, \lambda_s = 0.125$", "$\lambda_k = 0.25, \lambda_s = 0.25$",
                       "$\lambda_k = 0.5, \lambda_s = 0.5$", "$\lambda_k = 1, \lambda_s = 1$",
                       "$\lambda_k = 2, \lambda_s = 2$", "$\lambda_k = 4, \lambda_s = 4$",
                       "$\lambda_k = 8, \lambda_s = 8$", "$\lambda_k = 16, \lambda_s = 16$",
                       "$\lambda_k = 32, \lambda_s = 32$"]
    min_c_r = align_result()
    ResultManager.handle_result(RESULT_FILE_NAME, min_c_r, len(curve_name_list), curve_name_list,
                                accuracy_lists, loss_lists)

    print("Time used:")
    end_time = datetime.datetime.now()
    print(((end_time - start_time).seconds / 60), "min")
    print(((end_time - start_time).seconds / 3600), "h")
