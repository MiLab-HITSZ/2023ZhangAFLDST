import ResultManager
import Tools
import numpy as np

ResultManager.draw_curves_from_txt('2023_01_02_2.txt')
# ResultManager.draw_curves_from_txt('_CIFAR10_200c_30000_SGD01_UIC7_BS2_DECAY0005_I20.txt')

# CUDA_VISIBLE_DEVICES=1

# for (s, x_max) in [(1, 20), (2, 12), (4, 18)]:
#     CLIENTS_WEIGHTS_PACKAGE_PATH = 'C:/Users/4444/PycharmProjects/Multi_server_federated_learning/clients_weights/'
#     Non_IID1_MNIST_list = np.load(CLIENTS_WEIGHTS_PACKAGE_PATH + "Non_IID" + str(s) + "_MNIST_list.npy",
#                                   allow_pickle=True).tolist()
#     Non_IID1_FashionMNIST_list = np.load(CLIENTS_WEIGHTS_PACKAGE_PATH + "Non_IID" + str(s) + "_FashionMNIST_list.npy",
#                                          allow_pickle=True).tolist()
#     Non_IID1_CIFAR10_list = np.load(CLIENTS_WEIGHTS_PACKAGE_PATH + "Non_IID" + str(s) + "_CIFAR10_list.npy",
#                                     allow_pickle=True).tolist()
#     bar_lists = [Non_IID1_CIFAR10_list, Non_IID1_FashionMNIST_list, Non_IID1_MNIST_list]
#     fig_name = "DPC_NON_IID" + str(s)
#     ResultManager.draw_gradual_bars(bar_lists, fig_name, x_max)

# q_s_list = [84, 56, 52, 82, 79, 51, 0, 39, 48, 22]C:\Users\4444\PycharmProjects\Multi_server_federated_learning
# q_s_list = [21, 14, 7, 2, 4, 8, 1, 4, 0, 9]
# select_list = Tools.balance_select(q_s_list, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100], 100, 4)
# print("select_list:", select_list)

# bar_lists = [
#     [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]],
#
#     [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]],
#
#     [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
# ]

# bar_lists = []
# for i in range(3):
#     bar_lists.append([])
#     for j in range(3):
#         bar_lists[i].append([])
#         for k in range(100):
#             bar_lists[i][j].append(k * 0.01)
# fig_name = "test"

# data_lists = [[[0.7, 8.4, 1.4], [0.4, 5.1, 1.1], [0.4, 2.9, 0.8]],
#               [[1.0, 13.2, 3.3], [0.7, 8.0, 3.1], [0.6, 4.7, 2.6]],
#               [[0.6, 11.3, 3.1], [0.8, 7.0, 1.8], [0.7, 5.0, 1.5]]
#               ]
# fig_name = "DPC_UIC1"
# data_lists = [[[0.3, 2.8, 2.7], [0.4, 1.4, 1.9], [0.3, 0.7, 1.2]],
#               [[2.7, 1.1, 5.6], [1.4, 0.8, 3.1], [1.42, 0.07, 1.83]],
#               [[0.9, 2.1, 2.5], [3.34, 0, 0], [2.36, 0, 0]]
#               ]
# fig_name = "DPC_UIC2"
# data_lists = [[[0.6, 7.5, 1.4], [0.6, 4.3, 1.1], [0.3, 2.6, 0.8]],
#               [[0.8, 12.2, 2.8], [0.6, 7.2, 3.0], [0.6, 4.1, 2.6]],
#               [[1.12, 4.58, 4.2], [0.9, 2.5, 2.5], [1.1, 1.1, 1.3]]
#               ]
# fig_name = "DPC_UIC4"
# data_lists = [[[0.3, 1.2, 0.9], [0.2, 0.5, 0.6], [0.2, 0.2, 0.4]],
#               [[0.3, 1.0, 1.0], [0.4, 0.4, 0.6], [0.3, 0.2, 0.6]],
#               [[0.7, 4.6, 2.6], [1.2, 1.2, 1.9], [2.9, 0, 0]]
#               ]
# fig_name = "DPC_UIC5"
# data_lists = [[[0.12, 0.38, 0.38], [0.1, 0.05, 0.29], [0.28, 0, 0]],
#               [[0.14, 0.09, 0.59], [0.125, 0.011, 0.314], [0.35, 0, 0]],
#               [[1.22, 2.35, 1.35], [1.23, 0.24, 1.2], [2.65, 0, 0]]
#               ]
# fig_name = "DPC_UIC7"
# ResultManager.draw_gradual_bars(bar_lists, fig_name, 20)
# ResultManager.draw_bars(data_lists, fig_name)





# # full_layer = [3.55, 3.54, 3.642857142857143, 3.494949494949495, 3.6907216494845363, 3.642857142857143, 3.676767676767677, 3.5918367346938775, 3.595959595959596, 3.62]
# full_layer = [3.5918367346938775, 3.606060606060606, 3.673469387755102, 3.5833333333333335, 3.65, 3.673469387755102, 3.5257731958762886, 3.673469387755102, 3.622448979591837, 3.5454545454545454]
# x = []
# # y = [1.0, 0.7992577564436806, 0.670044960573386, 0.5299112358612116, 0.3755079540407228, 0.2274054323989568, 0.13414592395499694, 0.07315002222283565, 0.03831029235063559, 0.019473832329265504]
# y = [1.0, 0.8141896643964945, 0.6662734176254579, 0.5229005456349207, 0.3662424886621316, 0.22749159965589388, 0.1321417364652643, 0.0718186279946968, 0.03835405936559384, 0.020274669822840825]
#
# reduce = []
# for i in range(len(full_layer)):
#     x.append(full_layer[i] / 6)
#     reduce.append((((1 - x[i]) * (1 - y[i]) * 21) / 41) + (x[i] * (1 - y[i])))
# print(x)
# print(reduce)