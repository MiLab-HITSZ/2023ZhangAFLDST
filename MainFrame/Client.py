import os
import random

import numpy as np
import copy
import math
from random import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

import utils.Tools as Tools
import Models.FC3 as FC3
import Models.VGG13 as VGG13
import Models.CNN as CNN
import Datasets.MNIST as MNIST

# gpu_id = random.randint(0, 4)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)


class Client:
    def __init__(self, use_gpu_id, data_type, model_name, client_data, batch_size):
        self.use_gpu_id = use_gpu_id
        self.data_type = data_type
        self.model_name = model_name
        self.client_data = client_data
        self.batch_size = batch_size

        self.adjust = False
        self.ready_to_return = False
        self.staleness = 0
        self.staleness_penalty = 0.0
        self.decay_rate = 0.0
        self.return_count = 0
        self.full_layer_interval = 0
        self.last_full_layer_count = 1
        self.select_list = []
        self.select_percentage_list = []
        self.global_model_weights = []
        self.select_percentage_history = []

        # 生成一个神经网络模型
        if self.model_name == "FC3":
            self.network = FC3.FC3(self.data_type)
        elif self.model_name == "VGG13":
            self.network = VGG13.VGG13(self.data_type)
        elif self.model_name == "CNN":
            self.network = CNN.CNN(self.data_type)
        else:
            pass

    def client_train_one_epoch(self, b, e, use_weight_regularization=False, rho=0):
        with tf.device('/gpu:' + str(self.use_gpu_id)):
            # with tf.device('/gpu:1'):
            # 批处理数据
            if self.batch_size != b:
                self.re_batch(b)

            if use_weight_regularization:
                # 存储接受到的初始模型参数
                r_weights = copy.deepcopy(self.network.get_weights())
                # 进行训练
                for i in range(e):
                    self.network.model_train_one_epoch(self.client_data[0], self.client_data[1], self.batch_size,
                                                       r_weights, rho)
            else:
                # 进行训练
                for i in range(e):
                    self.network.model_train_one_epoch(self.client_data[0], self.client_data[1], self.batch_size)

    def receive_global_model(self, global_model_weights):
        if self.ready_to_return is False:  # 本来就在训练中，就什么都不做，当做没收到
            self.network.set_weights(global_model_weights)
            self.global_model_weights = global_model_weights

            # 进行训练
            self.client_train_one_epoch(self.batch_size, 1)

            self.ready_to_return = True

    def return_weights(self):
        self.ready_to_return = False
        self.return_count += 1

        print("staleness: ", self.staleness, "staleness_penalty: ", self.staleness_penalty,
              " == ", (self.staleness * self.staleness_penalty))
        if self.adjust is True:
            # 根据陈旧性调整 select_percentage_list
            if self.staleness == 0:  # 无陈旧性
                for sp in range(len(self.select_percentage_list)):
                    self.select_percentage_list[sp] = 1.0 / (self.return_count * self.decay_rate + 1)
            else:  # 有陈旧性
                for sp in range(len(self.select_percentage_list)):
                    self.select_percentage_list[sp] = 1.0 / (self.staleness * self.staleness_penalty + 1)
                self.staleness = 0  # 还原 stalenesss

            # 间歇性进行全部层传输
            if self.return_count == self.last_full_layer_count + self.full_layer_interval:
                print("-----------------------full select return_count:", self.return_count)
                self.select_list = [True, True, True, True, True, True]
                self.last_full_layer_count = self.return_count
                self.full_layer_interval += 1
            else:
                print("------------------- no full select return_count:", self.return_count, self.last_full_layer_count, "+", self.full_layer_interval)
                self.select_list = [True, True, False, False, False, False]

        self.select_percentage_history.append(self.select_percentage_list[0])
        print("select_percentage_list: ", self.select_percentage_list, "      select_list: ", self.select_list)

        # 先选择部分参数
        part_client_weights, select_np_array_list = Tools.largest_part_transmission_of_weights_list(
            self.network.get_weights(), self.global_model_weights, self.select_percentage_list)
        self.network.set_weights(part_client_weights)

        # 再部分层传输
        full_client_weights = self.network.get_weights()
        fake_global_weights = [[] for _ in range(len(full_client_weights))]
        part_layer_client_weights = Tools.selective_transmission(self.network.get_weights(), fake_global_weights,
                                                                 self.select_list)

        return copy.deepcopy(part_layer_client_weights), self.select_list

    def is_ready_to_return(self):
        random_number = random.random()
        if self.ready_to_return is True:
            if random_number < 0.5:
                return True
            else:
                self.staleness += 1
                return False
        else:
            return False

    def set_adjust(self, adjust):
        self.adjust = adjust

    def set_select(self, select_list, select_percentage_list, staleness_penalty, decay_rate):
        self.select_list = select_list
        self.select_percentage_list = select_percentage_list
        self.staleness_penalty = staleness_penalty
        self.decay_rate = decay_rate

    def get_return_count(self):
        return self.return_count

    def get_select_percentage_history(self):
        if len(self.select_percentage_history) == 0:
            return [], -1
        else:
            return self.select_percentage_history, \
                   sum(self.select_percentage_history) / len(self.select_percentage_history)

    def re_batch(self, b):
        self.batch_size = b

    def get_client_weights(self):
        return self.network.get_weights()

    def set_client_weights(self, weights):
        self.network.set_weights(weights)

    def get_data(self):
        return self.client_data

    def set_data(self, client_data):
        self.client_data = client_data
