import tensorflow as tf
from tensorflow import keras
import os
import datetime
import time
import threading
from multiprocessing import Queue
import copy
import random

import tensorflow as tf

import AsyncClient
import utils.Tools as Tools
import utils.ResultManager as ResultManager
from Server import Server
from Server import Server
import Datasets.MNIST as MNIST
import Datasets.FashionMNIST as FashionMNIST
import Datasets.CIFAR10 as CIFAR10
import Models.FC3 as FC3
import Models.VGG13 as VGG13
import DensityPeaks.ClusterWithDensityPeaks as ClusterWithDensityPeaks
import os
import numpy as np
import math
import copy
from random import shuffle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, losses, optimizers, datasets, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

import utils.Tools as Tools


# 执行自定义训练过程
def mnist_fnn_linear_attack_train(model, optimizer):
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist).batch(128)

    # 初始化模型
    model.build(input_shape=[128, 784])

    # 获得要窃取的数据并转化成与权重矩阵相同shape
    data, weight_position = mnist_fnn_linear_attack(model, x_test)

    loss_list = []
    acc_list = []

    # 执行训练过程
    for epoch in range(10):
        loss_print = 0
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x, training=True)
                out = tf.squeeze(out, axis=1)
                # 计算正则项
                regular = tf.constant(0, dtype=tf.float32)
                for i in range(len(data)):
                    assert (data[i].shape == model.trainable_variables[weight_position[i]].shape)
                    regular += tf.reduce_mean(abs(data[i] - abs(model.trainable_variables[weight_position[i]]) -
                                                  abs(model.trainable_variables[weight_position[i] + 1])))

                # 计算损失函数
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False)) + 20 * regular
                # loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False))
                loss_print += float(loss)

            # 执行梯度下降
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 获得对测试集的准确率
        acc = test(model, test_db)
        loss_list.append(loss_print)
        acc_list.append(acc_list)
        print('epoch:', epoch, 'loss:', loss_print / (50000 / 128), 'Evaluate Acc:', float(acc))
    # 展示结果
    # plt.plot(loss_list)
    # plt.plot(acc_list)
    # plt.show()
    show_linear_attack_data(x_test, model)
