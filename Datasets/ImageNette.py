import tensorflow as tf
import numpy as np
import os
import cv2
import platform
import matplotlib.pyplot as plt
import utils.Tools as Tools

import Datasets.CIFAR10 as CIFAR10
import Models.FC3 as FC3
import Models.CNN as CNN
import Models.VGG13 as VGG13
import Models.VGG16 as VGG16

# image_path = 'ILSVRC2012_val_00000321.JPEG'
if platform.system().lower() == 'windows':
    train_images_path = 'C:/HIT/ImageNet/ImageNette/imagenette2/train'
    val_images_path = 'C:/HIT/ImageNet/ImageNette/imagenette2/val'
    test_image_path = 'C:/HIT/ImageNet/ImageNette/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG'
elif platform.system().lower() == 'linux':
    train_images_path = '/home/zrz/ImageNette/ImageNette/imagenette2/train'
    val_images_path = '/home/zrz/ImageNette/ImageNette/imagenette2/val'
    test_image_path = '/home/zrz/ImageNette/ImageNette/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG'
else:
    train_images_path = '/home/zrz/ImageNette/ImageNette/imagenette2/train'
    val_images_path = '/home/zrz/ImageNette/ImageNette/imagenette2/val'
    test_image_path = '/home/zrz/ImageNette/ImageNette/imagenette2/train/n01440764/ILSVRC2012_val_00000293.JPEG'

short_side_scale = (256, 384)
mean = [103.939, 116.779, 123.68]
std = [58.393, 57.12, 57.375]
category_num = 10


def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)  # 类型转换
    return x, y


def preprocess_data(x, y, batch):
    # 构建训练集对象，随机打乱，预处理，批量化
    db = tf.data.Dataset.from_tensor_slices((x, y))
    db = db.shuffle(1000).map(preprocess, num_parallel_calls=8).batch(batch)  # 构建测试集对象，预处理，批量化
    return db


def draw_pictures(cifar10_pictures):
    for picture in cifar10_pictures:
        plt.figure()
        plt.imshow(picture)
    plt.show()


def random_size(image, target_size=None):
    height, width, _ = np.shape(image)
    if target_size is None:
        target_size = np.random.randint(*short_side_scale)
    if height < width:
        size_ratio = target_size / height
    else:
        size_ratio = target_size / width
    resize_shape = (
        int(width * size_ratio), int(height * size_ratio))
    return cv2.resize(image, resize_shape)


def center_crop(image):
    height, width, _ = np.shape(image)
    input_height, input_width, _ = (224, 224, 3)
    crop_x = (width - input_width) // 2
    crop_y = (height - input_height) // 2
    return image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]


def normalize(image):
    for i in range(3):
        image[..., i] = (image[..., i] - mean[i]) / std[i]
    return image


def load_image(image_path_str, label_number):
    # image = cv2.imread(image_path_str.numpy().decode()).astype(np.float32)
    image = cv2.imread(image_path_str.numpy().decode()).astype(np.uint8)

    # print(type(image), image.shape)
    # print(type(image[0][0]), image[0][0].shape, image[0][0])
    # print(type(image[0][0][0]), image[0][0][0].shape, image[0][0][0])
    image = random_size(image, target_size=256)
    # print(type(image), image.shape)
    # print(type(image[0][0]), image[0][0].shape, image[0][0])
    # print(type(image[0][0][0]), image[0][0][0].shape, image[0][0][0])
    image = center_crop(image)
    # print(type(image), image.shape)
    # print(type(image[0][0]), image[0][0].shape, image[0][0])
    # print(type(image[0][0][0]), image[0][0][0].shape, image[0][0][0])
    # image = normalize(image)
    # print(type(image), image.shape)
    # print(type(image[0][0]), image[0][0].shape, image[0][0])
    # print(type(image[0][0][0]), image[0][0][0].shape, image[0][0][0])
    # print()

    label_one_hot = np.zeros(category_num)
    label_one_hot[label_number] = 1.0
    img_label = tf.argmax(label_one_hot)
    img_label = np.array([img_label])

    return image, img_label


def train_one_epoch(network, train_data, optimizer, received_weights=None, rho=0.005):
    for step, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:  # 构建梯度记录环境
            out = network(x, training=True)
            out = tf.reshape(out, [out.shape[0], -1])

            # 真实标签 one-hot 编码，[b] => [b, 10]
            # if len(y.shape) > 1:
            #     y = tf.squeeze(y, axis=1)
            # print("out, y:", out, y)

            y = tf.squeeze(y, axis=1)
            y_one_hot = tf.one_hot(y, depth=10)
            # y_one_hot = tf.reshape(y_one_hot, [y_one_hot.shape[0], -1])
            # 计算交叉熵损失函数，标量
            loss = tf.losses.categorical_crossentropy(y_one_hot, out, from_logits=True)

            if received_weights is not None:
                # 计算正则项
                difference = tf.constant(0, dtype=tf.float32)
                for layer in range(len(received_weights)):
                    received_weights[layer] = tf.cast(received_weights[layer], dtype=tf.float32)
                    w_difference = tf.math.square(received_weights[layer] - network.trainable_variables[layer])
                    difference += tf.math.reduce_sum(w_difference)

                loss = tf.math.reduce_mean(loss) + (rho * difference)
            else:
                loss = tf.math.reduce_mean(loss)

        # 对所有参数求梯度
        grads = tape.gradient(loss, network.trainable_variables)
        # 自动更新
        optimizer.apply_gradients(zip(grads, network.trainable_variables))
    return network.get_weights()


class ImageNette:
    def __init__(self):
        x, y, x_test, y_test = [], [], [], []
        for [images_packages_path, x_list, y_list] in [[train_images_path, x, y], [val_images_path, x_test, y_test]]:
            image_label = 0
            for images in os.listdir(images_packages_path):
                images_path = images_packages_path + '/' + str(images)
                for image_file in os.listdir(images_path):
                    image_path = images_path + '/' + str(image_file)
                    image, label = load_image(tf.constant(image_path), image_label)
                    x_list.append(image)
                    y_list.append(label)
                image_label += 1

        x = np.array(x)
        y = np.array(y)
        y = y.astype('uint8')
        x_test = np.array(x_test)
        y_test = np.array(y_test)
        y_test = y_test.astype('uint8')

        print(type(x), x.shape, "|-\t-\t-|", type(y), y.shape)
        print(type(x[0]), x[0].shape, "|-\t-\t-\t-|", type(y[0]), y[0].shape, y[0])
        print(type(x[0][0]), x[0][0].shape, "|-\t-\t-\t-|", type(y[0][0]), y[0][0].shape, y[0][0])
        print(type(x[0][0][0]), x[0][0][0].shape, x[0][0][0])
        print(type(x[0][0][0][0]), x[0][0][0][0].shape, x[0][0][0][0])
        print("====================================================================================")
        print(type(x_test), x_test.shape, "|-\t-\t-|", type(y_test), y_test.shape)
        print(type(x_test[0]), x_test[0].shape, "|-\t-\t-\t-|", type(y_test[0]), y_test[0].shape, y_test[0])
        print(type(x_test[0][0]), x_test[0][0].shape, "|-\t-\t-\t-|", type(y_test[0][0]), y_test[0][0].shape, y_test[0][0])
        print(type(x_test[0][0][0]), x_test[0][0][0].shape, x_test[0][0][0])
        print(type(x_test[0][0][0][0]), x_test[0][0][0][0].shape, x_test[0][0][0][0])

        self.train_data = [x, y]
        self.pre_valid_data = [x_test[0: 50], y_test[0: 50]]
        self.valid_data = [x_test[100: 1500], y_test[100: 1500]]
        self.test_data = [x_test[1500: 3925], y_test[1500: 3925]]
        self.big_test_data = [x_test, y_test]

    def get_sorted_dataset(self):
        x, y = self.train_data[0], self.train_data[1]
        sorted_imagenette_x, sorted_imagenette_y = Tools.generate_sorted_dataset(x, y, 9469)
        return sorted_imagenette_x, sorted_imagenette_y

    def get_train_data(self):
        return self.train_data[0], self.train_data[1]

    def get_pre_valid_data(self):
        return self.pre_valid_data[0], self.pre_valid_data[1]

    def get_valid_data(self):
        return self.valid_data[0], self.valid_data[1]

    def get_test_data(self):
        return self.test_data[0], self.test_data[1]

    def get_big_test_data(self):
        return self.big_test_data[0], self.big_test_data[1]


if __name__ == '__main__':
    Tools.set_gpu_with_increasing_occupancy_mode()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.random.set_seed(2345)

    img, img_label = load_image(tf.constant(test_image_path), 4)
    print(type(img), img.shape)
    print(type(img_label), img_label.shape, img_label)
    draw_pictures([img])

    # cifar10 = CIFAR10.CIFAR10()
    # cifar10_train_data = cifar10.get_train_data()
    # print(type(cifar10_train_data[0][0]), cifar10_train_data[0][0].shape)
    # print(type(cifar10_train_data[1][0]), cifar10_train_data[1][0].shape, cifar10_train_data[1][0])

    # vgg13 = VGG13.VGG13("CIFAR10")
    # vgg13.model_train_one_epoch([cifar10_train_data[0][0]], [cifar10_train_data[1][0]], 1)

    vgg16 = VGG16.VGG16("ImageNette")
    vgg16.model_train_one_epoch([img], [img_label], 1)

    accuracy, loss = vgg16.evaluate_network([img], [img_label])
    print(accuracy, loss)

    ImageNette = ImageNette()
    x, y = ImageNette.get_train_data()
    x_test, y_test = ImageNette.get_big_test_data()

    # x, y, x_test, y_test = [], [], [], []
    # for [images_packages_path, x_list, y_list] in [[train_images_path, x, y], [val_images_path, x_test, y_test]]:
    #     image_label = 0
    #     for images in os.listdir(images_packages_path):
    #         images_path = images_packages_path + '/' + str(images)
    #         for image_file in os.listdir(images_path):
    #             image_path = images_path + '/' + str(image_file)
    #             image, label = load_image(tf.constant(image_path), image_label)
    #             x_list.append(image)
    #             y_list.append(label)
    #         image_label += 1
    #     print(len(x_list), len(y_list))
    #     draw_pictures([x_list[0]])

    vgg16.model_train_one_epoch(x[0: 1], y[0: 1], 1)

    accuracy, loss = vgg16.evaluate_network(x_test[0: 3], y_test[0: 3])
    print(accuracy, loss)

    accuracy, loss = vgg16.evaluate_network(x_test[454: 459], y_test[454: 459])
    print(accuracy, loss)
