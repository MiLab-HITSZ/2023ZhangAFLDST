import os
import math

IOU_THRESHOLD = 0.5
TARGET_TXT_PATH = 'C:/HIT/TZB/train/mask'
PREDICT_TXT_PATH = 'C:/HIT/TZB/train/mask'


def read_txt(file_path):
    r_file = open(file_path, 'r')
    info_list = []
    # 读取epochs
    for line in r_file:
        if line == '\n':
            break
        info_list.append(line)
    return info_list


def calculate_iou(ship1, ship2):
    x1, x2 = min(ship1.x1, ship1.x3), max(ship1.x1, ship1.x3)  # ship1的左上角与右下角的x坐标
    y1, y2 = min(ship1.y1, ship1.y3), max(ship1.y1, ship1.y3)  # ship1的左上角与右下角的y坐标
    x3, x4 = min(ship2.x1, ship2.x3), max(ship2.x1, ship2.x3)  # ship2的左上角与右下角的x坐标
    y3, y4 = min(ship2.y1, ship2.y3), max(ship2.y1, ship2.y3)  # ship2的左上角与右下角的y坐标

    if x2 <= x3 or x4 <= x1 or y2 <= y3 or y4 <= y1:
        iou = 0  # 无重叠部分
    else:
        length = min(x2, x4) - max(x1, x3)
        width = min(y2, y4) - max(y1, y3)
        overlap_area = length * width
        iou = (2 * overlap_area) / (ship1.area + ship2.area)
    return iou


def calculate_f1(correct_num, wrong_num, miss_num):
    # 召回率
    if (correct_num + miss_num) > 0:
        r = correct_num / (correct_num + miss_num)
    else:
        r = -1
    # 精确率
    if (correct_num + wrong_num) > 0:
        p = correct_num / (correct_num + wrong_num)
    else:
        p = -1
    # F1 分数
    if (p + r) > 0:
        f1 = (2 * p * r) / (p + r)
    else:
        f1 = -1
    return f1, r, p


class Ship:
    def __init__(self, ship_id, ship_info):
        self.id = ship_id
        self.split_ship_info = ship_info.split(",")
        self.ship_name = self.split_ship_info[0]
        self.x1 = float(self.split_ship_info[1])
        self.y1 = float(self.split_ship_info[2])
        self.x2 = float(self.split_ship_info[3])
        self.y2 = float(self.split_ship_info[4])
        self.x3 = float(self.split_ship_info[5])
        self.y3 = float(self.split_ship_info[6])
        self.x4 = float(self.split_ship_info[7])
        self.y4 = float(self.split_ship_info[8])
        self.length = math.sqrt(((self.x3 - self.x2) ** 2) + ((self.y3 - self.y2) ** 2))
        self.width = math.sqrt(((self.x3 - self.x4) ** 2) + ((self.y3 - self.y4) ** 2))
        # self.length = math.sqrt(((self.x4 - self.x1) ** 2) + ((self.y4 - self.y1) ** 2))
        # self.width = math.sqrt(((self.x2 - self.x1) ** 2) + ((self.y2 - self.y1) ** 2))
        # print(self.length, self.width)
        # self.area = (self.x3 - self.x1) * (self.y3 - self.y1)
        self.area = self.length * self.width
        # print("self.area =", self.area)

    def calculate_score(self, ship_list):
        n = 0
        is_correct = False
        for s in ship_list:
            iou = calculate_iou(self, s)
            if iou > 0:
                n += 1
            if iou >= IOU_THRESHOLD:
                is_correct = True

        if is_correct:
            correct_score = 1
        else:
            correct_score = 0
        wrong_score = n - correct_score
        if n > 0:
            miss_score = 0
        else:
            miss_score = 1
        return correct_score, wrong_score, miss_score


if __name__ == '__main__':
    for target_txt in os.listdir(TARGET_TXT_PATH):
        target_ship_info_list = read_txt(TARGET_TXT_PATH + "/" + target_txt)
        predict_ship_info_list = read_txt(PREDICT_TXT_PATH + "/1_test.txt")

        # 生成各个target_ship和各个predict_ship
        target_ship_list, predict_ship_list = [], []
        for t_s_id in range(len(target_ship_info_list)):
            target_ship_list.append(Ship(t_s_id, target_ship_info_list[t_s_id]))
        for p_s_id in range(len(predict_ship_info_list)):
            predict_ship_list.append(Ship(p_s_id, predict_ship_info_list[p_s_id]))

        # 计算该图中正确预测数与错误预测数，然后计算召回率、精确率和F1分数
        correct_sum, wrong_sum, miss_sum = 0, 0, 0
        for ship in target_ship_list:
            correct, wrong, miss = ship.calculate_score(predict_ship_list)
            correct_sum += correct
            wrong_sum += wrong
            miss_sum += miss
        image_f1, image_r, image_p = calculate_f1(correct_sum, wrong_sum, miss_sum)
        print("Image", target_txt, ":  F1 =", image_f1, ", R =", image_r, ", P =", image_p)
