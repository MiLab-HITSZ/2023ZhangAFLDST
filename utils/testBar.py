# import matplotlib.pyplot as plt
# from matplotlib import cm
#
#
# def draw_bar(key_name, key_values):
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False
#
#     # 标准柱状图的值
#     def autolable(rects):
#         for rect in rects:
#             height = rect.get_height()
#             if height >= 0:
#                 plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height + 0.02, '%.3f' % height)
#             else:
#                 plt.text(rect.get_x() + rect.get_width() / 2.0 - 0.3, height - 0.06, '%.3f' % height)
#                 # 如果存在小于0的数值，则画0刻度横向直线
#                 plt.axhline(y=0, color='black')
#
#     # 归一化
#     norm = plt.Normalize(0, 1)
#     norm_values = norm(key_values)
#     map_vir = cm.get_cmap(name='inferno')
#     colors = map_vir(norm_values)
#     fig = plt.figure()  # 调用figure创建一个绘图对象
#     plt.subplot(111)
#     ax = plt.bar(key_name, key_values, width=0.5, color=colors, edgecolor='black')  # edgecolor边框颜色
#
#     sm = cm.ScalarMappable(cmap=map_vir, norm=norm)  # norm设置最大最小值
#     sm.set_array([])
#     plt.colorbar(sm)
#     autolable(ax)
#
#     plt.show()
#
#
# if __name__ == '__main__':
#     # multi_corr()
#     key_name = ['时长', '鹤位', '设定量', '发油量', '发油率', '时间', '月份', '日期', '损溢量', '温度', '密度']
#     key_values = [0.1, 0.9, 1, 1, 0.4, 0.3, -0.1, -0.6, 0.2, 0.4, 0.5]
#     draw_bar(key_name, key_values)
#

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as p3d
from collections import OrderedDict
import seaborn as sns

# l1_1 = plt.barh(1, 1, height=0.1, left=0, color='#ff7f0e', alpha=1)
# l1_2 = plt.barh(1, 1, height=0.1, left=1, color='#ff7f0e', alpha=0.5)
# l1_3 = plt.barh(1, 1, height=0.1, left=2, color='#ff7f0e', alpha=0.2)

# n = 100
# for i in range(n):
#     left = i
#     alpha = i * float(1 / n)
#     l1_1 = plt.barh(1, 1, height=0.1, left=left, color='#1f77b4', alpha=alpha)
# l1_1 = plt.barh(1, 100, height=0.1, left=n, color='#1f77b4', alpha=1)
# for i in range(n):
#     left = i + 200
#     alpha = 1 - i * float(1 / n)
#     l1_1 = plt.barh(1, 1, height=0.1, left=left, color='#1f77b4', alpha=alpha)

cmaps = OrderedDict()
cmaps['Sequential'] = ['Blues']
print(cmaps.items())
print(plt.get_cmap('Blues'))

# sns.palplot(sns.color_palette("Blues", 10))
#
# sns.palplot(sns.color_palette("Blues", 10)[0:7])
#
# new_blues = sns.color_palette("Blues", 10)[0:7]

fig = plt.figure()
ax = plt.axes()
# x = np.random.rand(1000)
# y = np.random.rand(1000)
# fig = plt.figure()
# ax = fig.subplots(2, 1)
# # ax = fig.subplots(2)
# # ax = fig.subplots(1, 2)
# ax = ax.flatten()
# ax0 = ax[0].scatter(x, y, c=x, cmap="Blues")
# fig.colorbar(ax0, ax=ax[0])
# b = plt.barh(1, width=1, height=1, left=1, alpha=1)
# colors = [(225, 119, 76, 1), (180, 137, 31, 1), (225, 119, 76, 1), (180, 119, 31, 1)]
colors = []
bn = []
for i in range(100):
    colors.append((180, 119, 31, 1))
    bn.append(i * 0.01)
# cmp = mpl.colors.ListedColormap(['#1f77b4', 'g', 'b', 'r'])
cmp = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bn, cmp.N)
fcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), ax=ax)
# plt.scatter(x, y, c=y, cmap="Blues")
# plt.colorbar()

plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
#
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
#
# # iris_df = pd.read_csv('../Topics/iris.csv', index_col='index_col')
# # x = iris_df['PetalLength'].values
# # y = iris_df['SepalLength'].values
#
# x = np.random.rand(1000)
# y = np.random.rand(1000)
#
# fig = plt.figure()
# ax = plt.axes()
#
# # 创建一个ListedColormap实例
# # 定义了[0, 1]区间的浮点数到颜色的映射规则
# # cmp = mpl.colors.ListedColormap(['r', 'g', 'b'])
# cmp = mpl.colors.BoundaryNorm([0, 1], ncolors="red")
#
# # 创建一个BoundaryNorm实例
# # BoundaryNorm是数据分组中数据归一化比较好的方法
# # 定义了变量值到 [0, 1]区间的映射规则，即数据归一化
# norm = mpl.colors.BoundaryNorm([0, 2, 6.4, 7], cmp.N)
#
# # 绘制散点图，用x值着色，
# # 使用norm对变量值进行归一化，
# # 使用自定义的ListedColormap颜色映射实例
# # norm将变量x的值归一化
# # cmap将归一化的数据映射到颜色
# plt.scatter(x, y, c=x, cmap=cmp, norm=norm, alpha=0.5)
#
# fcb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmp), ax=ax)
#
# plt.show()
