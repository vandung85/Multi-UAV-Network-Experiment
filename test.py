import gym
import sys
import math
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def test_dataRate():
    f = 2
    d = 100
    c = 3e8
    LoS = 3  # 单位dB
    NLoS = 23  # 单位dB
    B = 4  # 4MHz
    noise_power = 1e-13
    p_tr = 1
    LoS = 10 ** (math.log(LoS / 10, 10))
    NLoS = 10 ** (math.log(NLoS / 10, 10))
    H = 100
    r = 0
    probability = test_probability(H, r)
    distance = math.sqrt(H**2 + r**2)
    pathLoss_LoS = LoS * (4 * math.pi * f * 1e9 * distance / c) ** 2
    pathLoss_NLoS = NLoS * (4 * math.pi * f * 1e9 * distance / c) ** 2
    pathLoss = probability * pathLoss_LoS + (1 - probability) * pathLoss_NLoS
    capacity = B * math.log(1 + p_tr * (1 / pathLoss) / noise_power, 2)
    print('{}MBps'.format(capacity))
    # pathLoss = LoS * ((4 * math.pi * f * d) / c)**2
    # print(pathLoss)
    # dB = 10 * math.log(pathLoss, 10)
    # print(dB)
    # R = B * math.log(1 + (p_tr / pathLoss) / noise_power, 2)
    # print(R)
    # R = R / 1e6
    # print(R)  # 100m约为30多Mbps

def test_energy():
    V = 25
    P_0 = 99.66
    P_1 = 120.16
    U = 120
    v_0 = 0.002
    A = 0.5
    d_0 = 0.48
    s = 0.0001
    p = 1.225

    # 计算推进能耗
    # 第一部分，blade profile
    part_1 = P_0 * (1 + (3*V**2) / U**2)
    # 第二部分，induced
    part_2 = P_1 * math.sqrt(math.sqrt(1 + (V**4)/4*v_0**4) - V**2/2*v_0**2)
    # 第三部分，parasite
    part_3 = 0.5 * d_0 * p * s * A * V**3
    sum = part_1 + part_2 + part_3
    print(sum)

def test_probability(H, r):  # H表示无人机飞行高度,r表示无人机和用户间的水平距离。
    A = 12.08  # 环境参数a
    B = 0.11  # 环境参数b
    eta = 0
    if r == 0:
        eta = (180 / math.pi) * math.pi / 2  # 单位是°
    else:
        eta = (180 / math.pi) * np.arctan(H / r)
    probability_los = float(1 / (1 + A * np.exp(-B * (eta - A))))
    # print(probability_los)
    return probability_los



if __name__ == '__main__':

    # with open("reward.txt", 'rb') as f:
    #     data = pickle.load(f)
    #     x = [i for i in range(100000)]
    #     plt.plot(x, data[:100000])
    #     plt.xlabel('epoch')
    #     plt.ylabel('reward')
    #     plt.show()
    # with open("duration.txt", 'rb') as f:
    #     data = pickle.load(f)
    #     x = [i for i in range(100000)]
    #     plt.plot(x, data[:100000])
    #     plt.xlabel('epoch')
    #     plt.ylabel('dutation')
    #     plt.show()
    # # with open("fairness.txt", 'rb') as f:
    # #     data = pickle.load(f)
    # #     print(data)







    # name_list = ['1000*1000', '500*500']
    # # num_list = [567.81, 672.32]
    # # num_list1 = [413.21, 532.12]
    # num_list = [0.2122, 0.2312]
    # num_list1 = [0.1892, 0.2331]
    # x = list(range(len(num_list)))
    # total_width, n = 0.5, 2
    # width = total_width / n
    #
    # plt.bar(x, num_list, width=width, label='MADDPG', fc='y')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list1, width=width, label='Random', tick_label=name_list, fc='r')
    # plt.ylim(0, 1)
    # plt.legend()
    # plt.show()