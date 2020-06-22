import gym
import sys
import math
import numpy as np

def test_dataRate():
    f = 2e9
    d = 100
    c = 3e8
    LoS = 2  # 对应3dB
    B = 2e6  # 2MHz
    noise_power = 1e-13
    p_tr = 1

    # 计算信道容量
    pathLoss = LoS * ((4 * math.pi * f * d) / c)**2
    print(pathLoss)
    dB = 10 * math.log(pathLoss, 10)
    print(dB)
    R = B * math.log(1 + (p_tr / pathLoss) / noise_power, 2)
    print(R)
    R = R / 1e6
    print(R)  # 100m约为30多Mbps

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

def test_probability():
    A = 12.08  # 环境参数a
    B = 0.11  # 环境参数b
    eta = 57
    probability_los = float(1 / (1 + A * np.exp(-B * (eta - A))))
    print(probability_los)


if __name__ == '__main__':
    a = np.array([True, True])
    b = np.array([2, 3])
    c = np.multiply(~a, b)
    print(~c)