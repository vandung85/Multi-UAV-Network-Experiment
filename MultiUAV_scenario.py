import numpy as np
import math
from core import World, UAV, Landmark, Entity

# 全局参数 这里暂且设置可偏转角度为90°，即在飞行高度为100m的情况下，无人机投影和可服务的地面用户的水平最大距离为100m
epsilon = 1e-3
R_eq = 1  # 14  # 需要平均数据率，需要综合各种参数综合制定。在AB的概率，H=100m条件下,K=5,Bandwidth=20,每个子信道的带宽资源为4,最优情况为75MBps,最差情况capacity为65MBps。在每个时隙，2个UAV最多可以同时服务10个地面用户，以65为标准，50个地面用户的平均速率为10*64/50≈12.8
A = 12.08  # 城市环境参数a 城郊为4.88
B = 0.11  # 城市环境参数b 城郊为0.43
F = 2  # 载波频率 单位GHz
LoS = 3  # LoS条件下额外路径损失系数，单位为dB，3dB对应2倍
NLoS = 23  # NLoS条件下的
sigma_power = 1e-13  # 加性高斯白噪声功率，单位为dBm  -100dBm 对应 0.1mW
H = 100  # UAV飞行高度
P_tr = 1  # 发送功率 单位W
Bandwidth = 20  # 信道带宽 单位MHz,但带入香农公式中应转化为Hz
# K = 3  # 子信道划分数量 该版本不再使用，该版本中每个UAV服务覆盖范围内的所有UE
# 注意，信道容量计算公式中信噪比的单位不是dB
# 注意，单位需要转化，代码中的单位未进行转化
T = 1000  # 总执行时长，代表多少个timeslot,120个时隙为2400s。
t = 20  # 每个时隙为20s
alpha = 0.2  # 能耗所占比重
# 能耗参数
P_h = 40  # 悬停所消耗的能量 单位W
P_f = 120  # 飞行时的功耗
Energy = 2e5  # 无人机初始总能量，单位J 可供无人机进行2000s的悬停
V = 20  # 无人机固定飞行速度，单位m/s
Range = 500

# 单位转换 dB转为W
LoS = 10 ** (math.log(LoS / 10, 10))
NLoS = 10 ** (math.log(NLoS / 10, 10))


class Scenario:
    def make_world(self):
        world = World()
        world.num_UAVs = 2
        world.num_landmarks = 30  # 50
        world.UAVs = [UAV() for i in range(world.num_UAVs)]
        world.association = []
        world.probability_LoS = 1 / (1 + A)
        for i, uav in enumerate(world.UAVs):
            uav.name = 'UAV %d'.format(i)
            uav.id = i
            uav.size = 10
            uav.state.energy = Energy
        # # 列表形式的landmarks
        # world.landmarks = [Landmark() for i in range(world.num_landmarks)]
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.name = 'landmark %d'.format(i)
        #     landmark.id = i
        #     landmark.size = 10
        # 字典形式的landmarks
        for i in range(world.num_landmarks):
            dic = {str(i): Landmark()}
            world.landmarks.update(dic)
        for i, landmark in enumerate(world.landmarks.values()):
            landmark.name = 'landmark %d'.format(i)
            landmark.id = i
            landmark.size = 5
        self.reset_world(world)  # 这里不reset会导致MultiUAVEnv中init中获得observation维度报错
        return world

    def reset_world(self, world):
        # world时隙初始化
        world.t = 0
        # 位置初始化,设置随机参数
        np.random.seed(666)
        landmarks_position = np.random.uniform(0, Range, (len(world.landmarks), 2))
        np.random.seed(None)  # 取消随机种子
        for i, landmark in enumerate(world.landmarks.values()):
            landmark.state.pos = landmarks_position[i]
            landmark.weight = 1
            landmark.sum_throughput = 0
            landmark.avg_dataRate = 0
        for uav in world.UAVs:
            uav.state.pos = np.array([Range / 2, Range / 2])
            # uav.state.pos = np.random.uniform(0, Range, world.dim_p)
        # 能耗初始化
        for uav in world.UAVs:
            uav.state.energy = Energy
        self.reset_service(world)

    # 使用全局奖励 or 分开？
    def reward(self, world):
        _, reward_list = self.get_sum_capacity(world)
        # capacity_sum = np.sum(capacity_list)
        # reward = capacity_sum
        reward = []
        for i in range(world.num_UAVs):
            reward.append(reward_list[i]/100)  # 归一化奖励
        return reward

    def observation(self, world, uav):
        # 覆盖范围/观测范围
        obs_position = [uav.state.pos / Range]  # 归一化
        for uav_tmp in world.UAVs:
            if uav is uav_tmp:
                continue
            else:
                obs_position.append((uav_tmp.state.pos - uav.state.pos)/Range)
            # obs_position.append(uav.state.pos)
        obs_weight = []
        for landmark in world.landmarks.values():
            obs_weight.append(landmark.weight)
        return np.concatenate((np.concatenate(obs_position), np.array(obs_weight)))

    def step(self, world):
        # 标致位，用来判断UAV此次运动是否越界
        is_out_bound = [False for _ in range(world.num_UAVs)]
        # 时隙自增
        world.t += 1
        # reset 服务关联
        for uav in world.UAVs:
            uav.associator.clear()
        for landmark in world.landmarks.values():
            landmark.connected = False
        # 更新位置和能耗
        for i, uav in enumerate(world.UAVs):
            distance_x = uav.action.distance_x * uav.max_distance
            distance_y = uav.action.distance_y * uav.max_distance
            uav.state.pos[0] += distance_x
            uav.state.pos[1] += distance_y

            if uav.state.pos[0] < 0 or uav.state.pos[0] > Range or uav.state.pos[1] < 0 or uav.state.pos[1] > Range:
                is_out_bound[i] = True
                continue

            # uav统计覆盖范围内的地面用户数量
            for landmark in world.landmarks.values():
                if landmark.connected is False and np.sqrt(np.sum((landmark.state.pos - uav.state.pos) ** 2)) <= 40:
                    world.landmarks[str(landmark.id)].connected = True
                    # world.landmarks[str(landmark.id)].sum_throughput += self.get_capacity(uav, landmark)
                    uav.associator.append(landmark.id)

            # 计算uav和覆盖范围内的用户的数据传输率
            for landmark_id in uav.associator:
                world.landmarks[str(landmark_id)].sum_throughput += self.get_capacity(uav,
                                                                                      world.landmarks[str(landmark_id)])

            # 更新landmark的平均数据率和公平比例权重
        for landmark in world.landmarks.values():
            landmark.avg_dataRate = landmark.sum_throughput / world.t  # 考虑是否要乘上t（时隙长度）
            landmark.weight = 1 / (1 + landmark.avg_dataRate)  # epsilon

        return is_out_bound

    def get_done(self, world):
        done = []
        for uav in world.UAVs:
            if uav.state.pos[0] < 0 or uav.state.pos[0] > Range or uav.state.pos[1] < 0 or uav.state.pos[1] > Range:
                done.append(True)
            else:
                done.append(False)
        return done

    def reset_service(self, world):
        for uav in world.UAVs:
            uav.state.curServiceNum = 0
            uav.associator = []
        for landmark in world.landmarks.values():
            landmark.connected = False

    # 计算当前数据传输率
    def get_sum_capacity(self, world):
        capacity_list = []
        reward_list = []
        for uav in world.UAVs:
            capacity = 0
            reward = 0
            for id in uav.associator:
                landmark = world.landmarks[str(id)]
                probability_los = self.get_probability(uav.state.pos, landmark.state.pos)  # 获得LoS概率
                # print("建立LoS的概率为{:.4f}".format(probability_los))
                pathLoss = self.get_passLoss(uav.state.pos, landmark.state.pos, probability_los)  # 获得平均路径损失
                capacity += (Bandwidth / len(uav.associator)) * math.log(1 + P_tr * (1 / pathLoss) / sigma_power, 2)
                reward += (Bandwidth / len(uav.associator)) * math.log(1 + P_tr * (1 / pathLoss) / sigma_power,
                                                                       2) * landmark.weight  # 根据香农公式计算信道容量,并乘上权重。
            capacity_list.append(capacity)
            reward_list.append(reward)
        return capacity_list, reward_list

    # 计算某个UAV和地面设备之间的数据速率
    def get_capacity(self, uav, landmark):
        probability_los = self.get_probability(uav.state.pos, landmark.state.pos)  # 获得LoS概率
        pathLoss = self.get_passLoss(uav.state.pos, landmark.state.pos, probability_los)  # 获得平均路径损失
        # 根据香农公式计算信道容量
        capacity = (Bandwidth / len(uav.associator)) * math.log(1 + P_tr * (1 / pathLoss) / sigma_power, 2)
        return capacity

    def get_probability(self, uav_pos, landmark_pos):
        r = np.sqrt(np.sum((landmark_pos - uav_pos) ** 2))
        eta = 0
        if r == 0:
            eta = (180 / math.pi) * math.pi / 2  # 单位是°
        else:
            eta = (180 / math.pi) * np.arctan(H / r)
        # print("eta:{}".format(eta))
        # print(A)
        # print(B)
        probability_los = float(1 / (1 + A * np.exp(-B * (eta - A))))
        # print(probability_los)
        return probability_los

    def get_passLoss(self, uav_pos, landmark_pos, probability_los):
        distance = self.get_distance(uav_pos, landmark_pos)
        pathLoss_LoS = LoS * (4 * math.pi * F * 1e9 * distance / 3e8) ** 2
        pathLoss_NLoS = NLoS * (4 * math.pi * F * 1e9 * distance / 3e8) ** 2
        return probability_los * pathLoss_LoS + (1 - probability_los) * pathLoss_NLoS

    def get_distance(self, uav_pos, landmark_pos):
        distance = np.sqrt(np.sum((landmark_pos - uav_pos) ** 2) + H ** 2)
        return distance


if __name__ == '__main__':
    sc = Scenario()
    a = np.array([816.21, 531.57])
    b = np.array([752.02, 523.63])
    probability = sc.get_probability(a, b)
    print(probability)
