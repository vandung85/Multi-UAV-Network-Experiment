import numpy as np
import math
from core import World, UAV, Landmark, Entity

# 全局参数
R_eq = 1000  # 需要平均数据率，需要综合各种参数综合制定。
A = 12.08  # 城市环境参数a 城郊为4.88
B = 0.11  # 城市环境参数b 城郊为0.43
F = 2  # 载波频率 单位GHz
LoS = 3  # LoS条件下额外路径损失系数，单位为dB，3dB对应2倍
NLoS = 23  # NLoS条件下的
sigma_power = 1e-13  # 加性高斯白噪声功率，单位为dBm  -100dBm 对应 0.1mW
H = 100  # UAV飞行高度
P_tr = 1  # 发送功率 单位W
Bandwidth = 20  # 信道带宽 单位MHz,但带入香农公式中应转化为Hz
K = 3  # 子信道划分数量
# 注意，信道容量计算公式中信噪比的单位不是dB
# 注意，单位需要转化，代码中的单位未进行转化
T = 120  # 总执行时长，代表多少个timeslot,120个时隙为2400s。
t = 20  # 每个时隙为20s
alpha = 0.2  # 能耗所占比重
# 能耗参数
P_h = 40  # 悬停所消耗的能量 单位W
P_f = 120  # 飞行时的功耗
Energy = 2e5  # 无人机初始总能量，单位J 可供无人机进行2000s的悬停
V = 20  # 无人机固定飞行速度，单位m/s

# 单位转换 dB转为W
LoS = 10**(math.log(LoS/10, 10))
NLoS = 10**(math.log(NLoS/10, 10))


class Scenario:
    def make_world(self):
        world = World()
        world.num_UAVs = 2
        world.num_landmarks = 30
        world.UAVs = [UAV() for i in range(world.num_UAVs)]
        world.association = []
        world.probability_LoS = 1/(1+A)
        world.rate = np.zeros((3,50))  # 传输速率矩阵
        for i, uav in enumerate(world.UAVs):
            uav.name = 'UAV %d'.format(i)
            uav.id = i
            uav.size = 20
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
            landmark.size = 10
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # world时隙初始化
        world.t = 0
        # 位置初始化,设置随机参数
        np.random.seed(666)
        landmarks_position = np.random.uniform(0, 1000, (len(world.landmarks), 2))
        for uav in world.UAVs:
            uav.state.pos = np.random.uniform(0, 1000, world.dim_p)
        for i, landmark in enumerate(world.landmarks.values()):
            landmark.state.pos = landmarks_position[i]
        # 能耗初始化
        for uav in world.UAVs:
            uav.state.energy = Energy

    def reward(self, world):
        capacity_list = self.get_sum_capacity(world)
        capacity_sum = np.sum(capacity_list)
        reward = capacity_sum
        return reward

    def observation(self, world, uav):
        # 覆盖范围/观测范围
        coverage = 100
        obs_position = []
        for uav in world.UAVs:
            obs_position.append(uav.state.pos)
        obs_weight = []
        for landmark in world.landmarks.values():
            obs_weight.append(landmark.weight)
        return np.concatenate((np.concatenate(obs_position), np.array(obs_weight)))

    def step(self, world):
        # 时隙自增
        world.t += 1
        # reset 服务关联
        for uav in world.UAVs:
            uav.associator.clear()
            uav.state.curServiceNum = 0
        for landmark in world.landmarks.values():
            landmark.connected = False
        # 更新位置和能耗
        for uav in world.UAVs:
            pos_temp = uav.state.pos.copy()  # 保存位置信息，用于越界后的回退。。。这样搞貌似内存消耗有点大
            direction = uav.action.direction * 2 * math.pi
            distance = uav.action.distance * uav.max_distance
            if direction <= math.pi/2:
                uav.state.pos[0] -= distance * math.cos(direction)
                uav.state.pos[1] += distance * math.sin(direction)
            elif math.pi/2 < direction <= math.pi:
                uav.state.pos[0] += distance * math.cos(math.pi-direction)
                uav.state.pos[1] += distance * math.sin(math.pi-direction)
            elif math.pi < direction <= math.pi*3/2:
                uav.state.pos[0] += distance * math.cos(direction-math.pi)
                uav.state.pos[1] -= distance * math.sin(direction-math.pi)
            else:
                uav.state.pos[0] -= distance * math.cos(2*math.pi - direction)
                uav.state.pos[1] -= distance * math.sin(2*math.pi - direction)
            if uav.state.pos[0] < 0 or uav.state.pos[0] > 1000 or uav.state.pos[1] < 0 or uav.state.pos[1] > 1000:
                uav.state.pos = pos_temp
                # 更新能耗，此时只有盘旋能耗
                print('剩余能量{}'.format(uav.state.energy))
                print('消耗能量{}'.format(P_h * t))
                uav.state.energy -= P_h * t
            else:
                # 更新能耗
                print('剩余能量{}'.format(uav.state.energy))
                print('消耗能量{}'.format(P_f * (uav.action.distance * uav.max_distance / V) + P_h * (t - uav.action.distance * uav.max_distance / V)))
                uav.state.energy -= P_f * (uav.action.distance * uav.max_distance / V) + P_h * (t - uav.action.distance * uav.max_distance / V)
            # 更新用户关联 sorted返回的是排序好的副本
            landmarks_order = sorted(world.landmarks.values(), key=lambda mark: np.sum(
                np.square(uav.state.pos - mark.state.pos)))  # 将landmark按距离排序
            for landmark in landmarks_order:
                if uav.state.curServiceNum < uav.maxServiceNum:
                    if landmark.connected is False and np.sqrt(np.sum((landmark.state.pos - uav.state.pos)**2)) <= 100:
                        world.landmarks[str(landmark.id)].connected = True
                        world.landmarks[str(landmark.id)].sum_throughput += self.get_capacity(uav, landmark)
                        uav.state.curServiceNum += 1
                        uav.associator.append(landmark.id)
            # 更新landmark的平均数据率和公平比例权重
            for landmark in world.landmarks.values():
                landmark.avg_dataRate = landmark.sum_throughput / world.t  # 考虑是否要乘上t（时隙长度）
                landmark.weight = R_eq / (R_eq + landmark.avg_dataRate)

    def get_done(self, world):
        for uav in world.UAVs:
            if uav.state.energy <= 0:
                return True
        return False

    def reset_service(self, world):
        for uav in world.UAVs:
            uav.state.curServiceNum = 0
            uav.associator = []
        for landmark in world.landmarks.values():
            landmark.connected = False

    # 计算当前数据传输率
    def get_sum_capacity(self, world):
        capacity_list = []
        for uav in world.UAVs:
            capacity = 0
            for id in uav.associator:
                landmark = world.landmarks[str(id)]
                probability_los = self.get_probability(uav.state.pos, landmark.state.pos)  # 获得LoS概率
                # print("建立LoS的概率为{:.4f}".format(probability_los))
                pathLoss = self.get_passLoss(uav.state.pos, landmark.state.pos, probability_los)  # 获得平均路径损失
                capacity += (Bandwidth / K) * math.log(1 + P_tr * (1/pathLoss) / sigma_power, 2)  # 根据香农公式计算信道容量
            capacity_list.append(capacity)
        return capacity_list

    # 计算某个UAV和地面设备之间的数据速率
    def get_capacity(self, uav, landmark):
        probability_los = self.get_probability(uav.state.pos, landmark.state.pos)  # 获得LoS概率
        pathLoss = self.get_passLoss(uav.state.pos, landmark.state.pos, probability_los)  # 获得平均路径损失
        capacity = (Bandwidth / K) * math.log(1 + P_tr * (1 / pathLoss) / sigma_power, 2)  # 根据香农公式计算信道容量
        return capacity

    def get_probability(self, uav_pos, landmark_pos):
        r = np.sqrt(np.sum((landmark_pos - uav_pos)**2))
        eta = 0
        if r == 0:
            eta = math.pi/2  # 单位是°
        else:
            eta = (180/math.pi) * np.arctan(H/r)
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
        distance = np.sqrt(np.sum((landmark_pos - uav_pos)**2) + H**2)
        return distance

if __name__ == '__main__':
    sc = Scenario()
    a = np.array([816.21, 531.57])
    b = np.array([752.02, 523.63])
    probability = sc.get_probability(a, b)
    print(probability)
