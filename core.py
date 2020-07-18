import numpy as np


class State:
    def __init__(self):
        self.pos = None


class LandmarkState(State):
    def __init__(self):
        super(LandmarkState, self).__init__()


class UAVState(State):
    def __init__(self):
        super(UAVState, self).__init__()
        self.energy = None
        self.curServiceNum = 0
        self.servicedUsers = []


class Action:
    def __init__(self):
        self.direction = None
        self.distance = None


class Entity:
    def __init__(self):
        self.name = ''
        self.type = None
        self.id = None
        self.size = 5
        self.movable = False
        self.color = None
        self.state = None


class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
        self.type = 'Landmark'
        self.state = LandmarkState()
        self.avg_dataRate = 0  # 当前平均数据率
        self.connected = False  # 是否已建立UAV连接
        self.weight = 1 # 时变权重
        self.sum_throughput = 0


class UAV(Entity):
    def __init__(self):
        super(UAV, self).__init__()
        self.movable = True
        self.state = UAVState()
        self.action = Action()
        self.height = 50
        self.max_distance = 200
        self.coverage = 100
        self.maxServiceNum = 5
        self.associator = [] # 关联服务用户列表


class World:
    def __init__(self):
        self.UAVs = []
        # self.landmarks = []
        self.landmarks = {}
        self.dim_p = 2
        self.dt = 0.1  # simulation timestep
        self.length = 1000
        self.width = 1000
        self.t = 0 # 时隙
        self.num_UAVs = 0
        self.num_landmarks = 0

    @property  # 装饰器，将函数改为可以直接调用的变量
    def entities(self):
        return self.UAVs + self.landmarks

