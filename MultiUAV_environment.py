import gym
from gym import spaces
import numpy as np
from MultiUAV_scenario import Scenario as sc
import time
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)


class MultiUAVEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, scenario):
        self.sc = scenario # 场景，相当于简单工厂，用于制造world，和处理world的一些函数
        world = self.sc.make_world()
        self.world = world
        self.time = 0
        self.action_space = []
        self.observation_space = []
        for uav in self.world.UAVs:
            # act_space
            total_action_space = []
            action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32) # 动作空间，值域0到1，维度2(方向和距离)
            self.action_space.append(action_space)
            # obs_space
            obs_dim = len(self.sc.observation(self.world, uav))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32)) # 观察空间
        # render
        self.viewer = None
        self._reset_render()

    def step(self, action_n):
        info = {}
        obs_n = []
        for i, uav in enumerate(self.world.UAVs):
            self._set_action(action_n[i], uav, self.action_space[i])
        # 更新状态
        self.sc.step(self.world)
        # 记录观察
        for uav in self.world.UAVs:
            obs_n.append(self._get_obs(uav))
        # 查看当前环境是否因越界或能量耗尽而结束
        done = sc.get_done(self.world)
        reward = self.sc.reward(self.world) # 由于是协作关系，共享一个reward
        return obs_n, reward, done, info

    def _get_obs(self, uav):
        return self.sc.observation(self.world, uav)

    def _get_reward(self, uav):
        return self.sc.reward(self.world, uav)

    def _get_done(self, uav):
        return self.sc.reward(self.world, uav)

    def _set_action(self, action, uav, action_space, time=None):
        uav.action.direction = action[0]
        uav.action.distance = action[1]

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        # screen_width = 1000
        # screen_height = 1000
        # # 如果没有viewer，创建viewer和uav、landmarks
        # if self.viewer is None:
        #     self.viewer = rendering.Viewer(screen_height, screen_width)
        #     # 保存service关联直线
        #     self.services_count = 0
        #     self.render_landmarks = []
        #     self.render_landmarks_tranform = []
        #     for landmark in self.world.landmarks.values():
        #         geom = rendering.make_circle(landmark.size)
        #         geom_transform = rendering.Transform(translation=(landmark.state.pos[0], landmark.state.pos[1]))
        #         geom.set_color(0, 1, 0)
        #         geom.add_attr(geom_transform)
        #         self.render_landmarks.append(geom)
        #         self.render_landmarks_tranform.append(geom_transform)
        #         self.viewer.add_geom(geom)
        #     self.render_geom = []
        #     self.render_tranform = []
        #     for uav in self.world.UAVs:
        #         geom = rendering.make_circle(uav.size)
        #         form = rendering.Transform()
        #         geom.set_color(1, 0, 0)
        #         geom.add_attr(form)
        #         self.render_geom.append(geom)
        #         self.render_tranform.append(form)
        #         self.viewer.add_geom(geom)
        # self.viewer.set_bounds(0, 1000, 0, 1000)
        # # 刷新landmark位置
        # for i, landmark in enumerate(self.world.landmarks.values()):
        #     self.render_landmarks_tranform[i].set_translation(*landmark.state.pos)
        # # 刷新uav位置
        # for i, uav in enumerate(self.world.UAVs):
        #     self.render_tranform[i].set_translation(*uav.state.pos)
        # # 消除上个时间片的用户服务关联
        # for i in range(self.services_count):
        #     self.viewer.geoms.pop(-1-i)
        # self.services_count = 0
        # # 刷新uav和landmark之间的关联
        # for uav in self.world.UAVs:
        #     print(uav.associator)
        #     for i in uav.associator:
        #         line = rendering.Line(uav.state.pos, self.world.landmarks[str(i)].state.pos)
        #         self.viewer.add_geom(line)
        #         self.services_count += 1

        # 新代码 不适用tranform，直接刷新全局组件
        screen_width = 1000
        screen_height = 1000
        # 如果没有viewer，创建viewer和uav、landmarks
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_height, screen_width)
        self.viewer.set_bounds(0, 1000, 0, 1000)
        self.viewer.geoms.clear()
        for uav in self.world.UAVs:
            geom = rendering.make_circle(uav.size)
            geom.set_color(1, 0, 0)
            geom_form = rendering.Transform(translation=(uav.state.pos[0], uav.state.pos[1]))
            geom.add_attr(geom_form)
            self.viewer.add_geom(geom)
        for landmark in self.world.landmarks.values():
            geom = rendering.make_circle(landmark.size)
            geom.set_color(0, 1, 0)
            geom_transform = rendering.Transform(translation=(landmark.state.pos[0], landmark.state.pos[1]))
            geom.add_attr(geom_transform)
            self.viewer.add_geom(geom)
        for uav in self.world.UAVs:
            for i in uav.associator:
                line = rendering.Line(uav.state.pos, self.world.landmarks[str(i)].state.pos)
                self.viewer.add_geom(line)

        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def reset(self):
        self.sc.reset_world(self.world)
        self._reset_render()
        obs_n = []
        for uav in self.world.UAVs:
            obs_n.append(self.sc.observation(self.world, uav))
        return obs_n

    def _reset_render(self):
        self.render_geom = None
        self.render_transform = None

    def random_action(self):
        action_n = []  # 随机联合动作
        for action in self.action_space:
            action_n.append(action.sample())
        return action_n

if __name__ == '__main__':
    sc = sc()
    env = MultiUAVEnv(sc)
    env.render()
    for i in range(len(env.world.landmarks.values())):
        print("地标位置：")
        print(env.world.landmarks[str(i)].state.pos)
    while True:
        print('----------------------')
        action_n = env.random_action()
        o, r, done, _ = env.step(action_n)
        if done:
            break
        for uav in env.world.UAVs:
            print("当前无人机{}位置：".format(uav.id))
            print(uav.state.pos)
            print("当前无人机关联用户：")
            print(uav.associator)
            for i in uav.associator:
                print(env.world.landmarks[str(i)].state.pos)

        print(o)
        print("当前系统的总吞吐量：{}".format(r))
        env.render()
        time.sleep(5)
    # sc = sc()
    # env = MultiUAVEnv(sc)
    # env.reset()
    # capacity = 0
    # energy = []
    # capacity_list = []
    # while True:
    #     action_n = env.random_action()
    #     print(action_n)
    #     o, r, done, _ = env.step(action_n)
    #     capacity += r
    #     if done:
    #         break
    # for uav in env.world.UAVs:
    #     energy.append(uav.state.energy)
    # for landmark in env.world.landmarks.values():
    #     capacity_list.append(landmark.sum_throughput)
    # sum_capacity = np.sum(np.array(capacity_list))
    # print('执行回合数:{}'.format(env.world.t))
    # print('总吞吐量:{}'.format(sum_capacity))
    # print('剩余能量：')
    # print(energy)
    # x = [i for i in range(env.world.num_landmarks)]
    # plt.figure()
    # plt.plot(x, capacity_list)
    # plt.xlabel('user_id')
    # plt.ylabel('data_volume')
    # plt.show()