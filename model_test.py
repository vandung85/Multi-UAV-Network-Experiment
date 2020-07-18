import os
import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ReplayBuffer import ReplayBuffer
from model import actor_agent, critic_agent, openai_actor, openai_critic
from MultiUAV_environment import MultiUAVEnv
from MultiUAV_scenario import Scenario as sc

per_episode_max_len = 200
model_name = 'models/2UAVs_25000/'
def get_trainers(env):
    actors_tar = [torch.load(model_name + 'a_c{}.pt'.format(agent_idx)) for agent_idx in range(env.world.num_UAVs)]
    return actors_tar

def get_initial_trainer(env):
    actors_tar = [None for _ in range(env.world.num_UAVs)]
    num_hidden_1 = 256
    num_hidden_2 = 64
    obs_shape = 54
    action_shape = 2
    for i in range(env.world.num_UAVs):
        actors_tar[i] = actor_agent(obs_shape, num_hidden_1, num_hidden_2, action_shape)
    return actors_tar

# 用于检测训练好的模型
def model_test():
    episode_step = 0

    # 初始化环境
    scenario = sc()
    env = MultiUAVEnv(scenario)

    # 初始化智能体
    # actors_tar = get_trainers(env)
    actors_tar = get_initial_trainer(env)

    obs_n = env.reset()

    while True:
        episode_step += 1

        try:
            action_n = []
            # action_n = [agent.actor(torch.from_numpy(obs).to(arglist.device, torch.float)).numpy() \
            # for agent, obs in zip(trainers_cur, obs_n)]
            for actor, obs in zip(actors_tar, obs_n):
                # action = torch.clamp(actor(torch.from_numpy(obs).float()), -1, 1)
                action = actor(torch.from_numpy(obs).float())
                action_n.append(action)
            print(action_n)
        except:
            print(obs_n)

        new_obs_n, rew_n, done_n, _ = env.step(action_n)
        # update the flag
        done = done_n
        terminal = (episode_step >= per_episode_max_len)
        obs_n = new_obs_n

        # reset the env
        if done or terminal:
            episode_step = 0
            obs_n = env.reset()

        # render the env
        # print(rew_n)
        # print(new_obs_n)
        env.render()
        time.sleep(3)

if __name__ == '__main__':
    model_test()