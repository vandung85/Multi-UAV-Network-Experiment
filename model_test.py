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
import torch.nn.init as init
import matplotlib.pyplot as plt


per_episode_max_len = 200
model_name = 'models/2UAVs_24012/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def weigth_init(m):
   if isinstance(m, nn.Conv2d):
       init.xavier_uniform_(m.weight.data)
       init.constant_(m.bias.data,0.1)
   elif isinstance(m, nn.BatchNorm2d):
       m.weight.data.fill_(1)
       m.bias.data.zero_()
   elif isinstance(m, nn.Linear):
       m.weight.data.normal_(0, 0.01)
       m.bias.data.zero_()

def get_trainers(env):
    actors_tar = [torch.load(model_name + 'a_c{}.pt'.format(agent_idx), map_location='cpu') for agent_idx in range(env.world.num_UAVs)]
    return actors_tar

def get_initial_trainer(env):
    actors_tar = [None for _ in range(env.world.num_UAVs)]
    num_hidden_1 = 50
    num_hidden_2 = 20
    obs_shape = 34
    action_shape = 2
    for i in range(env.world.num_UAVs):
        actors_tar[i] = actor_agent(obs_shape, num_hidden_1, num_hidden_2, action_shape)
        # actors_tar[i].apply(weigth_init)
        # actors_tar[i].eval()
    return actors_tar

# 用于检测训练好的模型
def model_test():
    episode_step = 0
    scenario = sc()
    env = MultiUAVEnv(scenario)
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.world.num_UAVs)]

    # 初始化环境

    # 初始化智能体
    actors_tar = get_trainers(env)
    # actors_tar = get_initial_trainer(env)

    obs_n = env.reset()

    while True:
        episode_step += 1


        action_n = []
        # action_n = [agent.actor(torch.from_numpy(obs).to(arglist.device, torch.float)).numpy() \
        # for agent, obs in zip(trainers_cur, obs_n)]
        for actor, obs in zip(actors_tar, obs_n):
            # action = torch.clamp(actor(torch.from_numpy(obs).float()), -1, 1)
            action = actor(torch.from_numpy(obs).float())
            action_n.append(action)
        print(action_n)

        new_obs_n, rew_n, done_n, _ = env.step(action_n)
        print(new_obs_n)
        # update the flag
        done = False
        if True in done_n:
            done = True

        terminal = (episode_step >= per_episode_max_len)
        obs_n = new_obs_n

        # reset the env
        if done or terminal:
            print(episode_step)
            # episode_step = 0
            # obs_n = env.reset()
            volumn = []
            for landmark in env.world.landmarks.values():
                volumn.append(landmark.sum_throughput)
            x = [i for i in range(30)]
            plt.plot(x, volumn)
            plt.show()
            env.close()
            break


        # render the env
        # print(rew_n)
        # print(new_obs_n)
        env.render()
        # time.sleep(0.5)

    # game_step = 0
    # for episode_gone in range(1000):
    #
    #     while True:
    #         game_step += 1
    #         # get action
    #         action_n = []
    #         for actor, obs in zip(actors_tar, obs_n):
    #             action = actor(torch.from_numpy(obs).float())
    #             action_n.append(action)
    #         # 和环境交互
    #         # print(game_step)
    #         # print(action_n)
    #         new_obs_n, reward, done, info_n = env.step(action_n)
    #         # print(new_obs_n)
    #         # 保存到经验池
    #         episode_rewards[-1] += np.sum(reward)
    #         for i, rew in enumerate(reward):
    #             agent_rewards[i][-1] += reward[i]
    #         if done:
    #             obs_n = env.reset()
    #             for r in agent_rewards:
    #                 r.append(0)
    #             episode_rewards.append(0)
    #             break
    #
    #     print('=Training: steps:{} episode:{}'.format(game_step, episode_gone))
    #     mean_ep_r = round(episode_rewards[-2], 3)
    #     mean_agents_r = [round(agent_rewards[idx][-2], 2) for idx in range(env.world.num_UAVs)]
    #     print(" " * 43 + 'episode reward:{}    agent_rewards:{}'.format(mean_ep_r, mean_agents_r))

if __name__ == '__main__':
    model_test()