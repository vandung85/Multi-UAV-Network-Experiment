import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class actor_agent(nn.Module):
    def __init__(self, num_inputs, num_hidden_1, num_hidden_2, action_size):
        super(actor_agent, self).__init__()
        self.linear_1 = nn.Linear(num_inputs, num_hidden_1)
        self.linear_2 = nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_3 = nn.Linear(num_hidden_2, action_size)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        self.train()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.LeakyReLU(x)
        x = self.linear_2(x)
        x = self.LeakyReLU(x)
        x = self.linear_3(x)
        policy = self.tanh(x)
        return policy

class critic_agent(nn.Module):
    def __init__(self, obs_shape_n, num_hidden_1, num_hidden_2, action_space_n, ):
        super(critic_agent, self).__init__()
        self.linear_o_c1 = nn.Linear(obs_shape_n, num_hidden_1)
        self.linear_a_c1 = nn.Linear(action_space_n, num_hidden_1)
        self.linear_c2 = nn.Linear(num_hidden_1*2, num_hidden_2)
        self.linear_c = nn.Linear(num_hidden_2, 1)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()

    def forward(self, obs_input, action_input):
        x_o = self.LeakyReLU(self.linear_o_c1(obs_input))
        x_a = self.LeakyReLU(self.linear_a_c1(action_input))
        x = torch.cat([x_o, x_a], dim=1)  # 不懂
        value = self.linear_c(x)
        return value
