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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.LeakyReLU(x)
        x = self.linear_2(x)
        x = self.LeakyReLU(x)
        x = self.linear_3(x)
        policy = self.sigmoid(x)
        return policy

class critic_agent(nn.Module):
    def __init__(self, obs_shape_n, num_hidden_1, num_hidden_2, action_space_n):
        super(critic_agent, self).__init__()
        self.linear_o_c1 = nn.Linear(obs_shape_n, num_hidden_1)
        self.linear_a_c1 = nn.Linear(action_space_n, num_hidden_1)
        self.linear_c2 = nn.Linear(num_hidden_1*2, num_hidden_2)
        self.linear_c = nn.Linear(num_hidden_2, 1)
        self.LeakyReLU = nn.LeakyReLU(0.1)

    def forward(self, obs_input, action_input):
        x_o = self.LeakyReLU(self.linear_o_c1(obs_input))
        x_a = self.LeakyReLU(self.linear_a_c1(action_input))
        x = torch.cat([x_o, x_a], dim=1)  # 不懂
        x = self.LeakyReLU(self.linear_c2(x))
        value = self.linear_c(x)
        return value


class openai_critic(nn.Module):
    def __init__(self, obs_shape_n, num_hidden_1, num_hidden_2, action_space_n):
        super(openai_critic,self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_space_n+obs_shape_n, num_hidden_1)
        self.linear_c2 = nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_c3 = nn.Linear(num_hidden_2, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c3.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c3(x)
        return value

class openai_actor(nn.Module):
    def __init__(self, num_inputs, num_hidden_1, num_hidden_2, action_size):
        super(openai_actor, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, num_hidden_1)
        self.linear_a2 = nn.Linear(num_hidden_1, num_hidden_2)
        self.linear_a3 = nn.Linear(num_hidden_2, action_size)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a3.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, input, model_original_out=False):
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a3(x)
        u = torch.rand_like(model_out)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out == True:   return model_out, policy # for model_out criterion
        return policy




