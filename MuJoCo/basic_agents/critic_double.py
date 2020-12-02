import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class Critic_Double(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Critic_Double, self).__init__()

        self.input_layer = nn.Linear(state_dim + action_dim, config.layer_info[0][0])
        self.layers = nn.ModuleList()
        for n_in, n_out in config.layer_info:
            self.layers.append(nn.Linear(n_in, n_out))
        self.output_layer = nn.Linear(config.layer_info[-1][1], 1)
        self.output_layer.weight.data.uniform_(-config.init_w, config.init_w)
        self.output_layer.bias.data.uniform_(-config.init_w, config.init_w)

        self.input_layer_2 = nn.Linear(state_dim + action_dim, config.layer_info[0][0])
        self.layers_2 = nn.ModuleList()
        for n_in, n_out in config.layer_info:
            self.layers_2.append(nn.Linear(n_in, n_out))
        self.output_layer_2 = nn.Linear(config.layer_info[-1][1], 1)
        self.output_layer_2.weight.data.uniform_(-config.init_w, config.init_w)
        self.output_layer_2.bias.data.uniform_(-config.init_w, config.init_w)


    def forward(self, state, action):
        q1 = F.relu(self.input_layer(torch.cat([state, action], 1)))
        for layer in self.layers:
            q1 = F.relu(layer(q1))
        q1 = self.output_layer(q1)

        q2 = F.relu(self.input_layer_2(torch.cat([state, action], 1)))
        for layer in self.layers_2:
            q2 = F.relu(layer(q2))
        q2 = self.output_layer_2(q2)

        return q1, q2

    def Q1(self, state, action):
        q1 = F.relu(self.input_layer(torch.cat([state, action], 1)))
        for layer in self.layers:
            q1 = F.relu(layer(q1))
        q1 = self.output_layer(q1)

        return q1