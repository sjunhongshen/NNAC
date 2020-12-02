import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Critic, self).__init__()

        self.input_layer = nn.Linear(state_dim + action_dim, config.layer_info[0][0])
        self.layers = nn.ModuleList()
        for n_in, n_out in config.layer_info:
            self.layers.append(nn.Linear(n_in, n_out))
        self.output_layer = nn.Linear(config.layer_info[-1][1], 1)
        self.output_layer.weight.data.uniform_(-config.init_w, config.init_w)
        self.output_layer.bias.data.uniform_(-config.init_w, config.init_w)

    def forward(self, state, action):
        q = F.relu(self.input_layer(torch.cat([state, action], 1)))
        for layer in self.layers:
            q = F.relu(layer(q))
        return self.output_layer(q)


class Critic2(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(Critic2, self).__init__()

        self.l1 = nn.Linear(state_dim, config.layer_info[0][0])
        self.l2 = nn.Linear(config.layer_info[0][0] + action_dim, config.layer_info[0][1])
        self.l3 = nn.Linear(config.layer_info[0][1], 1)

    def forward(self, state, action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        return self.l3(q)