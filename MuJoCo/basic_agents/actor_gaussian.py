import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class Actor_Gaussian(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, config):
        super(Actor_Gaussian, self).__init__()
        
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

        self.input_layer = nn.Linear(state_dim, config.layer_info[0][0])
        self.layers = nn.ModuleList()
        for n_in, n_out in config.layer_info:
            self.layers.append(nn.Linear(n_in, n_out))

        self.output_layer_mean = nn.Linear(config.layer_info[-1][1], action_dim)
        self.output_layer_mean.weight.data.uniform_(-config.init_w, config.init_w)
        self.output_layer_mean.bias.data.uniform_(-config.init_w, config.init_w)

        self.output_layer_std = nn.Linear(config.layer_info[-1][1], action_dim)
        self.output_layer_std.weight.data.uniform_(-config.init_w, config.init_w)
        self.output_layer_std.bias.data.uniform_(-config.init_w, config.init_w)
        
        self.max_action = max_action
        self.config = config

        
    def forward(self, state):
        a = F.relu(self.input_layer(state))

        for layer in self.layers:
            a = F.relu(layer(a))

        return self.max_action * torch.tanh(self.output_layer_mean(a)), torch.clamp(self.output_layer_std(a), self.log_std_min, self.log_std_max)