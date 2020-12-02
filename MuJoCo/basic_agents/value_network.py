import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super(ValueNetwork, self).__init__()

        self.input_layer = nn.Linear(state_dim, config.layer_info[0][0])
        self.layers = nn.ModuleList()
        for n_in, n_out in config.layer_info:
            self.layers.append(nn.Linear(n_in, n_out))
        self.output_layer = nn.Linear(config.layer_info[-1][1], 1)
        self.output_layer.weight.data.uniform_(-config.init_w, config.init_w)
        self.output_layer.bias.data.uniform_(-config.init_w, config.init_w)

    def forward(self, state):
        q = F.relu(self.input_layer(state))
        for layer in self.layers:
            q = F.relu(layer(q))
        return self.output_layer(q)
