import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), change_done=True):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.change_done = change_done


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        if self.change_done:
            self.not_done[self.ptr] = 1. - done
        else:
            self.not_done[self.ptr] = done

        self.ptr = int((self.ptr + 1) % self.max_size)
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size, consec = False):
        if not consec:
            ind = np.random.randint(0, self.size, size = batch_size)
        else:
            ind = np.random.randint(0, self.size, size = 1)[0]
            ind = np.arange(ind, ind + batch_size) % self.max_size

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
	        torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
            )