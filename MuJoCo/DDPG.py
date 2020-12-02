import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import copy

from basic_agents.actor import Actor
from basic_agents.critic import Critic2
from basic_agents.replay_buffer import ReplayBuffer

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, env, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action, config).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = config.lr_actor)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size = config.decay_step, gamma = config.decay_rate)

        self.critic = Critic2(state_dim, action_dim, config).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = config.lr_critic, weight_decay = 1e-3)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size = config.decay_step, gamma = config.decay_rate)
        
        self.config = config
        self.discount = config.discount
        self.tau = config.tau
        self.env = env

        self.replay_buffer = ReplayBuffer(state_dim, action_dim)

    def select_action(self, state, noise = 0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise != 0:
            action = (action + np.random.normal(0, noise, size = self.env.action_space.shape[0]))
        return action.clip(self.env.action_space.low, self.env.action_space.high)

    def train(self, episode_timesteps, batch_size, current_episode):
        for it in range(episode_timesteps):
            state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (not_done * self.discount * target_Q).detach()
            current_Q = self.critic(state, action)

            critic_loss = F.mse_loss(current_Q, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(state, self.actor(state)).mean()
        
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)

    def save(self, directory):
        torch.save(self.critic.state_dict(), '%s/critic.pth' % (directory))
        torch.save(self.critic_optimizer.state_dict(), '%s/critic_optimizer.pth' % (directory))
        torch.save(self.actor.state_dict(), '%s/actor.pth' % (directory))
        torch.save(self.actor_optimizer.state_dict(), '%s/actor_optimizer.pth' % (directory))

    def load(self, directory):
        self.critic.load_state_dict(torch.load('%s/critic.pth' % (directory)))
        self.critic_optimizer.load_state_dict(torch.load('%s/critic_optimizer.pth' % (directory)))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load('%s/actor.pth' % (directory)))
        self.actor_optimizer.load_state_dict(torch.load('%s/actor_optimizer.pth' % (directory)))
        self.actor_target = copy.deepcopy(self.actor)