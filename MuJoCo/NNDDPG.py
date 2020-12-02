import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.spatial import cKDTree
import copy
from basic_agents.actor import Actor
from basic_agents.critic import Critic2
from basic_agents.replay_buffer import ReplayBuffer
from basic_agents.critic_nearest_neighbor import Critic_NearestNeighbor
from utils import v_wrap

class NNDDPG(object):    
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

        self.critic_nn = Critic_NearestNeighbor(self, config)

        self.config = config
        self.discount = config.discount
        self.tau = config.tau
        self.a1 = config.agent_param
        self.a2 = config.nn_param_actor
        self.distribution = torch.distributions.Normal
        self.env = env

        self.replay_buffer = ReplayBuffer(state_dim, action_dim)
        self.replay_buffer_td = ReplayBuffer(state_dim, action_dim, int(1e6), False)

    def replay_td(self):
        if self.replay_buffer_td.size == 0:
            return
        for _ in range(int(min(self.config.td_batch_size, len(self.replay_buffer_td.state)) / self.config.mini_batch)):
            state, action, next_state, reward, td_error  = self.replay_buffer_td.sample(self.config.mini_batch)
            nn_loss = self.distribution(self.actor(state), self.config.policy_noise).log_prob(action).mean(1, True)
            actor_loss = - (self.a2 * (td_error.detach() * nn_loss)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for lp in self.actor.parameters():
                lp.grad = lp.grad.clamp(-self.config.grad_clip, self.config.grad_clip)

            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (self.discount * target_Q).detach()
            current_Q = self.critic(state, action)
            critic_loss = (self.a2 * F.mse_loss(target_Q - current_Q, td_error.detach())).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for lp in self.critic.parameters():
                lp.grad = lp.grad.clamp(-self.config.grad_clip, self.config.grad_clip)

    def select_action(self, state, noise = 0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise != 0:
            action = (action + np.random.normal(0, noise, size = self.env.action_space.shape[0]))
        return action.clip(self.env.action_space.low, self.env.action_space.high)

    def train(self, episode_timesteps, batch_size, current_episode):
        if current_episode < self.config.turning_point:
            self.tau = self.config.tau_nn
            self.a2 = self.config.nn_param_actor
            self.critic_nn.tree = cKDTree(self.critic_nn.states_all)
            episode_timesteps = 1
        else:
            self.tau = self.config.tau
            self.a2 = self.config.eps
 
        for it in range(episode_timesteps):
            if self.a2 == self.config.eps:
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

                if it == episode_timesteps - 1:
                    self.replay_td()

            else:
                for _ in range(int(batch_size / self.config.mini_batch)):
                    state, action, next_state, reward, not_done = self.replay_buffer.sample(self.config.mini_batch, True)
                    state_numpy, action_numpy, reward_numpy, next_state_numpy = state.clone().detach().cpu().data.numpy(), action.clone().detach().cpu().data.numpy(), reward.clone().detach().cpu().data.numpy(), next_state.clone().detach().cpu().data.numpy()
                    nn_loss = self.distribution(self.actor(state), self.config.policy_noise_dist).log_prob(action).mean(1, True)

                    td_errors = []
                    for i in range(self.config.mini_batch):
                        td_errors.append(self.critic_nn.learn(state_numpy[i], action_numpy[i], reward_numpy[i], next_state_numpy[i], 0))
                        if td_errors[-1] > 0:
                            self.replay_buffer_td.add(state_numpy[i], action_numpy[i], next_state_numpy[i], reward_numpy[i], td_errors[-1])

                    td_errors = v_wrap(np.vstack(td_errors))
                    actor_loss = - (self.a1 * self.critic(state, self.actor(state)) + self.a2 * (td_errors.detach() * nn_loss)).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    target_Q = self.critic_target(next_state, self.actor_target(next_state))
                    target_Q = reward + (not_done * self.discount * target_Q).detach()
                    current_Q = self.critic(state, action)

                    critic_loss = self.a1 * F.mse_loss(current_Q, target_Q) + self.config.nn_param_critic * F.mse_loss(target_Q - current_Q, td_errors.detach())

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                self.replay_td()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)
        self.critic_nn.remember(state, action, reward, next_state, 0, done)

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