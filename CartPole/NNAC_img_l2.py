import math
import random

import gym, os, copy, cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scipy.spatial import cKDTree
from scipy.stats import ortho_group
from easydict import EasyDict as edict
import matplotlib.pyplot as plt


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, config):
        super(Actor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encode_layer = nn.Conv3d(1, config.encode_dim, (4, 11, 11), stride = (1, 2, 2), padding = (0, 5, 5))
        self.input_layer = nn.Linear(config.encode_dim * 100, config.layer_info[0][0])
        self.output_layer = nn.Linear(config.layer_info[-1][-1], action_dim)
        
        if config.init_w is not None:
            self.encode_layer.weight.data.uniform_(-config.init_w, config.init_w)
            self.encode_layer.bias.data.uniform_(-config.init_w, config.init_w)
            self.output_layer.weight.data.uniform_(-config.init_w, config.init_w)
            self.output_layer.bias.data.uniform_(-config.init_w, config.init_w)

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

    def forward(self, state):
        state = self.encode_layer(state)
        state = state.reshape(1, -1)
        a = F.relu(self.input_layer(state))
        return F.softmax(torch.tanh(self.output_layer(a)), dim = 1)

    def select_action(self, state, greedy = False):
        state = torch.FloatTensor(state.reshape(-1, 1, 4, self.config.row, self.config.col)).to(self.device)
        action = self.forward(state).cpu().detach().numpy().flatten()

        if greedy:
            return np.argmax(action)
        action = np.random.choice(self.action_dim, p = action.ravel())
        return action


class Critic_NearestNeighbor:
    def __init__(self, actor, config):
        self.actor = actor
        self.config = config

        # parameters to tune
        self.horizon = config.planning_horizon
        self.L = config.L
        self.gamma = config.gamma

        self.max_size = config.buf_max_size
        self.states_all = []
        self.storage = []
        self.storage_ = []
        self.td_storage = []
        self.ptr = 0
        self.weights = np.array([1 / 4 for i in range(4)] + [1])

        self.K_neighbors = config.K_neighbors
        self.tree = None

    def remember(self, state, action, reward, next_state, step, terminal, state_, next_state_):
        if len(self.storage) == self.max_size:
            self.states_all[self.ptr] = np.concatenate((state.flatten(), action))
            self.storage[self.ptr] = (state, action, self.config.r_scale * reward, next_state, step, terminal)
            self.storage_[self.ptr] = (state_, next_state_)
            self.ptr = int((self.ptr + 1) % self.max_size)
        else:
            self.states_all.append(np.concatenate((state.flatten(), action)))
            self.storage.append((state, action, self.config.r_scale * reward, next_state, step, terminal))
            self.storage_.append((state_, next_state_))
  
    def estimate(self, step, s,  s_, a = None):
        if step == self.horizon:
            return 0

        if a is None:
            a = self.actor.select_action(s_)
            a = np.reshape(a, (1))

        distances, indices = self.tree.query(self.weights * np.concatenate((s.flatten(), a)), k = self.K_neighbors, n_jobs = -1)
        nearest_neighbors = self.storage[indices]
        nearest_neighbors_ = self.storage_[indices]
        
        vals = []
        for i in range(self.K_neighbors):
            nn = nearest_neighbors[i] if self.K_neighbors > 1 else nearest_neighbors
            nn_ = nearest_neighbors_[i] if self.K_neighbors > 1 else nearest_neighbors_
            d = distances[i] if self.K_neighbors > 1 else distances
            if nn[-1]:
                vals.append(nn[2] + self.L * d)
            else:
                vals.append(nn[2] + self.L * d + self.gamma * self.estimate(step + 1, nn[3], nn_[1]))

        return np.min(vals)
        
    def learn(self, s, a, r, s_, step, s_full, s_next_full):
        step = 0
        v = self.estimate(step, s, s_full)
        v_ = self.estimate(step + 1, s_, s_next_full)
        td_error = r + self.gamma * v_ - v

        return td_error if td_error > 0 else td_error * self.config.neg_td_scale


class NNAC:
    def __init__(self, env, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.config = config
        self.actor = Actor(config.dim_after, env.action_space.n, None, config).to(self.device)
        self.critic = Critic_NearestNeighbor(self.actor, config)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = config.lr_actor, weight_decay = 1e-4)
        self.distribution = torch.distributions.Normal
        self.weights = np.array([1 / 4 for i in range(4)] + [1])

    def remember(self, s, a, r, s_, terminal, step, s_ori, s_ori_):
        s, s_full = self.transform(s)
        s_, s_next_full = self.transform(s_)
        self.critic.remember(s_ori, a, r, s_ori_, step, terminal, s_full, s_next_full)

    def select_action(self, s, greedy = False):
        s, s_full = self.transform(s)
        return self.actor.select_action(s_full, greedy)

    def learn(self, current_episode, episode_t, skip = False):
        if not skip:
            self.critic.tree = cKDTree(self.weights * self.critic.states_all)

        s, a, r, s_, step, terminal = self.critic.storage[-1]
        s_full, s_next_full = self.critic.storage_[-1]
        td_error_scale = self.critic.learn(s, a, r, s_, step, s_full, s_next_full)
        if self.config.td_decay:
            td_param = self.config.td_param * self.config.td_decay_rate ** int(current_episode / self.config.td_decay_step)
        else:
            td_param = self.config.td_param
        td_param *= (1 / (2 * math.floor(max(episode_t, 0) / 100) + 1))
        td_error_scale *= td_param

        self.critic.td_storage.append((s_full, a, r, s_next_full, step, td_error_scale))
            
        if len(self.critic.td_storage) <= self.config.batch_size:
            return
        start = np.random.randint(0, len(self.critic.td_storage) - self.config.batch_size, size = 1)[0]
        batch_indices = [-1]# + list(np.arange(start, start + self.config.batch_size))
        for i in batch_indices:
            s, a, r, s_, step, td = self.critic.td_storage[i]
            s = torch.FloatTensor(s.reshape(1, 1, 4, self.config.row, self.config.col)).to(self.device)
            a = torch.FloatTensor(a.reshape(1, -1)).to(self.device)
            td = torch.FloatTensor(td.reshape(1, -1)).to(self.device)

            if i == -1: td /= self.config.td_param
            actor_loss = -1 * torch.log(self.actor(s))[0, a.long()]
            actor_loss = (td.detach() * actor_loss).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()

            for lp in self.actor.parameters():
                lp.grad = lp.grad.clamp(-self.config.grad_clip, self.config.grad_clip)
    
            self.actor_optimizer.step()

    def transform(self, states):
        imgs = []
        for s in states:
            img = np.squeeze(np.array((s[:,:,0] - 255)))
            img = np.array(cv2.resize(img, (240, 160)), dtype = bool)
            shape = (self.config.col, self.config.row)

            pixels = img.flatten()[::-1] 
            ylen = img.shape[1]
            flag = False
            idx1 = idx2 = None
            for idx in range(len(pixels)):
                if pixels[idx]:
                    if not flag:
                        idx1 = idx
                    flag = True
                if flag and not pixels[idx]:
                    idx2 = idx
                    break
            middle = int((idx2 + idx1) / 2)
            x = int((len(pixels) - middle) / img.shape[1])
            y = (len(pixels) - middle) % img.shape[1]
            starty = max(0, y - 20)
            startx = max(0, x - 100)
            
            img = np.array(img[startx:startx + 40, starty:starty + 40], dtype = np.float32)
            img = np.array(cv2.resize(img, dsize = (shape[0], shape[1])), dtype = int)
            imgs.append(np.reshape(img, (self.config.col, self.config.row, 1)))
        return np.squeeze(imgs[-1]), np.concatenate(imgs, axis = 2)

    def save(self, env_name, agent_name, seed, current_episode):
        directory = os.path.join(env_name, 'Experiments', agent_name, 'Trial_%d' % seed)
        torch.save(self.actor.state_dict(), '%s/actor.pth' % (directory))
        torch.save(self.actor_optimizer.state_dict(), '%s/actor_optimizer.pth' % (directory))

    def load(self, env_name, agent_name, seed):
        directory = os.path.join(env_name, 'Experiments', agent_name, 'Trial_%d' % seed)
        self.actor.load_state_dict(torch.load('%s/actor.pth' % (directory)))
        self.actor_optimizer.load_state_dict(torch.load('%s/actor_optimizer.pth' % (directory)))
        self.actor_target = copy.deepcopy(self.actor)


def adaptive_lr(agent, max_ep_steps, episode_t, skip, solve):
    if episode_t > 350 and not skip:
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = 1e-4
        skip = True
    
    if solve == 3:
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = 1e-5
    if solve == 5:
        for param_group in agent.actor_optimizer.param_groups:
            param_group['lr'] = 1e-6
    return skip


def run_env(seed, agent_name, config):
    max_episodes = 10000
    max_steps = 50000
    max_ep_steps = 500
    env_name = 'CartPole-v1'
    render = False

    dim_after = 4

    if not os.path.exists(os.path.join(env_name, 'Experiments', agent_name, 'Trial_%d' % seed)):
        os.makedirs(os.path.join(env_name, 'Experiments', agent_name,'Trial_%d' % seed))

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    env._max_episode_steps = max_ep_steps
    env.seed(seed)

    action_dim = env.action_space.n
    agent = NNAC(env, config)
    
    scores, rewards = [], []
    timestep = 0
    solve = 0
    skip = False

    for i_episode in range(max_episodes):
        if timestep > max_steps:
            break
        episode_r, episode_t = 0, 0

        s_ori = env.reset()
        s = env.render(mode = 'rgb_array')
        states = [s, s, s, s]
        states_ = [s, s, s]

        while True:
            if render: env.render()

            a = agent.select_action(states, skip)
            s_ori_, r, done, _ = env.step(a)
            a = np.reshape(a, (1))
            states_.append(env.render(mode = 'rgb_array'))

            agent.remember(states, a, r, states_, done, episode_t, s_ori, s_ori_)

            episode_r += r
            episode_t += 1
            timestep += 1

            states = copy.deepcopy(states_)
            states_ = states_[1:]
            s_ori = s_ori_

            if solve <= 20:
                agent.learn(i_episode, episode_t, skip)

            if done or episode_t > max_ep_steps:
                if skip:
                    agent.critic.tree = cKDTree(agent.weights * agent.critic.states_all)            
                
                solve += 1 if episode_t == max_ep_steps else 0

                skip = adaptive_lr(agent, max_ep_steps, episode_t, skip, solve)

                rewards.append(episode_r)
                scores.append((timestep, episode_r))
                print("[Timestep %d] Episode %d: %f, Running reward: %f" % (timestep, i_episode, episode_r, np.mean(rewards[-20:])))

                if (i_episode + 1) % config.save_freq == 0:
                    agent.save(env_name, agent_name, seed, i_episode)
                    np.save(os.path.join(env_name, 'Experiments', agent_name, 'Trial_%d' % seed, 'train_scores.npy'), scores)

                break

    agent.save(env_name, agent_name, SEED, i_episode)
    np.save(os.path.join(ENV_NAME, 'Experiments', agent_name, 'Trial_%d' % seed, 'train_scores.npy'), scores)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    agent_name = "NNACIMG"

    config = edict({'lr_actor': 5e-4, 'layer_info': [(32, 32)], 'save_freq': 20,  'init_w': 3e-3, 
                    'buf_max_size': 1e5, 'planning_horizon': 12, 'L': 7, 'gamma': 0.99, 'K_neighbors': 1, 'r_scale': 1, 'grad_clip': 10,
                    'td_decay': False, 'td_decay_step': 10, 'td_decay_rate':0.95,'td_param': 1, 'neg_td_scale': 1, 'dim_after': 1600,
                    'row': 20, 'col': 20, 'encode_dim': 1, 'batch_size': 32})

    for i in range(5,8):
        run_env(i, agent_name, config)






