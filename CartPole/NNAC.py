import math
import random

import gym, os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scipy.spatial import cKDTree
from easydict import EasyDict as edict
from scipy.stats import ortho_group


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, config):
        super(Actor, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_layer = nn.Linear(state_dim, config.layer_info[0][0])

        self.output_layer = nn.Linear(config.layer_info[-1][-1], action_dim)
        if config.init_w is not None:
            self.output_layer.weight.data.uniform_(-config.init_w, config.init_w)
            self.output_layer.bias.data.uniform_(-config.init_w, config.init_w)

        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

    def forward(self, state):
        a = F.relu(self.input_layer(state))
        return F.softmax(torch.tanh(self.output_layer(a)), dim = 1)

    def select_action(self, state, greedy = False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
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
        self.td_storage = []
        self.ptr = 0

        self.K_neighbors = config.K_neighbors
        self.tree = None

    def remember(self, state, action, reward, next_state, step, terminal):
        if len(self.storage) == self.max_size:
            self.states_all[self.ptr] = np.concatenate((state, action))
            self.storage[self.ptr] = (state, action, self.config.r_scale * reward, next_state, step, terminal)
            self.ptr = int((self.ptr + 1) % self.max_size)
        else:
            self.states_all.append(np.concatenate((state, action)))
            self.storage.append((state, action, self.config.r_scale * reward, next_state, step, terminal))
  
    def estimate(self, step, s, a = None):
        if step == self.horizon:
            return 0

        if a is None:
            a = self.actor.select_action(s)
            a = np.reshape(a, (1))

        distances, indices = self.tree.query(self.config.weights * np.concatenate((s, a)), k = self.K_neighbors, n_jobs = -1)
        nearest_neighbors = self.storage[indices]
        
        vals = []
        for i in range(self.K_neighbors):
            nn = nearest_neighbors[i] if self.K_neighbors > 1 else nearest_neighbors
            d = distances[i] if self.K_neighbors > 1 else distances
            if nn[-1]:
                vals.append(nn[2] + self.L * d)
            else:
                vals.append(nn[2] + self.L * d + self.gamma * self.estimate(step + 1, nn[3]))

        return np.min(vals)
        
    def learn(self, s, a, r, s_, step):
        step = 0
        v = self.estimate(step, s)
        v_ = self.estimate(step + 1, s_)
        td_error = r + self.gamma * v_ - v

        return td_error if td_error > 0 else td_error * self.config.neg_td_scale


class NNAC:
    def __init__(self, env, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.config = config
        self.actor = Actor(env.observation_space.shape[0] if config.dim_after is None else config.dim_after, env.action_space.n, None, config).to(self.device)
        self.critic = Critic_NearestNeighbor(self.actor, config)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = config.lr_actor, weight_decay = 1e-4)
        self.distribution = torch.distributions.Normal

        if config.dim_after is not None:
            self.mat = ortho_group.rvs(dim = config.dim_after)[:, :env.observation_space.shape[0]].T

    def remember(self, s, a, r, s_, terminal, step):
        if self.config.dim_after is not None:
            s = self.transform(s)
            s_ = self.transform(s_)
        self.critic.remember(s, a, r, s_, step, terminal)

    def select_action(self, s, greedy = False):
        if self.config.dim_after is not None:
            s = self.transform(s)
        return self.actor.select_action(s, greedy)

    def learn(self, current_episode, episode_t, skip = False):
        if not skip:
            self.critic.tree = cKDTree(self.config.weights * self.critic.states_all)
   
        s, a, r, s_, step, terminal = self.critic.storage[-1]
        td_error_scale = self.critic.learn(s, a, r, s_, step)
        if self.config.td_decay:
            td_param = self.config.td_param * self.config.td_decay_rate ** int(current_episode / self.config.td_decay_step)
        else:
            td_param = self.config.td_param
        td_param *= (1 / (math.floor(max(episode_t,0) / 100) + 1))
        td_error_scale *= td_param

        self.critic.td_storage.append((s, a, r, s_, step, td_error_scale))
            
        if len(self.critic.td_storage) <= self.config.batch_size:
            return
        start = np.random.randint(0, len(self.critic.td_storage) - self.config.batch_size, size = 1)[0]
        batch_indices = [-1] + list(np.arange(start, start + self.config.batch_size))
        for i in batch_indices:
            s, a, r, s_, step, td = self.critic.td_storage[i]
            s = torch.FloatTensor(s.reshape(1, -1)).to(self.device)
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

    def transform(self, s):
        return np.matmul([s], self.mat)[0]

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


def evaluate(env, agent):
    rewards = []
    for _ in range(10):
        s = env.reset()
        eval_r = 0
        done = False
        while not done:
            a = agent.select_action(s, True)
            s_, r, done, _ = env.step(a)
            eval_r += r
            s = s_
        rewards.append(eval_r)
    return np.mean(rewards)


def run_env(seed, agent_name, config):
    max_episodes = 10000
    max_steps = 50000
    max_ep_steps = 500
    env_name = 'CartPole-v1'
    render = False

    if not os.path.exists(os.path.join(env_name, 'Experiments', agent_name, 'Trial_%d' % seed)):
        os.makedirs(os.path.join(env_name, 'Experiments', agent_name,'Trial_%d' % seed))

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    env._max_episode_steps = max_ep_steps
    env.seed(seed)

    action_dim = env.action_space.n
    agent = NNAC(env, config)
    
    scores, eval_scores, rewards = [], [], []
    timestep = 0
    solve = 0
    skip = False
    
    for i_episode in range(max_episodes):
        if timestep > max_steps:
            break

        episode_r, episode_t = 0, 0
        s = env.reset()
        while True:
            if render: env.render()

            a = agent.select_action(s, skip)
            
            s_, r, done, _ = env.step(a)
            a = np.reshape(a, (1))
            agent.remember(s, a, r, s_, done, episode_t)

            episode_r += r
            episode_t += 1
            timestep += 1
            s = s_

            if solve <= 20:
                agent.learn(i_episode, episode_t, skip)

            if done or episode_t > max_ep_steps:
                if skip:
                    agent.critic.tree = cKDTree(config.weights * agent.critic.states_all)            
                
                solve += 1 if episode_t == max_ep_steps else 0

                skip = adaptive_lr(agent, max_ep_steps, episode_t, skip, solve)

                rewards.append(episode_r)
                scores.append((timestep, episode_r))
                print("[Timestep %d] Episode %d: %f, Running reward: %f" % (timestep, i_episode, episode_r, np.mean(rewards[-20:])))

                if (i_episode + 1) % config.save_freq == 0:
                    agent.save(env_name, agent_name, seed, i_episode)
                    np.save(os.path.join(env_name, 'Experiments', agent_name, 'Trial_%d' % seed, 'train_scores.npy'), scores)
                    
                    eval_scores.append((timestep, evaluate(env, agent)))
                    print("[EVAL Timestep %d] Episode %d: %f" % (timestep, i_episode, eval_scores[-1][1]))
                    np.save(os.path.join(env_name, 'Experiments', agent_name, 'Trial_%d' % seed, 'eval_scores.npy'), eval_scores)

                break

    agent.save(env_name, agent_name, SEED, i_episode)
    np.save(os.path.join(ENV_NAME, 'Experiments', agent_name, 'Trial_%d' % seed, 'train_scores.npy'), scores)
                    
    eval_scores.append((timestep, evaluate(env, agent)))
    np.save(os.path.join(env_name, 'Experiments', agent_name, 'Trial_%d' % seed, 'eval_scores.npy'), eval_scores)



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    agent_name = "NNAC_transform"

    if agent_name == "NNAC":
        config = edict({'lr_actor': 5e-4, 'layer_info': [(32, 32)], 'save_freq': 20, 'init_w': 3e-3, 'batch_size': 32, 'buf_max_size': 1e5,
                    'planning_horizon': 12, 'L': 7, 'gamma': 0.99, 'K_neighbors': 1, 'r_scale': 1, 'grad_clip': 10,
                    'td_decay':False, 'td_decay_step': 10, 'td_decay_rate': 0.95, 'td_param': 0.05, 'neg_td_scale': 1, 
                    'dim_after': None, 'weights': np.array([0.25, 0.25, 0.25, 0.25, 1])})
    
    elif agent_name == "NNAC_transform":
        dim_after = 10
        config = edict({'lr_actor': 5e-4, 'layer_info': [(32, 32)], 'save_freq': 20, 'init_w': 3e-3, 'batch_size': 32, 'buf_max_size': 1e5,
                    'planning_horizon': 12, 'L': 4, 'gamma': 0.99, 'K_neighbors': 1, 'r_scale': 1, 'grad_clip': 10,
                    'td_decay': False, 'td_decay_step': 10, 'td_decay_rate': 0.95, 'td_param': 0.05, 'neg_td_scale': 1, 
                    'dim_after': dim_after, 'weights': np.array([1 / dim_after for i in range(dim_after)] + [1])})

    for i in range(5):
        run_env(i, agent_name, config)
