import numpy as np
from scipy.spatial import cKDTree

class Critic_NearestNeighbor:
    def __init__(self, actor, config):
        self.actor = actor
        self.config = config

        # parameters to tune
        self.horizon = config.planning_horizon
        self.L = config.L
        self.discount = config.discount

        self.max_size = config.buf_max_size
        self.states_all = []
        self.storage = []
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
            a = self.actor.select_action(s, self.config.policy_noise)

        distances, indices = self.tree.query(np.concatenate((s, a)), k = self.K_neighbors, n_jobs = -1)
        nearest_neighbors = self.storage[indices]
        
        vals = []
        for i in range(self.K_neighbors):
            nn = nearest_neighbors[i] if self.K_neighbors > 1 else nearest_neighbors
            d = distances[i] if self.K_neighbors > 1 else distances
            if nn[-1]:
                vals.append(nn[2] + self.L * d)
            else:
                vals.append(nn[2] + self.L * d + self.discount * self.estimate(step + 1, nn[3]))

        return np.min(vals)
        

    def learn(self, s, a, r, s_, step):
        v = self.estimate(step, s)
        v_ = self.estimate(step + 1, s_)
        td_error = r + self.discount * v_ - v

        return td_error if td_error > 0 else td_error * self.config.neg_td_scale
