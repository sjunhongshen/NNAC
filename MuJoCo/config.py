from easydict import EasyDict as edict
import copy
from DDPG import DDPG
from TD3 import TD3
from NNDDPG import NNDDPG
from NNTD3 import NNTD3

configs = [   
    edict({'env_name': 'Walker2d-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'DDPG',
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps': 1e6, 'batch_size': 256, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'eps': 1e-3, 
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.95, 'decay_step': 20000, 'policy_noise': 0.15,
                    'discount': 0.99, "method": DDPG}),

    edict({'env_name': 'Walker2d-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'TD3', 
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps':1e6, 'batch_size': 256, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'policy_freq': 2, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'eps': 1e-3,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.3, 'decay_step': 200000, 'policy_noise': 0.2, 'noise_clip': 0.5, 
                    'discount': 0.99, "method": TD3}),

    edict({'env_name': 'Walker2d-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'NNDDPG', 'eps': 1e-3,
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps': 1e6, 'batch_size': 256, 'td_batch_size': 32, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'policy_noise_dist': 0.8,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.95, 'decay_step': 20000, 'policy_noise': 0.2, 'agent_param': 1,
                    'buf_max_size': 1e6, 'planning_horizon': 12, 'L': 7, 'discount': 0.99, 'K_neighbors': 1, 'r_scale': 0.1, 'grad_clip': 10, 
                    'tau_nn': 0.2, 'neg_td_scale': 0.3, 'mini_batch': 4, 'nn_param_actor': 1, 'nn_param_critic': 1, 'turning_point': 20,
                    'method': NNDDPG}),

    edict({'env_name': 'Walker2d-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'NNTD3', 'eps': 1e-3, 'agent_param': 1,
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps':1e6, 'batch_size': 256, 'td_batch_size': 32, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'policy_freq': 2, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'policy_noise_dist': 0.8,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.3, 'decay_step': 200000, 'policy_noise': 0.2, 'noise_clip': 0.5, 
                    'buf_max_size': 1e6, 'planning_horizon': 12, 'L': 4, 'discount': 0.99, 'K_neighbors': 1, 'r_scale': 0.1, 'grad_clip': 10, 
                    'tau_nn': 0.2, 'neg_td_scale': 0.3, 'mini_batch': 4, 'nn_param_actor': 10, 'nn_param_critic': 1, 'turning_point': 20,
                    'method': NNTD3}),

    edict({'env_name': 'Hopper-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'DDPG',
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps': 1e6, 'batch_size': 256, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'eps': 1e-3,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.95, 'decay_step': 20000, 'policy_noise': 0.3,
                    'discount': 0.99, "method": DDPG}),

    edict({'env_name': 'Hopper-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'TD3', 
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps':1e6, 'batch_size': 256, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'policy_freq': 2, 'lr_actor': 5e-4, 'layer_info': [(400, 300)], 'eps': 1e-3,
                    'lr_critic': 5e-4, 'lr_decay': True, 'decay_rate': 0.3, 'decay_step': 200000, 'policy_noise': 0.3, 'noise_clip': 0.5, 
                    'discount': 0.99, "method": TD3}),

    edict({'env_name': 'Hopper-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'NNDDPG', 'eps': 1e-3, 'agent_param': 1,
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps': 1e6, 'batch_size': 256, 'td_batch_size': 32, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'policy_noise_dist': 0.8,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.95, 'decay_step': 20000, 'policy_noise': 0.3, 
                    'buf_max_size': 1e6, 'planning_horizon': 12, 'L': 7, 'discount': 0.99, 'K_neighbors': 1, 'r_scale': 0.1, 'grad_clip': 10, 
                    'tau_nn': 0.2, 'neg_td_scale': 0.3, 'mini_batch': 4, 'nn_param_actor': 10, 'nn_param_critic': 1, 'turning_point': 20,
                    'method': NNDDPG}),

    edict({'env_name': 'Hopper-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'NNTD3', 'eps': 1e-3, 'agent_param': 1,
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps':1e6, 'batch_size': 256, 'td_batch_size': 32, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'policy_freq': 2, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'policy_noise_dist': 0.8,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.3, 'decay_step': 200000, 'policy_noise': 0.3, 'noise_clip': 0.5, 
                    'buf_max_size': 1e6, 'planning_horizon': 12, 'L': 4, 'discount': 0.99, 'K_neighbors': 1, 'r_scale': 0.1, 'grad_clip': 10, 
                    'tau_nn': 0.2, 'neg_td_scale': 0.3, 'mini_batch': 4, 'nn_param_actor': 10, 'nn_param_critic': 1, 'turning_point': 20,
                    'method': NNTD3}),

    edict({'env_name': 'HalfCheetah-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'DDPG',
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps': 1e6, 'batch_size': 256, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'eps': 1e-3,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.95, 'decay_step': 20000, 'policy_noise': 0.2,
                    'discount': 0.99, "method": DDPG}),

    edict({'env_name': 'HalfCheetah-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'TD3', 
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps':1e6, 'batch_size': 256, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'policy_freq': 2, 'lr_actor': 5e-4, 'layer_info': [(400, 300)], 'eps': 1e-3,
                    'lr_critic': 5e-4, 'lr_decay': False, 'decay_rate': 0.99, 'decay_step': 20000,'policy_noise': 0.2, 'noise_clip': 0.5, 
                    'discount': 0.99, "method": TD3}),

    edict({'env_name': 'HalfCheetah-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'NNDDPG', 'eps': 1e-3, 'agent_param': 1,
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps': 1e6, 'batch_size': 256, 'td_batch_size': 32, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'policy_noise_dist': 0.8,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.95, 'decay_step': 20000, 'policy_noise': 0.2, 
                    'buf_max_size': 1e6, 'planning_horizon': 12, 'L': 5, 'discount': 0.99, 'K_neighbors': 1, 'r_scale': 1, 'grad_clip': 10, 
                    'tau_nn': 0.2, 'neg_td_scale': 0.3, 'mini_batch': 4, 'nn_param_actor': 10, 'nn_param_critic': 10, 'turning_point': 20,
                    'method': NNDDPG}),

    edict({'env_name': 'HalfCheetah-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'NNTD3', 'eps': 1e-3, 'agent_param': 1,
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps':1e6, 'batch_size': 256, 'td_batch_size': 32, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'policy_freq': 2, 'lr_actor': 5e-4, 'layer_info': [(400, 300)], 'policy_noise_dist': 0.8,
                    'lr_critic': 5e-4, 'lr_decay': True, 'decay_rate': 0.99, 'decay_step': 20000, 'policy_noise': 0.2, 'noise_clip': 0.5, 
                    'buf_max_size': 1e6, 'planning_horizon': 12, 'L': 5, 'discount': 0.99, 'K_neighbors': 1, 'r_scale': 1, 'grad_clip': 10, 
                    'tau_nn': 0.2, 'neg_td_scale': 0.3, 'mini_batch': 4, 'nn_param_actor': 10, 'nn_param_critic': 1, 'turning_point': 20,
                    'method': NNTD3}),

    edict({'env_name': 'Ant-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'DDPG', 
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps': 1e6, 'batch_size': 256, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'lr_actor': 1e-4, 'layer_info': [(400, 300)], 'eps': 1e-3,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.95, 'decay_step': 20000,  'policy_noise': 0.2,
                    'discount': 0.99, "method": DDPG}),

    edict({'env_name': 'Ant-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'TD3', 
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps':1e6, 'batch_size': 256, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'policy_freq': 2, 'lr_actor': 5e-4, 'layer_info': [(400, 300)], 'eps': 1e-3,
                    'lr_critic': 5e-4, 'lr_decay': True, 'decay_rate': 0.3, 'decay_step': 200000,  'policy_noise': 0.2, 'noise_clip': 0.5, 
                    'discount': 0.99, "method": TD3}),

    edict({'env_name': 'Ant-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'NNDDPG', 'eps': 1e-3, 'agent_param': 0,
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps': 1e6, 'batch_size': 256, 'td_batch_size': 32, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'policy_noise_dist': 0.8,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.95, 'decay_step': 20000, 'policy_noise': 0.1, 
                    'buf_max_size': 1e6, 'planning_horizon': 5, 'L': 7, 'discount': 0.99, 'K_neighbors': 1, 'r_scale': 1, 'grad_clip': 10, 
                    'tau_nn': 0.2, 'neg_td_scale': 0.3, 'mini_batch': 4, 'nn_param_actor': 10, 'nn_param_critic': 1, 'turning_point': 10000,
                    'method': NNDDPG}),

    edict({'env_name': 'Ant-v2', 'eval_freq': 5000, 'eval_num': 10, 'agent': 'NNTD3', 'eps': 1e-3, 'agent_param': 1,
                    'load_observation': False, 'observation_steps': 10000, 'max_timesteps':1e6, 'batch_size': 256, 'td_batch_size': 32, 'init_w': 3e-3,
                    'save_freq': 100, 'tau': 0.005, 'policy_freq': 2, 'lr_actor': 1e-3, 'layer_info': [(400, 300)], 'policy_noise_dist': 0.8,
                    'lr_critic': 1e-3, 'lr_decay': True, 'decay_rate': 0.3, 'decay_step': 200000, 'policy_noise': 0.2, 'noise_clip': 0.5, 
                    'buf_max_size': 1e6, 'planning_horizon': 12, 'L': 4, 'discount': 0.99, 'K_neighbors': 1, 'r_scale': 0.1, 'grad_clip': 10, 
                    'tau_nn': 0.2, 'neg_td_scale': 0.3, 'mini_batch': 4, 'nn_param_actor': 10, 'nn_param_critic': 1, 'turning_point': 20,
                    'method': NNTD3})]

def get_config(agent, env):
    for c in configs:
        if c.agent == agent and c.env_name == env:
            return copy.deepcopy(c) 