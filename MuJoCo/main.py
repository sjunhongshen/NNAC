import numpy as np
from easydict import EasyDict as edict
import gym, torch, os
from utils import NormalizedActions

from basic_agents.replay_buffer import ReplayBuffer
from DDPG import DDPG
from TD3 import TD3
from NNDDPG import NNDDPG
from NNTD3 import NNTD3
from runner import Runner
import config
from config import get_config

def run_env(seed, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    env_name = config.env_name
    agent_name = config.agent
    config['seed'] = seed

    save_exp_dir = os.path.join(env_name, 'Experiments', 'Trial_%d' % seed, agent_name)
    save_data_dir = os.path.join(env_name, 'Data')
    if not os.path.exists(save_exp_dir):
        os.makedirs(save_exp_dir)
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    env = gym.make(env_name)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    print(config)
    agent = config.method(state_dim, action_dim, max_action, env, config)
    runner = Runner(env, agent, config)
    runner.run()

def main():
    agents = ["DDPG", "NNDDPG", "TD3", "NNTD3"]
    envs = ["Hopper-v2", "Walker2d-v2", "HalfCheetah-v2", "Ant-v2"]

    config = get_config("DDPG", "Walker2d-v2")

    for i in range(5):
        run_env(i, config)

if __name__ == '__main__':
    main()