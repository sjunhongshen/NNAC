from stable_baselines.ppo1 import PPO1
from stable_baselines import DQN, TRPO
from stable_baselines.bench import Monitor
import gym, os
from IMGCartPoleEnv import IMGCartPoleEnv

def run_cartpole(agent):
    log_dir = "%s/" % agent
    os.makedirs(log_dir, exist_ok = True)

    env = gym.make('CartPole-v1')
    env._max_episode_steps = 500
    env = Monitor(env, log_dir)

    if agent == "TRPO":
        model = TRPO('MlpPolicy', env, verbose = 1, seed = 0)
    elif agent == "PPO": 
        model = PPO1('MlpPolicy', env, schedule = 'linear', verbose = 1, seed = 0)
    elif agent == "DQN":
        model = DQN('MlpPolicy', env, verbose = 1, seed = 0)

    model.learn(total_timesteps = 50000)

def run_cartpole_img(agent):
    # Note that the CNN policy network should be different from the default implementation, 
    # check the appendix for the network structure used

    log_dir = "%s/" % agent
    os.makedirs(log_dir, exist_ok = True)

    env = IMGCartPoleEnv()
    env._max_episode_steps = 500
    env = Monitor(env, log_dir)

    if agent == "TRPO":
        model = TRPO('CnnPolicy', env, verbose = 1, seed = 0, vf_stepsize = 1e-3)
    elif agent == "PPO": 
        model = PPO1('CnnPolicy', env, schedule = 'linear', verbose = 1, seed = 0)
    elif agent == "DQN":
        model = DQN('CnnPolicy', env, learning_rate = 1e-4, verbose = 1, seed = 0)

    model.learn(total_timesteps = 50000)


if __name__ == '__main__':
    run_cartpole("TRPO")