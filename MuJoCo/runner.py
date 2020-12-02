import numpy as np
import copy, os, sys

class Runner():    
    def __init__(self, env, agent, config):   
        self.env = env
        self.agent = agent
        self.config = config
        self.obs = env.reset()
        
    def next_step(self, episode_timesteps):
        action = self.agent.select_action(np.array(self.obs), self.config.policy_noise)
        
        new_obs, reward, done, _ = self.env.step(action) 

        done_bool = float(done) if episode_timesteps < self.env._max_episode_steps else 0
        self.agent.add(self.obs, action, new_obs, reward, done_bool)        
        self.obs = new_obs
        
        return reward, done

    def evaluate(self, eval_episodes = 100, render = False):
        avg_reward = 0.
        for i in range(eval_episodes):
            obs = self.env.reset()
            done = False
            while not done:
                if render:
                    self.env.render() 
                action = self.agent.select_action(np.array(obs), noise = 0)
                obs, reward, done, _ = self.env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes

        print("\n-------------------------------------")
        print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, avg_reward))
        print("---------------------------------------")

        return avg_reward

    def observe(self):
        if not self.config.load_observation:
            time_steps = 0
            episode_timesteps = 0
            obs = self.env.reset()

            while time_steps < self.config.observation_steps:
                action = self.env.action_space.sample()
                new_obs, reward, done, _ = self.env.step(action)
                done_bool = float(done) if episode_timesteps < self.env._max_episode_steps else 0

                self.agent.add(obs, action, new_obs, reward, done_bool)

                obs = new_obs
                time_steps += 1
                episode_timesteps += 1

                if done:
                    obs = self.env.reset()
                    episode_timesteps = 0

                print("\rPopulating Buffer {}/{}.".format(time_steps, self.config.observation_steps), end = "")
                sys.stdout.flush()

            np.savez("%s/Data/storage-%d.npz" % (self.config.env_name, self.config.observation_steps), s = self.agent.replay_buffer.state, a = self.agent.replay_buffer.action, s_ = self.agent.replay_buffer.next_state, r = self.agent.replay_buffer.reward, d = self.agent.replay_buffer.not_done)
            np.savez("%s/Data/ptr-%d.npz" % (self.config.env_name, self.config.observation_steps), p = self.agent.replay_buffer.ptr, s = self.agent.replay_buffer.size)
        else:
            data = np.load("%s/Data/storage-%d.npz" % (self.config.env_name, self.config.observation_steps))
            self.agent.replay_buffer.state, self.agent.replay_buffer.action, self.agent.replay_buffer.next_state, self.agent.replay_buffer.reward, self.agent.replay_buffer.not_done = data['s'], data['a'], data['s_'], data['r'], data['d'] 
            param = np.load("%s/Data/ptr-%d.npz" % (self.config.env_name, self.config.observation_steps))
            self.agent.replay_buffer.ptr, self.agent.replay_buffer.size = param['p'], param['s']

    def run(self):
        self.observe()

        total_timesteps = 0
        eval_timesteps = 0
        episode_num = 0
        episode_reward = 0
        episode_timesteps = 0
        done = False 
        obs = self.env.reset()
        
        evaluations = []
        rewards = []
        reward_tuples = []
        eval_reward_tuples = []
        best_avg = -np.inf
        
        while total_timesteps < self.config.max_timesteps:      
            if done and total_timesteps != 0: 
                rewards.append(episode_reward)
                reward_tuples.append((total_timesteps, episode_reward))
                avg_reward = np.mean(rewards[-100:])
                
                self.agent.train(episode_timesteps, self.config.batch_size, episode_num)

                if (episode_num + 1) % self.config.save_freq == 0 or best_avg < avg_reward:
                    print("\rTotal T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f} Max r: {:f} Timestep: {:f}\n".format(
                        total_timesteps, episode_num, episode_reward, avg_reward, np.max(rewards), episode_timesteps), end = "")
                    
                    np.save(os.path.join(self.config.env_name, 'Experiments', 'Trial_%d' % self.config.seed, self.config.agent, 'train_rewards.npy'), reward_tuples)
                    self.agent.save(os.path.join(self.config.env_name, 'Experiments', 'Trial_%d' % self.config.seed, self.config.agent))

                if eval_timesteps > self.config.eval_freq:
                    eval_reward_tuples.append((total_timesteps, self.evaluate(self.config.eval_num)))
                    np.save(os.path.join(self.config.env_name, 'Experiments', 'Trial_%d' % self.config.seed, self.config.agent, 'eval_rewards.npy'), eval_reward_tuples)
                    eval_timesteps = 0

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 
                best_avg = max(best_avg, avg_reward)
                self.obs = self.env.reset()

            reward, done = self.next_step(episode_timesteps)

            episode_reward += reward
            episode_timesteps += 1
            total_timesteps += 1
            eval_timesteps += 1

            if self.config.lr_decay:
                self.agent.actor_scheduler.step()
                self.agent.critic_scheduler.step()

        np.save(os.path.join(self.config.env_name, 'Experiments', 'Trial_%d' % self.config.seed, self.config.agent, 'train_rewards.npy'), reward_tuples)
        self.agent.save(os.path.join(self.config.env_name, 'Experiments', 'Trial_%d' % self.config.seed, self.config.agent))
        
        eval_reward_tuples.append((total_timesteps, self.evaluate(self.config.eval_num)))
        np.save(os.path.join(self.config.env_name, 'Experiments', 'Trial_%d' % self.config.seed, self.config.agent, 'eval_rewards.npy'), eval_reward_tuples)
