for i in range(100):
	print(i)


import mujoco_py
import gym
import numpy as np
import torch
import argparse
import os

import utils
import DDPG

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state = eval_env.reset(seed=seed + 100)[0]
		terminated = False
		truncated = False
		while not (terminated or truncated):
			action = policy.select_action(np.array(state))
			print(eval_env.step(action))
			state, reward, terminated, truncated, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	print(avg_reward)
	return avg_reward

ENV = "HalfCheetah-v2"           # OpenAI gym environment name
SEED = 100           # Sets Gym, PyTorch and Numpy seeds
START_TIMESTEPS = 25e3         # Time steps initial random policy is used
EVAL_FREQ = 5e3            # How often (time steps) we evaluate
MAX_TIMESTEPS = 1e6            # Max time steps to run environment
EXPL_NOISE = 0.1         # Std of Gaussian exploration noise
BATCH_SIZE = 256           # Batch size for both actor and critic
DISCOUNT = 0.99          # Discount factor
TAU = 0.005          # Target network update rate
POLICY_NOISE = 0.2           # Noise added to target policy during critic update
NOISE_CLIP = 0.5         # Range to clip target policy noise
POLICY_FREQ = 2            # Frequency of delayed policy updates
SAVE_MODEL = "store_true"         # Save model and optimizer parameters
LOAD_MODEL = ""          # Model load file name, "" doesn't load, "default" uses file_name


env = gym.make(ENV)

# Set seeds
env.action_space.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

kwargs = {
	"state_dim": state_dim,
	"action_dim": action_dim,
	"max_action": max_action,
	"discount": DISCOUNT,
	"tau": TAU,
}


policy = DDPG.DDPG(**kwargs)

replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

# Evaluate untrained policy
evaluations = [eval_policy(policy, ENV, SEED)]
print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa")

state, done = env.reset(seed=SEED), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0

for t in range(int(MAX_TIMESTEPS)):
	
	episode_timesteps += 1

	# Select action randomly or according to policy
	if t < START_TIMESTEPS:
		action = env.action_space.sample()
	else:
		action = (
			policy.select_action(np.array(state))
			+ np.random.normal(0, max_action * EXPL_NOISE, size=action_dim)
		).clip(-max_action, max_action)

	# Perform action
	next_state, reward, done, _, _ = env.step(action) 
	done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

	# Store data in replay buffer
	replay_buffer.add(state[0], action, next_state, reward, done_bool)

	state = next_state
	episode_reward += reward

	# Train agent after collecting sufficient data
	if t >= START_TIMESTEPS:
		policy.train(replay_buffer, BATCH_SIZE)

	if done: 
		# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
		print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
		# Reset environment
		state, done = env.reset(seed=SEED), False
		episode_reward = 0
		episode_timesteps = 0
		episode_num += 1 

	# Evaluate episode
	if (t + 1) % EVAL_FREQ == 0:
		evaluations.append(eval_policy(policy, ENV, SEED))
		# np.save(f"./results/{'jonathan'}", evaluations)
		# if SAVE_MODEL: policy.save(f"./models/{'jonathan'}")
