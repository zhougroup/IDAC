import gym
from stable_baselines.common.vec_env import DummyVecEnv
import os
import sys
from stable_baselines.common import set_global_seeds
import numpy as np
import gym
from stable_baselines.bench import Monitor
from stable_baselines import bench, logger
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines import results_plotter
import matplotlib.pyplot as plt
import tensorflow as tf


seed = 1

env_name = 'Pendulum-v0'


total_timesteps = 1000 * 1000
env = gym.make(env_name)
env = Monitor(env, None, allow_early_resets=True)
env_eval = gym.make(env_name)
env_eval = Monitor(env_eval, None, allow_early_resets=True)

lr = 3e-4

set_global_seeds(seed)

from stable_baselines.idac.policy_idac import MlpPolicy as policy
from stable_baselines.idac.idac import IDAC as Model


cwd = os.getcwd()
log_dir = cwd + '/env_log/' + env_name + '/distribution_sac_d4pg'
os.makedirs(log_dir, exist_ok=True)

noise_dim = 5
noise_num = 21

String = log_dir + '/Seed_' + str(seed) + '_Noisedim_' + str(noise_dim)

file_name = String +'_score.txt'
f = open(file_name, "w+")

dis_log_dir = cwd + '/dis_log/' + env_name + '/idac'
rew_log_dir = cwd + '/rew_log/' + env_name + '/idac'
os.makedirs(dis_log_dir, exist_ok=True)
os.makedirs(rew_log_dir, exist_ok=True)
dis_eval_file = dis_log_dir + '/Seed_' + str(seed)
rew_eval_file = rew_log_dir + '/Seed_' + str(seed)
dis_eval_interval = int(total_timesteps/500)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = Model(policy, env,  noise_dim=noise_dim, buffer_size=int(1e6),
              verbose=2, batch_size=256, learning_rate=lr, gradient_steps=1, 
              target_update_interval=1)

model.learn(total_timesteps=total_timesteps, env_eval=env_eval, score_path=rew_eval_file, dis_path=dis_eval_file,
             log_interval=1, seed=seed, path=file_name)


