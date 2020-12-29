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
from absl import flags
from absl import app
import pybullet_envs


flags.DEFINE_enum('env_name', None, 
                  ['Walker2DBulletEnv-v0', 
                   'ReacherBulletEnv-v0',
                   'HalfCheetahBulletEnv-v0',
                   'AntBulletEnv-v0',
                   'HopperBulletEnv-v0',
                   'Pendulum-v0'],
                   "environment to use")

flags.DEFINE_integer('seed', 0, "seed to run")
flags.DEFINE_enum('mode', 'train', ['debug', 'train'], "running mode")
flags.DEFINE_enum('model', None, 
                  ['idac',\
                   'idac_v1'], "model to train")
FLAGS = flags.FLAGS
def main(argv):

    env_name = FLAGS.env_name
    seed = FLAGS.seed

    if FLAGS.mode == 'debug':
        policy_kwargs = dict(layers=[32, 32])
        batch_size = 8
        total_timesteps = int(8e3)
        eval_freq = 1
        learning_starts = 1
    elif FLAGS.mode == 'train':
        policy_kwargs = dict(layers=[256, 256])
        batch_size = 256
        total_timesteps = int(1e6)
        eval_freq = 2000
        learning_starts = 100
    env = gym.make(env_name)


    env = Monitor(env, None, allow_early_resets=True)
    eval_env = gym.make(env_name)
    eval_env = Monitor(eval_env, None, allow_early_resets=True)

    lr = 3e-4

    set_global_seeds(seed)

    if FLAGS.model == 'idac':

        from stable_baselines.idac.policy_idac import MlpPolicy as policy
        from stable_baselines.idac.idac import IDAC as Model


        cwd = os.getcwd()
        log_dir = cwd + '/env_log/' + env_name + '/distribution_idac'
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


        model = Model(policy, env,  noise_dim=noise_dim, buffer_size=int(1e6),
                    verbose=2, batch_size=256, learning_rate=lr, gradient_steps=1, 
                    target_update_interval=1, policy_kwargs=policy_kwargs)

        model.learn(total_timesteps=total_timesteps, env_eval=eval_env, score_path=rew_eval_file, dis_path=dis_eval_file,
                    log_interval=1, seed=seed, path=file_name)

    if FLAGS.model == 'idac_variant1':

        from stable_baselines.idac.policy_idac import MlpPolicy as policy
        from stable_baselines.idac.idac_variant1 import IDAC as Model


        cwd = os.getcwd()
        log_dir = cwd + '/env_log/' + env_name + '/distribution_idac'
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


        model = Model(policy, env,  noise_dim=noise_dim, buffer_size=int(1e6),
                    verbose=2, batch_size=256, learning_rate=lr, gradient_steps=1, 
                    target_update_interval=1, policy_kwargs=policy_kwargs)

        model.learn(total_timesteps=total_timesteps, env_eval=eval_env, score_path=rew_eval_file, dis_path=dis_eval_file,
                    log_interval=1, seed=seed, path=file_name)
if __name__ == '__main__':
    app.run(main)