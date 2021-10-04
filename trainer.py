import os
import numpy as np
import torch
import collections
from utils.logger import logger
from utils.logger import create_stats_ordered_dict
from utils import utils


class Trainer(object):
    def __init__(self, agent, expl_env, eval_env, replay_buffer, device, start_timesteps=25e3):
        self.agent = agent
        self.device = device

        self.expl_env = expl_env
        self.eval_env = eval_env

        self.replay_buffer = replay_buffer

        self.max_episode_steps = 1000
        self.start_timesteps = int(start_timesteps)

    def train(self, num_epochs=1000, num_iters_per_epoch=1000):

        total_timesteps = 0
        episode_num = 0
        episode_reward = 0
        episode_timesteps = 0
        evaluations = []

        state, done = self.expl_env.reset(), False
        for curr_epoch in range(num_epochs):
            for _ in range(num_iters_per_epoch):
                if total_timesteps < self.start_timesteps:
                    action = self.expl_env.action_space.sample()
                else:
                    action = self.agent.sample_action(np.array(state))

                next_state, reward, done, _ = self.expl_env.step(action)
                done_bool = float(done) if episode_timesteps < self.max_episode_steps else 0
                episode_timesteps += 1
                total_timesteps += 1
                self.replay_buffer.add(state, action, next_state, reward, done_bool)

                state = next_state
                episode_reward += reward

                if total_timesteps >= self.start_timesteps:
                    gf1_loss, gf2_loss, actor_loss, log_pi = self.agent.train_from_batch(self.replay_buffer)

                if done:
                    # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                    utils.print_banner(
                        f"Total T: {total_timesteps + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                    # Reset environment
                    state, done = self.expl_env.reset(), False
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1

            # TODO: add code
            # Evaluate episode
            if total_timesteps >= self.start_timesteps:
                utils.print_banner(f"Train step: {total_timesteps}", separator="*", num_star=90)
                eval_res = self.eval_policy()
                evaluations.append(eval_res)
                # np.save(os.path.join(output_dir, "eval"), evaluations)
                logger.record_tabular('Training Epochs', curr_epoch)
                logger.record_tabular('GF1 Loss', gf1_loss.cpu().data.numpy())
                logger.record_tabular('GF2 Loss', gf2_loss.cpu().data.numpy())
                logger.record_tabular('Actor Loss', actor_loss.cpu().data.numpy())
                logger.record_tabular('Policy log_pi', log_pi.cpu().data.numpy())
                logger.record_tabular('Average Episodic Reward', eval_res)

                logger.dump_tabular()

    # Runs policy for X episodes and returns average reward
    # A fixed seed is used for the eval environment
    def eval_policy(self, eval_episodes=10):

        avg_reward = 0.
        for _ in range(eval_episodes):
            state, done = self.eval_env.reset(), False
            while not done:
                action = self.agent.sample_action(np.array(state))
                state, reward, done, _ = self.eval_env.step(action)
                avg_reward += reward

        avg_reward /= eval_episodes

        utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        return avg_reward

