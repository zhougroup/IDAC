from collections import OrderedDict

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self,
                 max_replay_buffer_size,
                 state_dim,
                 action_dim,
                 use_gpu=True,
                 device=None):
        observation_dim = state_dim
        action_dim = action_dim

        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = torch.zeros((max_replay_buffer_size, observation_dim), dtype=torch.float).pin_memory()
        self._next_obs = torch.zeros((max_replay_buffer_size, observation_dim), dtype=torch.float).pin_memory()
        self._actions = torch.zeros((max_replay_buffer_size, action_dim), dtype=torch.float).pin_memory()
        self._rewards = torch.zeros((max_replay_buffer_size, 1), dtype=torch.float).pin_memory()
        self._terminals = torch.zeros((max_replay_buffer_size, 1), dtype=torch.float).pin_memory()

        self._top = 0
        self._size = 0
        self.device = device
        self.use_gpu = use_gpu

        if use_gpu:
            self.batch = None

    def add_sample(self, observation, action, reward, next_observation, terminal):
        self._observations[self._top] = torch.from_numpy(observation)
        self._actions[self._top] = torch.from_numpy(action)
        self._rewards[self._top] = torch.from_numpy(reward)
        self._terminals[self._top] = torch.from_numpy(terminal)
        self._next_obs[self._top] = torch.from_numpy(next_observation)

        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self._size, batch_size)
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        return batch

    def preload(self, batch_size):
        try:
            self.batch = self.random_batch(batch_size)
        except StopIteration:
            self.batch = None
            return
        if self.use_gpu:
            # with torch.cuda.stream(self.stream):
            for k in self.batch:
                self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def next_batch(self, batch_size):
        if self.batch is None:
            self.preload(batch_size)
        batch = self.batch
        self.preload(batch_size)
        return batch

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([('size', self._size)])
