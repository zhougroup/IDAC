import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.size])
		np.save(f"{save_folder}_action.npy", self.action[:self.size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]


def print_banner(s, separator="-", num_star=60):
	print(separator * num_star, flush=True)
	print(s, flush=True)
	print(separator * num_star, flush=True)

def fast_clip_grad_norm(parameters, max_norm):
	"""Clips gradient norm of an iterable of parameters.
	Only support norm_type = 2
	max_norm = 0, skip the total norm calculation and return 0
	https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
	Returns:
		Total norm of the parameters (viewed as a single vector).
	"""
	max_norm = float(max_norm)
	if abs(max_norm) < 1e-6:  # max_norm = 0
		return 0
	else:
		if isinstance(parameters, torch.Tensor):
			parameters = [parameters]
		parameters = list(filter(lambda p: p.grad is not None, parameters))
		total_norm = torch.stack([(p.grad.detach().pow(2)).sum() for p in parameters]).sum().sqrt().item()
		clip_coef = max_norm / (total_norm + 1e-6)
		if clip_coef < 1:
			for p in parameters:
				p.grad.detach().mul_(clip_coef)
		return total_norm