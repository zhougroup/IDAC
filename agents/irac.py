import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.distributions import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
NEGATIVE_SLOPE = 1. / 100.

class Implicit_Actor(nn.Module):
    """
    Implicit Policy
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 noise_dim,
                 max_action,
                 device,
                 hidden_sizes=[256,256],
                 layer_norm=False):
        super(Implicit_Actor, self).__init__()

        self.layer_norm = layer_norm
        self.noise_dim = noise_dim
        self.base_fc = []
        last_size = state_dim
        for next_size in hidden_sizes[:-1]:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)

        last_hidden_size = hidden_sizes[-1]
        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, last_hidden_size),
            nn.Tanh()
        )

        self.last_fc = nn.Sequential(
            nn.Linear(last_hidden_size, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action
        self.device = device

    def forward(self, state):
        noise = torch.randn((state.size(0), self.noise_dim), device=self.device)

        s_h = self.base_fc(state)
        n_h = self.noise_fc(noise)

        h = s_h * n_h

        action = self.last_fc(h) * self.max_action

        return action


class Critic(nn.Module):
    """
    Implicit Distributional Critic
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 layer_norm=False,
                 hidden_sizes=[256, 256]):
        super(Critic, self).__init__()

        self.device = device
        self.layer_norm = layer_norm
        self.base_fc = []
        last_size = state_dim + action_dim
        for next_size in hidden_sizes:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)

        last_hidden_size = hidden_sizes[-1]
        self.last_fc = nn.Linear(last_hidden_size, 1)

    def forward(self, state, action):
        h = torch.cat([state, action], dim=1)
        h = self.base_fc(h)
        q = self.last_fc(h)
        return q


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.hidden_size = (256, 256)
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, self.hidden_size[0]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE),
            nn.Linear(self.hidden_size[1], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        validity = self.model(x)
        return validity


class IRAC(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            noise_dim,
            max_action,
            device,
            hidden_sizes,
            discount,                       # discount factor
            tau,                           # target network update rate
            actor_lr,                       # actor learning rate
            critic_lr,                      # critic learning rate
            dis_lr=2e-4,
            batch_size=256,
            pi_bn=False,                    # policy batch normalization
            cr_bn=False,                    # critic batch normalization
            num_quantiles=21,
            log_alpha=2.0
    ):
        self.tau = tau
        self.device = device
        self.discount = discount
        self.batch_size = batch_size
        self.max_action = max_action
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        self.actor = Implicit_Actor(state_dim,
                                    action_dim,
                                    noise_dim,
                                    max_action,
                                    device,
                                    layer_norm=pi_bn,
                                    hidden_sizes=hidden_sizes).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(0.5, 0.999))

        self.gf1 = Critic(state_dim,
                          action_dim,
                          device,
                          layer_norm=cr_bn,
                          hidden_sizes=hidden_sizes).to(device)
        self.gf1_optimizer = torch.optim.Adam(self.gf1.parameters(), lr=critic_lr)

        self.gf2 = Critic(state_dim,
                          action_dim,
                          device,
                          layer_norm=cr_bn,
                          hidden_sizes=hidden_sizes).to(device)
        self.gf2_optimizer = torch.optim.Adam(self.gf2.parameters(), lr=critic_lr)

        self.gf1_target = copy.deepcopy(self.gf1)
        self.gf2_target = copy.deepcopy(self.gf2)

        self.discriminator = Discriminator(state_dim=state_dim, action_dim=action_dim).to(device)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=dis_lr, betas=(0.5, 0.999))

        self.adversarial_loss = torch.nn.BCELoss()

        self.alpha = torch.FloatTensor([log_alpha]).exp().to(self.device)

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def quantile_regression_loss(self, input, target, tau, weight=1.0):
        """
        input: (N, T)
        target: (N, T)
        tau: (N, T)
        """
        input = input.unsqueeze(-1)
        target = target.detach().unsqueeze(-2)
        tau = tau.detach().unsqueeze(-1)
        # weight = weight.detach().unsqueeze(-2)
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
        sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
        rho = torch.abs(tau - sign) * L * weight
        return rho.sum(dim=-1).mean()

    def get_tau(self, obs):
        """
        Get random monotonically increasing quantiles
        """
        presum_tau = torch.rand(len(obs), self.num_quantiles, device=self.device) + 0.1
        presum_tau /= presum_tau.sum(dim=-1, keepdims=True)

        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau, device=self.device)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau_hat

    def perturb_data(self, state, action, s_std=1e-3, a_std=1e-2):
        state_p = torch.randn_like(state) * s_std + state
        action_p = torch.randn_like(action) * a_std + action
        return state_p, action_p.clamp(-self.max_action, self.max_action)

    def train_from_batch(self, replay_buffer, num_disc_iters=2):
        obs, actions, next_obs, rewards, not_dones = replay_buffer.sample(self.batch_size)

        """
        Update Distributional Critics
        """
        with torch.no_grad():
            new_next_actions = self.actor_target(next_obs)
            target_g1_values = self.gf1_target(next_obs, new_next_actions)
            target_g2_values = self.gf2_target(next_obs, new_next_actions)
            target_g_values = torch.min(target_g1_values, target_g2_values)
            g_target = rewards + not_dones * self.discount * target_g_values

        g1_pred = self.gf1(obs, actions)
        g2_pred = self.gf2(obs, actions)
        gf1_loss = F.mse_loss(g1_pred, g_target)
        gf2_loss = F.mse_loss(g2_pred, g_target)

        self.gf1_optimizer.zero_grad()
        gf1_loss.backward()
        self.gf1_optimizer.step()

        self.gf2_optimizer.zero_grad()
        gf2_loss.backward()
        self.gf2_optimizer.step()
        """
        Update Policy
        """
        new_actions = self.actor(obs)
        q1_new_actions = self.gf1(obs, new_actions)
        q2_new_actions = self.gf2(obs, new_actions)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions).mean()

        obs_g, _, _, _, _ = replay_buffer.sample(self.batch_size)
        actions_g = self.actor(obs_g)
        fake_samples = torch.cat([obs_g, actions_g], 1)
        generator_loss = self.adversarial_loss(self.discriminator(fake_samples),
                                               torch.ones(fake_samples.size(0), 1, device=self.device))

        policy_loss = self.alpha * generator_loss - q_new_actions
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        """
        Update Discriminator
        """
        for _ in range(num_disc_iters):
            obs_p, actions_p = self.perturb_data(obs, actions)
            true_samples = torch.cat([obs_p, actions_p], 1)
            obs_g, _, _, _, _ = replay_buffer.sample(self.batch_size)
            actions_g = self.actor(obs_g)
            fake_samples = torch.cat([obs_g, actions_g], 1)

            real_loss = self.adversarial_loss(self.discriminator(true_samples),
                                              torch.ones(true_samples.size(0), 1, device=self.device))
            fake_loss = self.adversarial_loss(self.discriminator(fake_samples.detach()),
                                              torch.zeros(fake_samples.size(0), 1, device=self.device))
            discriminator_loss = (real_loss + fake_loss) / 2
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
        """
        Soft Updates
        """
        # Update Target Networks
        for param, target_param in zip(self.gf1.parameters(), self.gf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.gf2.parameters(), self.gf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return dict(
            gf1_loss=gf1_loss.cpu().data.numpy(),
            gf2_loss=gf2_loss.cpu().data.numpy(),
            actor_loss=policy_loss.cpu().data.numpy(),
            generator_loss=generator_loss.cpu().data.numpy(),
            discriminator_loss=discriminator_loss.cpu().data.numpy()
        )


