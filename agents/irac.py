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
            nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
            nn.Tanh()
        )

        self.last_fc = nn.Sequential(
            nn.Linear(last_hidden_size, last_hidden_size),
            nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
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

class Implicit_Actor_2(nn.Module):
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
        super(Implicit_Actor_2, self).__init__()

        self.layer_norm = layer_norm
        self.noise_dim = noise_dim
        self.base_fc = []
        last_size = state_dim + noise_dim
        for next_size in hidden_sizes:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)

        last_hidden_size = hidden_sizes[-1]
        self.last_fc = nn.Sequential(
            nn.Linear(last_hidden_size, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action
        self.device = device

    def forward(self, state):
        noise = torch.randn((state.size(0), self.noise_dim), device=self.device)
        s_n = torch.cat([state, noise], dim=1)
        a = self.base_fc(s_n)
        a = self.last_fc(a) * self.max_action

        return a


# Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

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
            alpha=2.0
    ):
        self.tau = tau
        self.device = device
        self.discount = discount
        self.batch_size = batch_size
        self.max_action = max_action
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        self.actor = Implicit_Actor_2(state_dim,
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

        # self.discriminator = Discriminator(state_dim=state_dim, action_dim=action_dim).to(device)
        # self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=dis_lr, betas=(0.5, 0.999))
        # self.adversarial_loss = torch.nn.BCELoss()

        # latent_dim = action_dim * 2
        # self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        # self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=actor_lr)

        self.navigator = nn.Sequential(nn.Linear(state_dim+action_dim, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 5))
        self.nav_optimizer = torch.optim.Adam(self.navigator.parameters(), lr=actor_lr/2.)

        self.alpha = torch.tensor(alpha, device=device)

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
        presum_tau /= presum_tau.sum(dim=-1, keepdim=True)

        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = torch.zeros_like(tau, device=self.device)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau_hat

    def ct_loss(self, x, y, netN, rho):
        ######################## compute cost ######################
        f_x = x  # feature of x: B x d
        f_y = y  # feature of y: B x d
        cost = torch.norm(f_x[:, None] - f_y, dim=-1).pow(2)  # pairwise cost: B x B

        ######################## compute transport map ######################
        mse_n = (f_x[:, None] - f_y).pow(2)  # pairwise mse for navigator network: B x B x d
        d = netN(mse_n).squeeze().mul(-1)  # navigator distance: B x B
        forward_map = torch.softmax(d, dim=1)  # forward map is in y wise
        backward_map = torch.softmax(d, dim=0)  # backward map is in x wise

        ######################## compute CT loss ######################
        # element-wise product of cost and transport map
        ct = rho * (cost * forward_map).sum(1).mean() + (1 - rho) * (cost * backward_map).sum(0).mean()
        return ct

    def perturb_action(self, action, a_std=3e-3):
        action_p = torch.randn_like(action) * a_std + action
        return action_p.clamp(-self.max_action, self.max_action)

    def train_from_batch(self, replay_buffer, epoch):
        obs, actions, next_obs, rewards, not_dones = replay_buffer.sample(self.batch_size)

        # Variational Auto-Encoder Training
        # recon, mean, std = self.vae(obs, actions)
        # recon_loss = F.mse_loss(recon, actions)
        # KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        # vae_loss = recon_loss + 0.5 * KL_loss
        #
        # self.vae_optimizer.zero_grad()
        # vae_loss.backward()
        # self.vae_optimizer.step()
        """
        Update Distributional Critics
        """
        with torch.no_grad():
            new_next_actions = self.actor_target(next_obs)
            # vae_actions = self.vae.decode(next_obs)
            # next_fake_samples = torch.cat([next_obs, new_next_actions], 1)
            # q_penalty = self.adversarial_loss(self.discriminator(next_fake_samples),
            #                                   torch.ones(next_fake_samples.size(0), 1, device=self.device))
            # q_penalty = torch.sum((new_next_actions - vae_actions) ** 2, dim=1, keepdim=True)
            # scheduled_alpha = (self.alpha - np.exp((epoch - 200) / 800))
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
        # """
        # Update Discriminator
        # """
        # for _ in range(num_disc_iters):
        #     actions_p = self.perturb_action(actions)
        #     true_samples = torch.cat([obs, actions_p], 1)
        #     actions_pi = self.actor(obs)
        #     fake_samples = torch.cat([obs, actions_pi], 1)
        #
        #     real_loss = self.adversarial_loss(self.discriminator(true_samples),
        #                                       torch.ones(true_samples.size(0), 1, device=self.device))
        #     fake_loss = self.adversarial_loss(self.discriminator(fake_samples.detach()),
        #                                       torch.zeros(fake_samples.size(0), 1, device=self.device))
        #     discriminator_loss = (real_loss + fake_loss) / 2
        #     self.discriminator_optimizer.zero_grad()
        #     discriminator_loss.backward()
        #     self.discriminator_optimizer.step()
        """
        Update Policy
        """
        new_actions = self.actor(obs)
        ct_obs, ct_actions, _, _, _ = replay_buffer.sample(self.batch_size)
        # with torch.no_grad:
        #     perturbed_actions = self.perturb_action(new_actions)
        # regularization = F.mse_loss(new_actions, perturbed_actions)
        q1_new_actions = self.gf1(obs, new_actions)
        q2_new_actions = self.gf2(obs, new_actions)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions).mean()
        ct_x = torch.cat((ct_obs, ct_actions), dim=1)
        ct_y = torch.cat((obs, new_actions), dim=1)
        regularization = self.ct_loss(ct_x, ct_y, self.navigator, 0.6)

        # fake_samples = torch.cat([obs, new_actions], 1)
        # generator_loss = self.adversarial_loss(self.discriminator(fake_samples),
        #                                        torch.ones(fake_samples.size(0), 1, device=self.device))

        policy_loss = regularization * self.alpha - q_new_actions
        self.nav_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.nav_optimizer.step()
        self.actor_optimizer.step()
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
            # generator_loss=generator_loss.cpu().data.numpy(),
            # discriminator_loss=discriminator_loss.cpu().data.numpy(),
            q_values=q_new_actions.cpu().data.numpy(),
            # q_penalty=q_penalty.mean().cpu().data.numpy(),
        )
