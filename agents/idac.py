import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.distributions import TanhNormal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
LOG_PROB_MIN = -50.
EPS = 1e-6

class G_Actor(nn.Module):
    """
    Gaussian Policy
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 hidden_sizes=[256,256],
                 layer_norm=False):
        super(G_Actor, self).__init__()

        self.layer_norm = layer_norm
        self.base_fc = []
        last_size = state_dim
        for next_size in hidden_sizes:
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)

        last_hidden_size = hidden_sizes[-1]
        self.last_fc_mean = nn.Linear(last_hidden_size, action_dim)
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)

        self.max_action = max_action
        self.device = device

    def forward(self, state):

        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        tanh_normal = TanhNormal(mean, std, self.device)
        action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
        log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        action = action * self.max_action

        return action, log_prob

    def sample(self,
               state,
               reparameterize=False,
               deterministic=False):

        h = self.base_fc(state)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        if deterministic:
            action = torch.tanh(mean) * self.max_action
        else:
            tanh_normal = TanhNormal(mean, std, self.device)
            if reparameterize:
                action = tanh_normal.rsample()
            else:
                action = tanh_normal.sample()
            action = action * self.max_action

        return action

class Implicit_Actor(nn.Module):
    """
    Implicit Policy
    """
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 noise_dim,
                 noise_num, # L in the paper
                 max_action,
                 device,
                 hidden_sizes=[256,256],
                 layer_norm=False):
        super(Implicit_Actor, self).__init__()

        self.layer_norm = layer_norm
        self.noise_dim = noise_dim
        self.noise_num = noise_num
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
        self.last_fc_mean = nn.Linear(last_hidden_size, action_dim)
        self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)

        self.max_action = max_action
        self.device = device

    def forward(self, state):
        action, log_prob_main = self._forward(state)
        log_prob_aux = self._entropy(state, action)
        log_prob = torch.log((log_prob_main.exp() + log_prob_aux.exp() + EPS) / (self.noise_num + 1))

        return action, log_prob

    def sample(self,
               state,
               reparameterize=False,
               deterministic=False):
        M, _ = state.shape
        xi = torch.randn((1, self.noise_dim), device=self.device).repeat(M, 1)
        # xi = torch.normal(torch.zeros([M, self.noise_dim]),
        #                  torch.ones([M, self.noise_dim]), device=self.device)
        h = self.base_fc(torch.cat((state, xi), axis=-1))
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        if deterministic:
            action = torch.tanh(mean) * self.max_action
        else:
            tanh_normal = TanhNormal(mean, std, self.device)
            if reparameterize:
                action = tanh_normal.rsample()
            else:
                action = tanh_normal.sample()
            action = action * self.max_action

        return action

    def _forward(self, state):
        xi = torch.randn((1, self.noise_dim), device=self.device).repeat(state.shape[0], 1)
        h = torch.cat((state, xi), axis=-1)
        h = self.base_fc(h)
        mean = self.last_fc_mean(h)
        std = self.last_fc_log_std(h).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()

        tanh_normal = TanhNormal(mean, std, self.device)
        action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
        log_prob = tanh_normal.log_prob(action, pre_tanh_value=pre_tanh_value)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        action = action * self.max_action

        return action, log_prob

    def _entropy(self, state, action):
        M, _ = state.shape
        state = torch.repeat_interleave(state, self.noise_num, dim=0)
        xi = torch.randn((self.noise_num, self.noise_dim), device=self.device).repeat(M, 1)
        # xi = torch.normal(torch.zeros([M * rep, self.noise_dim]),
        #                  torch.ones([M * rep, self.noise_dim]), device=self.device)
        
        hidden = self.base_fc(torch.cat((state, xi), axis=-1))
        mean = self.last_fc_mean(hidden)
        std = self.last_fc_log_std(hidden).clamp(LOG_SIG_MIN, LOG_SIG_MAX).exp()
        tanh_normal = TanhNormal(mean, std, self.device)
        # action, pre_tanh_value = tanh_normal.rsample(return_pretanh_value=True)
        action = torch.repeat_interleave(action, self.noise_num, dim=0)
        log_prob = tanh_normal.log_prob(action).clamp_min(LOG_PROB_MIN)
        log_prob = log_prob.sum(dim=-1, keepdim=True).view(M, self.noise_num).logsumexp(dim=-1, keepdim=True)

        # log_prob = torch.reshape(log_prob, (M, rep))
        # log_prob = torch.logsumexp(log_prob, dim=-1, keepdim=True)

        return log_prob


class D_Critic(nn.Module):
    """
    Implicit Distributional Critic
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 noise_dim,
                 device,
                 layer_norm=True,
                 hidden_sizes=[256, 256]):
        super(D_Critic, self).__init__()

        self.device = device
        self.noise_dim = noise_dim
        self.layer_norm = layer_norm
        self.base_fc = []
        last_size = state_dim + action_dim + noise_dim
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

    def forward(self, state, action, noise):
        h = torch.cat([state, action, noise], dim=1)
        h = self.base_fc(h)
        h = self.last_fc(h)
        return h

    def sample(self, state, action, num_samples=1):
        batch_size = state.size(0)
        noise = torch.randn((num_samples, self.noise_dim), device=self.device)

        state_rpt = torch.repeat_interleave(state, num_samples, dim=0)
        action_rpt = torch.repeat_interleave(action, num_samples, dim=0)
        noise_rpt = noise.repeat(batch_size, 1)

        h = self.forward(state_rpt, action_rpt, noise_rpt)
        h = h.view(batch_size, num_samples)
        g_values = h.sort()[0]

        return g_values


class IDAC(object):
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
            batch_size=256,
            pi_bn=False,
            num_quantiles=21,
            target_entropy=None,
            alpha=0.2,
            use_automatic_entropy_tuning=False,
            pi_type='gauss',
            implicit_actor_args={
                "actor_noise_num": 5,
                "actor_noise_dim": 5
            }
    ):
        self.tau = tau
        self.device = device
        self.discount = discount
        self.batch_size = batch_size
        self.max_action = max_action
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles

        if pi_type == 'gauss':
            self.actor = G_Actor(state_dim, action_dim, max_action, device,
                                 layer_norm=pi_bn,
                                 hidden_sizes=hidden_sizes).to(device)
        elif pi_type == 'implicit':
            actor_noise_dim = implicit_actor_args['actor_noise_dim']
            actor_noise_num = implicit_actor_args['actor_noise_num']
            self.actor = Implicit_Actor(state_dim, action_dim,
                                 actor_noise_dim,
                                 actor_noise_num,
                                 max_action, device,
                                 layer_norm=pi_bn,
                                 hidden_sizes=hidden_sizes).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.gf1 = D_Critic(state_dim,
                            action_dim,
                            noise_dim,
                            device,
                            layer_norm=True,
                            hidden_sizes=hidden_sizes).to(device)
        self.gf1_optimizer = torch.optim.Adam(self.gf1.parameters(), lr=critic_lr)

        self.gf2 = D_Critic(state_dim,
                            action_dim,
                            noise_dim,
                            device,
                            layer_norm=True,
                            hidden_sizes=hidden_sizes).to(device)
        self.gf2_optimizer = torch.optim.Adam(self.gf2.parameters(), lr=critic_lr)

        self.gf1_target = copy.deepcopy(self.gf1)
        self.gf2_target = copy.deepcopy(self.gf2)

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(action_dim).item()  # heuristic value from Tuomas
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=actor_lr,
            )
        else:
            self.alpha = alpha

    def sample_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor.sample(state)
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

    def train_from_batch(self, replay_buffer):
        obs, actions, next_obs, rewards, not_dones = replay_buffer.sample(self.batch_size)
        """
        Update Alpha
        """
        new_actions, log_pi = self.actor(obs)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.alpha
        """
        Update Distributional Critics
        """
        with torch.no_grad():
            new_next_actions, new_log_pi = self.actor_target(next_obs)
            target_g1_values = self.gf1_target.sample(next_obs, new_next_actions, num_samples=self.num_quantiles)
            target_g2_values = self.gf2_target.sample(next_obs, new_next_actions, num_samples=self.num_quantiles)
            target_g_values = torch.min(target_g1_values, target_g2_values) - alpha * new_log_pi
            g_target = rewards + not_dones * self.discount * target_g_values

        tau_hat_1 = self.get_tau(obs)
        tau_hat_2 = self.get_tau(obs)
        g1_pred = self.gf1.sample(obs, actions, num_samples=self.num_quantiles)
        g2_pred = self.gf2.sample(obs, actions, num_samples=self.num_quantiles)
        gf1_loss = self.quantile_regression_loss(g1_pred, g_target, tau_hat_1)
        gf2_loss = self.quantile_regression_loss(g2_pred, g_target, tau_hat_2)

        self.gf1_optimizer.zero_grad()
        gf1_loss.backward()
        self.gf1_optimizer.step()

        self.gf2_optimizer.zero_grad()
        gf2_loss.backward()
        self.gf2_optimizer.step()
        """
        Update Policy
        """
        q1_new_actions = self.gf1.sample(obs, new_actions, num_samples=10).mean(dim=1, keepdims=True)
        q2_new_actions = self.gf2.sample(obs, new_actions, num_samples=10).mean(dim=1, keepdims=True)
        q_new_actions = torch.min(q1_new_actions, q2_new_actions)

        policy_loss = (alpha * log_pi - q_new_actions).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
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
            log_pi=log_pi.mean().cpu().data.numpy()
        )


