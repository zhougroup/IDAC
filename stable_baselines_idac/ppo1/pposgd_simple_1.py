from collections import deque
import time

import gym
import tensorflow as tf
import numpy as np
from mpi4py import MPI

from stable_baselines.common import Dataset, explained_variance, fmt_row, zipsame, ActorCriticRLModel, SetVerbosity, \
    TensorboardWriter
from stable_baselines import logger
import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.mpi_moments import mpi_moments
from stable_baselines.trpo_mpi.utils import add_vtarg_and_adv, flatten_lists
from stable_baselines.a2c.utils import total_episode_reward_logger


class PPO1(ActorCriticRLModel):
    """
    Proximal Policy Optimization algorithm (MPI version).
    Paper: https://arxiv.org/abs/1707.06347

    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param timesteps_per_actorbatch: (int) timesteps per actor per update
    :param clip_param: (float) clipping parameter epsilon
    :param entcoeff: (float) the entropy loss weight
    :param optim_epochs: (float) the optimizer's number of epochs
    :param optim_stepsize: (float) the optimizer's stepsize
    :param optim_batchsize: (int) the optimizer's the batch size
    :param gamma: (float) discount factor
    :param lam: (float) advantage estimation
    :param adam_epsilon: (float) the epsilon value for the adam optimizer
    :param schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
        'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01,
                 optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
                 schedule='linear', verbose=0, tensorboard_log=None, max_ep_length=1000,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False):

        super().__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                         _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

        self.gamma = gamma
        self.timesteps_per_actorbatch = timesteps_per_actorbatch
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.optim_batchsize = optim_batchsize
        self.lam = lam
        self.adam_epsilon = adam_epsilon
        self.schedule = schedule
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.lossandgrad = None
        self.adam = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.params = None
        self.step = None
        self.proba_step = None
        self.initial_state = None
        self.summary = None
        self.episode_reward = None
        self.max_ep_length = max_ep_length

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_pi
        action_ph = policy.pdtype.sample_placeholder([None])
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, action_ph, policy.policy
        return policy.obs_ph, action_ph, policy.deterministic_action

    def setup_model(self):
        with SetVerbosity(self.verbose):

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)

                # Construct network for new policy
                self.policy_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                # Network for old policy
                with tf.variable_scope("oldpi", reuse=False):
                    old_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                         None, reuse=False, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    # Target advantage function (if applicable)
                    atarg = tf.placeholder(dtype=tf.float32, shape=[None])

                    # Empirical return
                    ret = tf.placeholder(dtype=tf.float32, shape=[None])

                    # learning rate multiplier, updated with schedule
                    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])

                    # Annealed cliping parameter epislon
                    clip_param = self.clip_param * lrmult

                    obs_ph = self.policy_pi.obs_ph
                    action_ph = self.policy_pi.pdtype.sample_placeholder([None])

                    kloldnew = old_pi.proba_distribution.kl(self.policy_pi.proba_distribution)
                    ent = self.policy_pi.proba_distribution.entropy()
                    meankl = tf.reduce_mean(kloldnew)
                    meanent = tf.reduce_mean(ent)
                    pol_entpen = (-self.entcoeff) * meanent

                    # pnew / pold
                    ratio = tf.exp(self.policy_pi.proba_distribution.logp(action_ph) -
                                   old_pi.proba_distribution.logp(action_ph))

                    # surrogate from conservative policy iteration
                    surr1 = ratio * atarg
                    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

                    # PPO's pessimistic surrogate (L^CLIP)
                    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
                    vf_loss = tf.reduce_mean(tf.square(self.policy_pi.value_flat - ret))
                    total_loss = pol_surr + pol_entpen + vf_loss
                    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
                    self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

                    tf.summary.scalar('entropy_loss', pol_entpen)
                    tf.summary.scalar('policy_gradient_loss', pol_surr)
                    tf.summary.scalar('value_function_loss', vf_loss)
                    tf.summary.scalar('approximate_kullback-leibler', meankl)
                    tf.summary.scalar('clip_factor', clip_param)
                    tf.summary.scalar('loss', total_loss)

                    self.params = tf_util.get_trainable_vars("model")

                    self.assign_old_eq_new = tf_util.function(
                        [], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                         zipsame(tf_util.get_globals_vars("oldpi"), tf_util.get_globals_vars("model"))])

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self.adam = MpiAdam(self.params, epsilon=self.adam_epsilon, sess=self.sess)

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.optim_stepsize))
                    tf.summary.scalar('advantage', tf.reduce_mean(atarg))
                    tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_param))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', ret)
                        tf.summary.histogram('learning_rate', self.optim_stepsize)
                        tf.summary.histogram('advantage', atarg)
                        tf.summary.histogram('clip_range', self.clip_param)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', obs_ph)
                        else:
                            tf.summary.histogram('observation', obs_ph)

                self.step = self.policy_pi.step
                self.proba_step = self.policy_pi.proba_step
                self.initial_state = self.policy_pi.initial_state

                tf_util.initialize(sess=self.sess)

                self.summary = tf.summary.merge_all()

                self.lossandgrad = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                                                    [self.summary, tf_util.flatgrad(total_loss, self.params)] + losses)
                self.compute_losses = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                                                       losses)

    def learn(self, total_timesteps, callback=None, seed=None, path=None, log_interval=100, tb_log_name="PPO1",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn(seed)

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO1 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            with self.sess.as_default():
                self.adam.sync()

                # Prepare for rollouts
                seg_gen = traj_segment_generator(self.policy_pi, self.env,
                                                 self.timesteps_per_actorbatch, max_ep_length=self.max_ep_length)

                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()

                # rolling buffer for episode lengths
                lenbuffer = deque(maxlen=100)
                # rolling buffer for episode rewards
                rewbuffer = deque(maxlen=100)

                self.episode_reward = np.zeros((self.n_envs,))

                while True:
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    if total_timesteps and timesteps_so_far >= total_timesteps:
                        break

                    if self.schedule == 'constant':
                        cur_lrmult = 1.0
                    elif self.schedule == 'linear':
                        cur_lrmult = max(1.0 - float(timesteps_so_far) / total_timesteps, 0)
                    else:
                        raise NotImplementedError

                    logger.log("********** Iteration %i ************" % iters_so_far)

                    seg = seg_gen.__next__()
                    add_vtarg_and_adv(seg, self.gamma, self.lam)

                    # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                    observations, actions = seg["observations"], seg["actions"]
                    atarg, tdlamret = seg["adv"], seg["tdlamret"]

                    # true_rew is the reward without discount
                    if writer is not None:
                        self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                          seg["true_rewards"].reshape((self.n_envs, -1)),
                                                                          seg["dones"].reshape((self.n_envs, -1)),
                                                                          writer, self.num_timesteps)

                    # predicted value function before udpate
                    vpredbefore = seg["vpred"]

                    # standardized advantage function estimate
                    atarg = (atarg - atarg.mean()) / atarg.std()
                    dataset = Dataset(dict(ob=observations, ac=actions, atarg=atarg, vtarg=tdlamret),
                                      shuffle=not self.policy.recurrent)
                    optim_batchsize = self.optim_batchsize or observations.shape[0]

                    # set old parameter values to new parameter values
                    self.assign_old_eq_new(sess=self.sess)
                    logger.log("Optimizing...")
                    logger.log(fmt_row(13, self.loss_names))

                    # Here we do a bunch of optimization epochs over the data
                    for k in range(self.optim_epochs):
                        # list of tuples, each of which gives the loss for a minibatch
                        losses = []
                        for i, batch in enumerate(dataset.iterate_once(optim_batchsize)):
                            steps = (self.num_timesteps +
                                     k * optim_batchsize +
                                     int(i * (optim_batchsize / len(dataset.data_map))))
                            if writer is not None:
                                # run loss backprop with summary, but once every 10 runs save the metadata
                                # (memory, compute time, ...)
                                if self.full_tensorboard_log and (1 + k) % 10 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess,
                                                                                 options=run_options,
                                                                                 run_metadata=run_metadata)
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                else:
                                    summary, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                       batch["atarg"], batch["vtarg"], cur_lrmult,
                                                                       sess=self.sess)

                            self.adam.update(grad, self.optim_stepsize * cur_lrmult)
                            losses.append(newlosses)
                        logger.log(fmt_row(13, np.mean(losses, axis=0)))

                    logger.log("Evaluating losses...")
                    losses = []
                    for batch in dataset.iterate_once(optim_batchsize):
                        newlosses = self.compute_losses(batch["ob"], batch["ob"], batch["ac"], batch["atarg"],
                                                        batch["vtarg"], cur_lrmult, sess=self.sess)
                        losses.append(newlosses)
                    mean_losses, _, _ = mpi_moments(losses, axis=0)
                    logger.log(fmt_row(13, mean_losses))
                    for (loss_val, name) in zipsame(mean_losses, self.loss_names):
                        logger.record_tabular("loss_" + name, loss_val)
                    logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

                    # local values
                    lrlocal = (seg["ep_lens"], seg["ep_rets"])

                    # list of tuples
                    listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
                    lens, rews = map(flatten_lists, zip(*listoflrpairs))
                    lenbuffer.extend(lens)
                    rewbuffer.extend(rews)
                    if len(lenbuffer) > 0:
                        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
                        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    timesteps_so_far += current_it_timesteps
                    self.num_timesteps += current_it_timesteps
                    iters_so_far += 1

                    if iters_so_far % log_interval == 0:
                        with open(path, 'a') as f1:
                            f1.write("%i " % int(iters_so_far * self.timesteps_per_actorbatch))
                            f1.write("%f\n" % np.mean(rewbuffer))

                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", self.num_timesteps)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)
                    if self.verbose >= 1 and MPI.COMM_WORLD.Get_rank() == 0:
                        logger.dump_tabular()

        return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "timesteps_per_actorbatch": self.timesteps_per_actorbatch,
            "clip_param": self.clip_param,
            "entcoeff": self.entcoeff,
            "optim_epochs": self.optim_epochs,
            "optim_stepsize": self.optim_stepsize,
            "optim_batchsize": self.optim_batchsize,
            "lam": self.lam,
            "adam_epsilon": self.adam_epsilon,
            "schedule": self.schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


def traj_segment_generator(policy, env, horizon, max_ep_length=1000, inf_net=None, noise_dim=None, noise_scale=None, total_timestep=None,
                           reward_giver=None, gail=False, noise_decay=None, double_sivi=False, calculate_margin=False,
                           noise_num=None):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param policy: (MLPPolicy) the policy
    :param env: (Gym Environment) the environment
    :param horizon: (int) the number of timesteps to run per batch
    :param reward_giver: (TransitionClassifier) the reward predicter from obsevation and action
    :param gail: (bool) Whether we are using this generator for standard trpo or with gail
    :return: (dict) generator that returns a dict with the following keys:

        - observations: (np.ndarray) observations
        - rewards: (numpy float) rewards (if gail is used it is the predicted reward)
        - true_rewards: (numpy float) if gail is used it is the original reward
        - vpred: (numpy float) action logits
        - dones: (numpy bool) dones (is end of episode, used for logging)
        - episode_starts: (numpy bool)
            True if first timestep of an episode, used for GAE
        - actions: (np.ndarray) actions
        - nextvpred: (numpy float) next action logits
        - ep_rets: (float) cumulated current episode reward
        - ep_lens: (int) the length of the current episode
        - ep_true_rets: (float) the real environment reward
    """
    # Check when using GAIL
    assert not (gail and reward_giver is None), "You must pass a reward giver when using GAIL"

    # Initialize state variables
    if noise_dim:
        total_iteration = int(total_timestep / horizon)
        noise_scale_now = noise_scale
        ite_now = 0
    step = 0
    action = env.action_space.sample()  # not used, just so we have the datatype
    observation = env.reset()

    cur_ep_ret = 0  # return in current episode
    current_it_len = 0  # len of current iteration
    current_ep_len = 0  # len of current episode
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # Episode lengths

    noise_mar_lik = []

    # Initialize history arrays
    observations = np.array([observation for _ in range(horizon)])
    true_rewards = np.zeros(horizon, 'float32')
    rewards = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    episode_starts = np.zeros(horizon, 'bool')
    dones = np.zeros(horizon, 'bool')
    actions = np.array([action for _ in range(horizon)])
    states = policy.initial_state
    n_ac = action.shape[0]
    stdn_gens = np.array([np.zeros([1, n_ac]) for _ in range(horizon)])
    if noise_dim:
        ep_gens = np.array([np.zeros([1, noise_dim]) for _ in range(horizon)])
    episode_start = True  # marks if we're on first timestep of an episode
    done = False
    if noise_dim:
        if inf_net:
            if double_sivi:
                epsilon = np.random.normal(loc=0, scale=1,
                                           size=(1, noise_dim))  # * np.random.binomial(n=1,p=0.5,size = (1, noise_dim))
                epsilons = np.array([epsilon for _ in range(horizon)])
                epsilons = np.squeeze(epsilons, 1)
                noise, _, _, _, ep_gen = inf_net.step(observation.reshape(-1, *observation.shape), epsilon, states,
                                                      done)
                ep_gens = np.array([ep_gen for _ in range(horizon)])
                noises = np.array([noise for _ in range(horizon)])
                noises = np.squeeze(noises, 1)
            else:
                noise = inf_net.step(observation.reshape(-1, *observation.shape), states, done)
                noises = np.array([noise for _ in range(horizon)])
                noises = np.squeeze(noises, 1)
        else:
            noise = np.float32(
                np.random.normal(loc=0.0, scale=noise_scale_now, size=(1, noise_dim)) * np.random.binomial(n=1, p=0.5,
                                                                                                           size=(1,
                                                                                                                 noise_dim)))
            noises = np.array([noise for _ in range(horizon)])
            noises = np.squeeze(noises, 1)

    while True:
        if noise_dim:
            action, vpred, states, _, stdn_gen = policy.step(observation.reshape(-1, *observation.shape), noise, states,
                                                             done)
        else:
            action, vpred, states, _ = policy.step(observation.reshape(-1, *observation.shape), states, done)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if step > 0 and step % horizon == 0:

            if noise_dim:
                ite_now += 1
                yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "vpred": vpreds,
                    "actions": actions,
                    "nextvpred": vpred[0] * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_len,
                    "noises": noises,
                    "stdn_gen": stdn_gens,
                    "epsilons": epsilons,
                    "ep_gens": ep_gens,
                    "noise_mar_lik": noise_mar_lik
                }
                noise_scale_now = noise_scale * (1 - ite_now / total_iteration) ** noise_decay
                print("noise_scale is {}".format(noise_scale_now))
            else:
                yield {
                    "observations": observations,
                    "rewards": rewards,
                    "dones": dones,
                    "episode_starts": episode_starts,
                    "true_rewards": true_rewards,
                    "vpred": vpreds,
                    "actions": actions,
                    "nextvpred": vpred[0] * (1 - episode_start),
                    "ep_rets": ep_rets,
                    "ep_lens": ep_lens,
                    "ep_true_rets": ep_true_rets,
                    "total_timestep": current_it_len,
                }
            #            _, vpred, _, _ = policy.step(observation.reshape(-1, *observation.shape)) # TODO

            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            # Reset current iteration length
            current_it_len = 0
        i = step % horizon
        observations[i] = observation
        vpreds[i] = vpred[0]
        actions[i] = action[0]
        episode_starts[i] = episode_start

        if noise_dim:
            noises[i] = noise
            epsilons[i] = epsilon
            ep_gens[i] = ep_gen
            if inf_net:
                if double_sivi:
                    epsilon = np.random.normal(loc=0, scale=1, size=(
                    1, noise_dim))  # * np.random.binomial(n=1,p=0.5,size = (1, noise_dim))
                    noise, _, _, _, ep_gen = inf_net.step(observation.reshape(-1, *observation.shape), epsilon, states,
                                                          done)
                else:
                    noise = inf_net.step(observation.reshape(-1, *observation.shape), states, done)
            else:
                noise = np.float32(
                    np.random.normal(loc=0.0, scale=noise_scale_now, size=(1, noise_dim)) * np.random.binomial(n=1,
                                                                                                               p=0.5,
                                                                                                               size=(1,
                                                                                                                     noise_dim)))
            stdn_gens[i] = stdn_gen

        clipped_action = action
        # Clip the actions to avoid out of bound error
        if isinstance(env.action_space, gym.spaces.Box):
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)

        if gail:
            reward = reward_giver.get_reward(observation, clipped_action[0])
            observation, true_reward, done, info = env.step(clipped_action[0])
        else:
            observation, reward, done, info = env.step(clipped_action[0])
            true_reward = reward
        rewards[i] = reward
        true_rewards[i] = true_reward
        dones[i] = done
        episode_start = done

        cur_ep_ret += reward
        cur_ep_true_ret += true_reward
        current_it_len += 1
        current_ep_len += 1
        if done or current_ep_len >= max_ep_length:
            # Retrieve unnormalized reward if using Monitor wrapper
            maybe_ep_info = info.get('episode')
            if maybe_ep_info is not None:
                if not gail:
                    cur_ep_ret = maybe_ep_info['r']
                cur_ep_true_ret = maybe_ep_info['r']

            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(current_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            current_ep_len = 0

            observation = env.reset()

        step += 1
