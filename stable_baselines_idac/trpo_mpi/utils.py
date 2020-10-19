import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv



def traj_segment_generator(policy, env, horizon, inf_net=None, noise_dim=None, noise_scale=None, total_timestep=None, reward_giver=None, gail=False, noise_decay=None, double_sivi=False, calculate_margin=False, noise_num=None):
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
    current_ep_len = 0 # len of current episode
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
    stdn_gens = np.array([np.zeros([1,n_ac]) for _ in range(horizon)])
    if noise_dim:
        ep_gens = np.array([np.zeros([1,noise_dim]) for _ in range(horizon)])
    episode_start = True  # marks if we're on first timestep of an episode
    done = False
    if noise_dim:
        if inf_net:
            if double_sivi:
                epsilon = np.random.normal(loc=0,scale=1,size = (1, noise_dim)) #* np.random.binomial(n=1,p=0.5,size = (1, noise_dim))
                epsilons = np.array([epsilon for _ in range(horizon)])
                epsilons = np.squeeze(epsilons,1)
                noise,_,_,_,ep_gen = inf_net.step(observation.reshape(-1, *observation.shape), epsilon, states, done)
                ep_gens = np.array([ep_gen for _ in range(horizon)])
                noises = np.array([noise for _ in range(horizon)])
                noises = np.squeeze(noises,1)
            else:
                noise = inf_net.step(observation.reshape(-1, *observation.shape), states, done)
                noises = np.array([noise for _ in range(horizon)])
                noises = np.squeeze(noises,1)
        else:    
            noise = np.float32(np.random.normal(loc = 0.0,scale = noise_scale_now,size = (1, noise_dim)) * np.random.binomial(n=1,p=0.5,size=(1,noise_dim)))
            noises = np.array([noise for _ in range(horizon)])
            noises = np.squeeze(noises,1)

    while True:
        if noise_dim:
            action, vpred, states, _, stdn_gen = policy.step(observation.reshape(-1, *observation.shape), noise, states, done)
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
                noise_scale_now =  noise_scale * (1 - ite_now / total_iteration) ** noise_decay
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
                    epsilon = np.random.normal(loc=0,scale=1,size = (1, noise_dim)) #* np.random.binomial(n=1,p=0.5,size = (1, noise_dim))
                    noise,_,_,_,ep_gen = inf_net.step(observation.reshape(-1, *observation.shape), epsilon, states, done)
                else:
                    noise = inf_net.step(observation.reshape(-1, *observation.shape), states, done)
            else:    
                noise = np.float32(np.random.normal(loc = 0.0,scale = noise_scale_now,size = (1, noise_dim)) * np.random.binomial(n=1,p=0.5,size=(1,noise_dim)))
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
        if done:
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
            if not isinstance(env, VecEnv):
                observation = env.reset()
        step += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

    :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
    :param gamma: (float) Discount factor
    :param lam: (float) GAE factor
    """
    # last element is only used for last vtarg, but we already zeroed it if last new = 1
    episode_starts = np.append(seg["episode_starts"], False)
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    rew_len = len(seg["rewards"])
    seg["adv"] = np.empty(rew_len, 'float32')
    rewards = seg["rewards"]
    lastgaelam = 0
    for step in reversed(range(rew_len)):
        nonterminal = 1 - float(episode_starts[step + 1])
        delta = rewards[step] + gamma * vpred[step + 1] * nonterminal - vpred[step]
        seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def flatten_lists(listoflists):
    """
    Flatten a python list of list

    :param listoflists: (list(list))
    :return: (list)
    """
    return [el for list_ in listoflists for el in list_]
