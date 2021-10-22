import argparse
import gym
import numpy as np
import os
import torch

from agents.idac import IDAC
from trainer import Trainer
from utils import utils
from utils.logger import logger, setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--ExpID", default='Exp1', type=str)  # Experiment ID
    parser.add_argument('--device', default='cpu', type=str)  # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="HalfCheetah-v2", type=str)  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--num_iters_per_epoch", default=1000, type=int)
    parser.add_argument("--start_timesteps", default=10000, type=int)
    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)  # Mini batch size for networks
    parser.add_argument("--actor_lr", default=3e-4, type=float)
    parser.add_argument("--critic_lr", default=3e-4, type=float)
    ### IDAC Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--noise_dim", default=5, type=int)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--target_entropy", default=None, type=float)
    parser.add_argument("--hidden_sizes", default=1, type=int, help="1: [400, 300]; 2: [256, 256, 64]")
    parser.add_argument("--pi_bn", default=0, type=int)
    parser.add_argument("--num_quantiles", default=32, type=int)
    parser.add_argument("--use_automatic_entropy_tuning", default=True, type=int)

    args = parser.parse_args()
    # d4rl.set_dataset_path('/datasets')

    if args.hidden_sizes == 1:
        args.hidden_sizes = [256, 256]
    elif args.hidden_sizes == 2:
        args.hidden_sizes = [400, 300]
    elif args.hidden_sizes == 99: # for test
        args.hidden_sizes = [10, 10]
        

    expl_env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)

    expl_env.seed(args.seed)
    eval_env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = expl_env.observation_space.shape[0]
    action_dim = expl_env.action_space.shape[0]
    max_action = float(expl_env.action_space.high[0])
    if args.noise_dim is None:
        args.noise_dim = min(10, (state_dim + action_dim) // 2)
    output_dir = os.path.join("results", args.ExpID)

    # Setup Logging
    file_name = f"{args.env_name}|{args.ExpID}|alpha{args.alpha}|noise_dim{args.noise_dim}|num_qtl{args.num_quantiles}|{args.seed}"
    results_dir = os.path.join(output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    # if os.path.exists(os.path.join(results_dir, 'variant.json')):
    #     raise AssertionError("Experiment under this setting has been done!")
    variant = vars(args)
    variant.update(version=f"IDAC with num_quantiles {args.num_quantiles}")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}")

    agent = IDAC(state_dim=state_dim,
                 action_dim=action_dim,
                 noise_dim=args.noise_dim,
                 max_action=max_action,
                 device=args.device,
                 hidden_sizes=args.hidden_sizes,
                 discount=args.discount,
                 tau=args.tau,
                 actor_lr=args.actor_lr,
                 critic_lr=args.critic_lr,
                 batch_size=args.batch_size,
                 pi_bn=args.pi_bn,
                 num_quantiles=args.num_quantiles,
                 target_entropy=None,
                 alpha=args.alpha,
                 use_automatic_entropy_tuning=args.use_automatic_entropy_tuning)

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.device)

    trainer = Trainer(agent,
                      expl_env,
                      eval_env,
                      replay_buffer,
                      args.device,
                      start_timesteps=args.start_timesteps)

    trainer.train(num_epochs=args.num_epochs,
                  num_iters_per_epoch=args.num_iters_per_epoch)
