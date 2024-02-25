# -*- coding:utf-8  -*-

import argparse
import os
from pathlib import Path
import sys
import torch

from RL_train.SACtrainer import MarketmakingTrainer
from env.chooseenv import make
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--save-dir", type=str, help="the dir to save log and model",
                    default='/devdata1/lhdata/kafang/SAC/win1')
parser.add_argument("--rl-step", type=float, help="steps for RL", default=1e8)
parser.add_argument("--critic-mlp-hidden-size", type=int, help="number of hidden units per layer in critic",
                    default=512)
parser.add_argument("--actor-mlp-hidden-size", type=int, help="number of hidden units per layer in actor",
                    default=512)
parser.add_argument("--batch-size", type=int, help="number of samples per minibatch", default=256)
parser.add_argument("--replay-buffer-capacity", type=int, help="replay buffer size", default=1e6)
parser.add_argument("--critic-lr", type=float, help="learning rate of critic", default=3e-4)
parser.add_argument("--actor-lr", type=float, help="learning rate of actor", default=1e-4)
parser.add_argument("--alpha-lr", type=float, help="learning rate of alpha", default=1e-4)
parser.add_argument("--log-alpha", type=float, help="initial alpha", default=0.0)
parser.add_argument("--target-entropy", type=float, help="target entropy in SAC", default=None)
parser.add_argument("--soft-tau", type=float, default=0.005)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--max-cache-len", type=int, default=1)
parser.add_argument("--state-keys", type=list, default=(
'signal0', 'signal1', 'signal2', 'ap0', 'bp0', 'ap1', 'bp1', 'ap2', 'bp2', 'ap3', 'bp3', 'ap4', 'bp4'))
parser.add_argument("--SRR", type=bool, default=False)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

env_type = "kafang_stock"
env = make(env_type, seed=None)
test_env = make(env_type, seed=None)
cache_single_dim = len(args.state_keys)
basic_state_dim = 2
if args.SRR:
    cache_single_dim += 5
    basic_state_dim += 1
state_dim = args.max_cache_len * cache_single_dim + basic_state_dim

agent = MarketmakingTrainer(state_dim=state_dim,
                              critic_mlp_hidden_size=args.critic_mlp_hidden_size,
                              actor_mlp_hidden_size=args.actor_mlp_hidden_size,
                              log_alpha=args.log_alpha,
                              critic_lr=args.critic_lr,
                              actor_lr=args.actor_lr,
                              alpha_lr=args.alpha_lr,
                              target_entropy=args.target_entropy,
                              gamma=args.gamma,
                              soft_tau=args.soft_tau,
                              env=env,
                              test_env=test_env,
                              replay_buffer_capacity=args.replay_buffer_capacity,
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                              max_cache_len=args.max_cache_len,
                              state_keys=args.state_keys
                              # device=torch.device("cpu")
                              )
agent.load_RL_part(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RL_part_500k.pt'))

def my_controller(observation, action_space, is_act_continuous=False):
    state = agent.extract_state(observation)
    state = torch.FloatTensor(state).to(agent.device)
    action = agent.actor.get_action(state)
    decoupled_action = agent.decouple_action(action=action, observation=observation)
    return decoupled_action

