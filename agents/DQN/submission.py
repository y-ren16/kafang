# -*- coding:utf-8  -*-

import argparse
import os
from pathlib import Path
import sys
import torch

from RL_train.DQNTrainer import DQNTrainer
from env.chooseenv import make
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./DQN/models/RL_part_1500k.pt')
parser.add_argument("--save-dir", type=str, help="the dir to save log and model",
                    default='/devdata1/lhdata/kafang/DQN')
parser.add_argument("--critic-mlp-hidden-size", type=int, help="number of hidden units per layer in critic",
                    default=512)
parser.add_argument("--critic-lr", type=float, help="learning rate of critic", default=3e-4)
parser.add_argument("--batch-size", type=int, help="number of samples per minibatch", default=256)
parser.add_argument("--replay-buffer-capacity", type=int, help="replay buffer size", default=1e6)
parser.add_argument("--target-entropy", type=float, help="target entropy in SAC", default=None)
parser.add_argument("--soft-tau", type=float, default=0.005)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--max-cache-len", type=int, default=1)
args = parser.parse_args()

env_type = "kafang_stock"
env = make(env_type, seed=None)
test_env = make(env_type, seed=None)
cache_single_dim = 5
basic_state_dim = 3
state_dim = args.max_cache_len * cache_single_dim + 2

agent = DQNTrainer(state_dim=state_dim,
                     critic_mlp_hidden_size=args.critic_mlp_hidden_size,
                     critic_lr=args.critic_lr,
                     gamma=args.gamma,
                     soft_tau=args.soft_tau,
                     env=env,
                     test_env=test_env,
                     replay_buffer_capacity=args.replay_buffer_capacity,
                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                     max_cache_len=args.max_cache_len
                     )
agent.load_RL_part(args.model_path)


def my_controller(observation, action_space, is_act_continuous=False):
    state = agent.extract_state(observation)
    state = torch.FloatTensor(state).to(agent.device)
    action = agent.get_action(state)
    decoupled_action = agent.decouple_action(action=action, observation=observation)
    return decoupled_action
