# -*- coding:utf-8  -*-

import argparse
import os
from pathlib import Path
import sys
import torch

from RL_train.trainer import MarketmakingTrainer
from env.chooseenv import make
import pdb

parser = argparse.ArgumentParser()
args = parser.parse_args()
parser.add_argument('--model_path', type=str, default='./test/models/RL_part_4760k.pt')
parser.add_argument('--state_dim', type=int, default=23)    
parser.add_argument("--save-dir", type=str, help="the dir to save log and model", default='/devdata1/lhdata/kafang/test')
parser.add_argument("--rl-step", type=float, help="steps for RL", default=1e8)
parser.add_argument("--critic-mlp-hidden-size", type=int, help="number of hidden units per layer in critic", default=512)
parser.add_argument("--actor-mlp-hidden-size", type=int, help="number of hidden units per layer in actor", default=512)
parser.add_argument("--batch-size", type=int, help="number of samples per minibatch", default=256)
parser.add_argument("--replay-buffer-capacity", type=int, help="replay buffer size", default=1e6)
parser.add_argument("--critic-lr", type=float, help="learning rate of critic", default=3e-4)
parser.add_argument("--actor-lr", type=float, help="learning rate of actor", default=1e-4)
parser.add_argument("--alpha-lr", type=float, help="learning rate of alpha", default=1e-4)
parser.add_argument("--log-alpha", type=float, help="initial alpha", default=0.0)
parser.add_argument("--target-entropy", type=float, help="target entropy in SAC", default=None)
parser.add_argument("--soft-tau", type=float, default=0.005)
parser.add_argument("--gamma", type=float, default=0.99)
args = parser.parse_args()

env_type = "kafang_stock"
env = make(env_type, seed=None)
test_env = make(env_type, seed=None)
agent = MarketmakingTrainer(state_dim=args.state_dim,
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
                        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
agent.load_RL_part(args.model_path)

def my_controller(observation, action_space, is_act_continuous=False):
    state = agent.extract_state(observation)
    state = torch.FloatTensor(state).to(agent.device)
    action = agent.actor.get_action(state)
    decoupled_action = agent.decouple_action(action=action, observation=observation)
    return decoupled_action

