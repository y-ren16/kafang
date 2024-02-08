# -*- coding:utf-8  -*-

import argparse
import os
from pathlib import Path
import sys
import torch

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from DQNTrainer import DQNTrainer

cache_single_dim = 5
basic_state_dim = 3
state_dim = 1 * cache_single_dim + 2

agent = DQNTrainer(state_dim=state_dim,
                   critic_mlp_hidden_size=512,
                   critic_lr=3e-4,
                   gamma=0.99,
                   soft_tau=0.005,
                   env=None,
                   test_env=None,
                   replay_buffer_capacity=1e6,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   max_cache_len=1
                   )
agent.load_RL_part(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RL_part_1500k.pt'))


def my_controller(observation, action_space, is_act_continuous=False):
    state = agent.extract_state(observation)
    state = torch.FloatTensor(state).to(agent.device)
    action = agent.get_action(state)
    decoupled_action = agent.decouple_action(action=action, observation=observation)
    return decoupled_action
