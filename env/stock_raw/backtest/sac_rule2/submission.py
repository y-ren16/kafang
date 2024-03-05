# -*- coding:utf-8  -*-

import argparse
import os
from pathlib import Path
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from SACAdjRule2 import MarketmakingTrainer
import pdb

agent1 = MarketmakingTrainer(state_dim=15,
                            critic_mlp_hidden_size=512,
                            actor_mlp_hidden_size=512,
                            log_alpha=0,
                            critic_lr=3e-4,
                            actor_lr=1e-4,
                            alpha_lr=1e-4,
                            target_entropy=-1,
                            gamma=0.99,
                            soft_tau=0.005,
                            env=None,
                            test_env=None,
                            replay_buffer_capacity=1e6,
                            device=torch.device("cpu"),
                            max_cache_len=1,
                            state_keys=(
                                'signal0', 'signal1', 'signal2', 'ap0', 'bp0', 'ap1', 'bp1', 'ap2', 'bp2', 'ap3', 'bp3',
                                'ap4', 'bp4'),
                            action_threshold=.6
                            # device=torch.device("cpu")
                            )
agent1.load_RL_part(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RL_part_1000k.pt'))

agent2 = MarketmakingTrainer(state_dim=15,
                            critic_mlp_hidden_size=512,
                            actor_mlp_hidden_size=512,
                            log_alpha=0,
                            critic_lr=3e-4,
                            actor_lr=1e-4,
                            alpha_lr=1e-4,
                            target_entropy=-1,
                            gamma=0.99,
                            soft_tau=0.005,
                            env=None,
                            test_env=None,
                            replay_buffer_capacity=1e6,
                            device=torch.device("cpu"),
                            max_cache_len=1,
                            state_keys=(
                                'signal0', 'signal1', 'signal2', 'ap0', 'bp0', 'ap1', 'bp1', 'ap2', 'bp2', 'ap3', 'bp3',
                                'ap4', 'bp4'),
                            action_threshold=.6
                            # device=torch.device("cpu")
                            )
agent2.load_RL_part(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RL_part_1000k_day15.pt'))



def my_controller(observation, action_space, is_act_continuous=False):
    state = agent1.extract_state(observation)
    state = torch.FloatTensor(state).to(agent1.device)
    action1 = agent1.actor.get_action(state, random_flag=False)
    action2 = agent2.actor.get_action(state, random_flag=False)
    decoupled_action = agent1.decouple_action(action=(action1 + action2) / 2, observation=observation)
    # decoupled_action = agent1.decouple_action(action=action1, observation=observation)
    return decoupled_action
