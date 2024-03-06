import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from RL_train.SACAdjRule2 import MarketmakingTrainer
import torch
import torch.nn as nn
from RL_train.network import MultiQNetwork, GaussianPolicyNetwork, ReplayBuffer
import copy
from typing import Dict
from torch import optim
import ray
from torch.utils.tensorboard import SummaryWriter
from env.chooseenv import make
from collections import deque
import numpy as np
import torch.nn.functional as F
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.data import from_items
from pathlib import Path
from ray import train

class NeuralNetworkForSAC(nn.Module):
    def __init__(self, state_dim, action_dim, critic_mlp_hidden_size, actor_mlp_hidden_size, log_alpha):
        super(NeuralNetworkForSAC, self).__init__()
        self.critic = MultiQNetwork(state_dim=state_dim,
                                    action_dim=action_dim,
                                    hidden_size=critic_mlp_hidden_size)

        self.actor = GaussianPolicyNetwork(state_dim=state_dim,
                                           action_dim=action_dim,
                                           hidden_dim=actor_mlp_hidden_size)

        self.register_parameter('log_alpha', nn.Parameter(torch.FloatTensor([log_alpha])))
        # self.log_alpha = torch.FloatTensor([log_alpha])
        # self.log_alpha.requires_grad = True


class ObservesCollect:
    def __init__(self, maxlen=5, keys=('signal0', 'signal1', 'signal2'), SRR=False):
        self.cache = deque(maxlen=maxlen)
        self.SRR = SRR
        self.keys = keys

    def calculate_sum_residual_ratio(self, all_observes):
        ap0 = all_observes['ap0']
        ap1 = all_observes['ap1']
        ap2 = all_observes['ap2']
        ap3 = all_observes['ap3']
        ap4 = all_observes['ap4']
        bp0 = all_observes['bp0']
        bp1 = all_observes['bp1']
        bp2 = all_observes['bp2']
        bp3 = all_observes['bp3']
        bp4 = all_observes['bp4']
        mid_price = (ap0 + bp0) / 2
        ar0 = (ap0 - mid_price) * all_observes['av0']
        ar1 = (ap1 - mid_price) * all_observes['av1']
        ar2 = (ap2 - mid_price) * all_observes['av2']
        ar3 = (ap3 - mid_price) * all_observes['av3']
        ar4 = (ap4 - mid_price) * all_observes['av4']
        br0 = (mid_price - bp0) * all_observes['bv0']
        br1 = (mid_price - bp1) * all_observes['bv1']
        br2 = (mid_price - bp2) * all_observes['bv2']
        br3 = (mid_price - bp3) * all_observes['bv3']
        br4 = (mid_price - bp4) * all_observes['bv4']
        f_sum_residual_ratio_0 = ar0 / br0
        f_sum_residual_ratio_1 = (ar1 + ar0) / (br1 + br0)
        f_sum_residual_ratio_2 = (ar2 + ar1 + ar0) / (br2 + br1 + br0)
        f_sum_residual_ratio_3 = (ar3 + ar2 + ar1 + ar0) / (br3 + br2 + br1 + br0)
        f_sum_residual_ratio_4 = (ar4 + ar3 + ar2 + ar1 + ar0) / (br4 + br3 + br2 + br1 + br0)
        return f_sum_residual_ratio_0, f_sum_residual_ratio_1, f_sum_residual_ratio_2, f_sum_residual_ratio_3, f_sum_residual_ratio_4

    def extract_state(self, all_observes) -> np.ndarray:
        # import pdb
        # pdb.set_trace()
        if isinstance(all_observes, list):
            all_observes = all_observes[0]['observation']
        else:
            all_observes = all_observes['observation']
        if len(self.cache) == 0:
            for i in range(self.cache.maxlen):
                self.cache.append(all_observes)
        self.cache.append(all_observes)

        # fsrr0, fsrr1, fsrr2, fsrr3, fsrr4 = self.calculate_sum_residual_ratio(all_observes)
        history = {key: [] for key in self.keys}
        for i in range(0, self.cache.maxlen):
            for key in self.keys:
                if 'p' in key:
                    history[key].append(self.cache[i][key] / self.cache[i]['ap0_t0'])
                else:
                    history[key].append(self.cache[i][key])
        for key in self.keys:
            history[key] = np.array(history[key])
        # state = np.concatenate((state, fsrr_history_np, signal0_history_np, signal1_history_np, signal2_history_np), axis=0)
        state = list(history.values())  # 注意必须在python3.6+之后字典的键值才是插入顺序，才能按照固定顺序直接转为list
        position = np.array([all_observes['code_net_position']])
        state.append(position)
        cost = np.array(
            all_observes['code_cash_pnl'] / position / all_observes['ap0_t0']) if position != 0 else np.array([10])
        state.append(cost)
        if self.SRR:
            fsrr_history = []
            for i in range(0, self.cache.maxlen):
                if i == self.cache.maxlen - 1:
                    fsrr_list = self.calculate_sum_residual_ratio(all_observes)
                    fsrr_history += fsrr_list
                else:
                    fsrr_history.append(self.calculate_sum_residual_ratio(self.cache[i])[0])
            state += fsrr_history

        state = np.concatenate(state, axis=0)
        # print(state)
        return state

    def clear(self):
        self.cache.clear()


def decouple_action(action: np.ndarray, observation: dict, action_threshold: float) -> list:
    """
    根据action和当前的订单簿生成具体的订单
    :param action: actor输出的动作
    :param observation: 观测值，其中'bp0'到'bp4'表示从高到低的买家报价，'bv0'到'bv4'表示对应的买家报价手数，
                        'ap0'到'ap4'表示从低到高的卖家报价，'av0'到'av4'表示对应的卖家报价手数，
    :return: 具体的订单
    """
    if isinstance(observation, list):
        observation = observation[0]
    if 'observation' in observation.keys():
        observation = observation['observation']

    obs = observation
    if obs['signal0'] > action_threshold:
        # Long opening
        price = (obs['ap0'] + obs['bp0']) / 2 * (1 + (obs['signal0'] * 0.0002 * action[0] + obs['signal1'] * 0.0002 * action[1] + obs['signal2'] * 0.0002 * action[2]))
        if obs['ap0'] <= price:
            side = [1, 0, 0]
            volumn = min(obs['av0'], 300. - obs['code_net_position'])
            price = obs['ap0']
        else:
            side = [0, 1, 0]
            volumn = 0.
            price = 0.
    elif obs['signal0'] < -action_threshold:

        # Short opening
        price = (obs['ap0'] + obs['bp0']) / 2 * (1 + (obs['signal0'] * 0.0002 * action[0] + obs['signal1'] * 0.0002 * action[1] + obs['signal2'] * 0.0002 * action[2]))
        if obs['bp0'] >= price:
            side = [0, 0, 1]
            volumn = min(obs['bv0'], 300. + obs['code_net_position'])
            price = obs['bp0']
        else:
            side = [0, 1, 0]
            volumn = 0.
            price = 0.
    else:
        side = [0, 1, 0]
        volumn = 0.
        price = 0.

    return [side, volumn, price]

    # bid_price = (observation['bp0'] + observation['ap0']) / 2 * (action[0] * 0.0005 + 1) # / (1 + 0.00007 + 0.00001)  # 计划买入价
    # ask_price = (observation['bp0'] + observation['ap0']) / 2 * (action[0] * 0.0005 + 1) # * (1 + 0.00007 + 0.00001)  # 计划卖出价
    # # 仅考虑以bp0价位卖出，不考虑bp1、bp2、bp3、bp4
    # if ask_price <= observation[f'bp0']:
    #     ask_volume = observation[f'bv0']
    #     ask_price = observation[f'bp0']
    # else:
    #     ask_volume = 0
    # ask_volume = min(ask_volume, 300 + observation['code_net_position'], 20)
    # if ask_volume > 0:
    #     return [[0, 0, 1], float(ask_volume), ask_price]  # 以ask_price价格卖出ask_volume手


    # if bid_price >= observation[f'ap0']:
    #     bid_volume = observation[f'av0']
    #     bid_price = observation[f'ap0']
    # else:
    #     bid_volume = 0
    # bid_volume = min(bid_volume, 300 - observation['code_net_position'], 20)
    # if bid_volume > 0:
    #     return [[1, 0, 0], float(bid_volume), bid_price]  # 以bid_price价格买入bid_volume手

    # return [[0, 1, 0], 0., 0.]  # 什么都不做


def train_func_per_worker(config: Dict):
    # state_dim = config['state_dim']
    action_dim = config['action_dim']
    critic_mlp_hidden_size = config['critic_mlp_hidden_size']
    actor_mlp_hidden_size = config['actor_mlp_hidden_size']
    init_log_alpha = config['log_alpha']
    critic_lr = config['critic_lr']
    actor_lr = config['actor_lr']
    alpha_lr = config['alpha_lr']
    target_entropy = config['target_entropy'] if config['target_entropy'] is not None else -action_dim
    gamma = config['gamma']
    soft_tau = config['soft_tau']
    save_dir = config['save_dir']
    rl_step = int(config['rl_step'])
    imitate_step = config['imitate_step']
    batch_size = config['batch_size']
    sample_num = config['sample_num']
    replay_buffer_capacity = config['replay_buffer_capacity']
    save_dir = os.path.join(save_dir, str(torch.distributed.get_rank()))

    # date_list = config['date_list']
    for date_list in ray.train.get_dataset_shard("dateList").iter_batches():
        date_list = date_list['item'].tolist()
        print(f'{date_list}')
    max_cache_len = config['max_cache_len']
    action_threshold = config['action_threshold']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_length = config['test_length']
    state_keys = config['state_keys']
    SRR = config['SRR']
    cache_single_dim = len(state_keys)
    basic_state_dim = 2
    if SRR:
        cache_single_dim += 5
        basic_state_dim += 1
    state_dim = max_cache_len * cache_single_dim + basic_state_dim

    model = NeuralNetworkForSAC(state_dim, action_dim, critic_mlp_hidden_size, actor_mlp_hidden_size, init_log_alpha).to(device)
    model = ray.train.torch.prepare_model(model, parallel_strategy=None)
    target_critic = copy.deepcopy(model.critic)
    critic_optimizer = optim.Adam(model.critic.parameters(), lr=critic_lr)
    actor_optimizer = optim.Adam(model.actor.parameters(), lr=actor_lr)
    alpha_optimizer = optim.Adam([model.log_alpha], lr=alpha_lr)

    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    env_type = "kafang_stock"
    env = make(env_type, dateList=date_list)
    test_env = make(env_type, dateList=date_list)

    if not os.path.exists(os.path.join(save_dir, 'log')):
        os.makedirs(os.path.join(save_dir, 'log'))
    if not os.path.exists(os.path.join(save_dir, 'models')):
        os.makedirs(os.path.join(save_dir, 'models'))

    logger_writer = SummaryWriter(os.path.join(save_dir, 'log'))

    observes_collect = ObservesCollect(maxlen=max_cache_len, keys=state_keys)
    test_observes_collect = ObservesCollect(maxlen=max_cache_len, keys=state_keys)
    env.reset()
    test_env.reset()

    for i in range(rl_step + 1):
        for _ in range(sample_num):
            state = observes_collect.extract_state(env.all_observes)
            action = model.actor.get_action(state=torch.FloatTensor(state).to(device))
            decoupled_action = decouple_action(action=action, observation=env.all_observes[0], action_threshold=action_threshold)
            all_observes, reward, done, info_before, info_after = env.step([decoupled_action])
            next_state = observes_collect.extract_state(all_observes)
            replay_buffer.push(state, action, reward, next_state, done)
            if done:  # 如果单只股票/单日/全部数据结束，则重置历史观测数据
                observes_collect.clear()
            if env.done:
                all_observes = env.reset()
                state = observes_collect.extract_state(all_observes)
            else:
                state = next_state

        state, action, reward, next_state, done = replay_buffer.sample(min(batch_size, len(replay_buffer)))
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(-1)  # shape=(batch_size, 1)
        done = torch.FloatTensor(done).to(device)
        if len(done.shape) == 1:
            done = done.unsqueeze(-1)  # shape=(batch_size, 1)
        with torch.no_grad():
            next_action, next_log_prob, _, _, _ = model.actor.evaluate(next_state)
            next_q_value = torch.min(target_critic(next_state, next_action), dim=0)[
                0]  # - self.alpha * next_log_prob
            target_q_value = reward + (1 - done) * gamma * next_q_value

        critic_loss = model.critic.qLoss(target_q_value, state, action, F.mse_loss)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        for target_param, param in zip(target_critic.parameters(), model.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        state, _, _, _, _ = replay_buffer.sample(min(batch_size, len(replay_buffer)))
        state = torch.FloatTensor(state).to(device)

        new_action, log_prob, z, mean, log_std = model.actor.evaluate(state)
        expected_new_q_value = torch.min(model.critic(state, new_action), dim=0)[0]

        # log_prob_target = expected_new_q_value - expected_value
        # actor_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        actor_loss = (model.log_alpha.exp() * log_prob - expected_new_q_value).mean()

        std_loss = 1e-3 * log_std.pow(2).mean()

        actor_loss += std_loss
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        with torch.no_grad():
            new_action, log_prob, _, _, _ = model.actor.evaluate(state)
            alpha_loss = log_prob + target_entropy
        entropy = -alpha_loss.mean() + target_entropy
        alpha_loss = -model.log_alpha.exp() * alpha_loss.mean()
        alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_optimizer.step()

        logger = {
            'critic_loss': critic_loss.item(),
            'reward': reward.mean().item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'entropy': entropy.item(),
            'alpha': model.log_alpha.exp().item(),
            'log_prob': log_prob.mean().item(),
            'expected_new_q_value': expected_new_q_value.mean().item()
        }

        if i % 10000 == 0:
            reward_sum = 0
            for _ in range(test_length):
                all_observes = test_env.all_observes
                state = test_observes_collect.extract_state(all_observes)
                # state = self.extract_state(all_observes)
                # while not self.test_env.done:
                state = torch.FloatTensor(state).to(device)
                action = model.actor.get_action(state, random_flag=False)
                decoupled_action = decouple_action(action=action, observation=all_observes[0], action_threshold=action_threshold)
                all_observes, reward, done, info_before, info_after = test_env.step([decoupled_action])
                reward_sum += reward
                if done:
                    test_observes_collect.clear()
                if test_env.done:
                    test_env.reset()
                # state = next_state
            logger['test_reward_mean'] = reward_sum

            for key in logger.keys():
                logger_writer.add_scalar(key, logger[key], i)
            torch.save({'actor': model.actor.state_dict(),
                        'critic': model.critic.state_dict(),
                        'target_critic': target_critic.state_dict(),
                        'log_alpha': model.log_alpha,
                        },
                       os.path.join(save_dir, 'models', 'RL_part_%dk.pt' % (i / 1000)))


def main(num_workers=4, use_gpu=True):
    train_config = {
                    "SRR": False,
                    "action_threshold": 0.8,
                    "action_dim": 3,
                    "actor_lr": 0.00001,
                    "actor_mlp_hidden_size": 256,
                    "alpha_lr": 0.00001,
                    "batch_size": 1024,
                    "critic_lr": 0.00003,
                    "critic_mlp_hidden_size": 256,
                    "gamma": 0.99,
                    "imitate_step": 0,
                    "log_alpha": 0.0,
                    "max_cache_len": 1,
                    "replay_buffer_capacity": 1000000.0,
                    "rl_step": 100000000.0,
                    "sample_num": 5,
                    "save_dir": "/data/lhdata/kafang/output_SAC_rule2/ray",
                    "seed": 1,
                    "soft_tau": 0.005,
                    "state_keys": [
                        "signal0",
                        "signal1",
                        "signal2",
                        "ap0",
                        "bp0",
                        "ap1",
                        "bp1",
                        "ap2",
                        "bp2",
                        "ap3",
                        "bp3",
                        "ap4",
                        "bp4"
                    ],
                    "target_entropy": None,
                    'test_length': 10000
    }

    CURRENT_PATH = str(Path(__file__).resolve().parent.parent)
    stock_path = os.path.join(CURRENT_PATH, 'env/stock_raw')
    signal_file_original_rootpath = os.path.join(stock_path, 'data')
    dateList = [name for name in os.listdir(signal_file_original_rootpath) if
                        os.path.isdir(os.path.join(signal_file_original_rootpath, name))]
    dateList.sort()
    dateList = from_items(dateList)

    # Configure computation resources
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    # Initialize a Ray TorchTrainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        datasets={'dateList': dateList}
    )

    # [4] Start distributed training
    # Run `train_func_per_worker` on all workers
    # =============================================
    result = trainer.fit()
    print(f"Training result: {result}")


if __name__ == '__main__':
    main()
