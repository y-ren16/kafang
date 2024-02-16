import torch
from torch import optim
from RL_train.basicDiscreteTrainer import basicDiscreteTrainer
from RL_train.network import DiscretePolicyNetwork
import copy
from env.stock_raw.utils import Order
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import math
import random
from collections import deque
import pdb

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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
                    history[key].append(self.cache[i][key]/self.cache[i]['ap0_t0'])
                else:
                    history[key].append(self.cache[i][key])
        for key in self.keys:
            history[key] = np.array(history[key])
        # state = np.concatenate((state, fsrr_history_np, signal0_history_np, signal1_history_np, signal2_history_np), axis=0)
        state = list(history.values())  # 注意必须在python3.6+之后字典的键值才是插入顺序，才能按照固定顺序直接转为list
        position = np.array([all_observes['code_net_position']])
        state.append(position)
        cost = np.array(all_observes['code_cash_pnl']/position/all_observes['ap0_t0']) if position != 0 else np.array([10])
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

class DQNTrainer(basicDiscreteTrainer):
    def __init__(self, state_dim, critic_mlp_hidden_size, critic_lr, gamma, soft_tau,
                 env, test_env, replay_buffer_capacity, device, priorities_coefficient=1, priorities_bias=1,
                 max_cache_len=5, state_keys=('signal0', 'signal1', 'signal2', 'ap0', 'bp0')):
        action_dim = 11  # 0-4：买入，5：不做：6-10：卖出
        super().__init__(state_dim, action_dim, critic_mlp_hidden_size, critic_lr, gamma, soft_tau, env, test_env,
                         replay_buffer_capacity, device, priorities_coefficient, priorities_bias)

        self.observes_collect = ObservesCollect(maxlen=max_cache_len, keys=state_keys)
        self.test_observes_collect = ObservesCollect(maxlen=max_cache_len, keys=state_keys)

        self.max_cache_len = max_cache_len
        self.state_keys = state_keys

    def extract_state(self, all_observes) -> np.ndarray:
        return self.observes_collect.extract_state(all_observes)
        # """
        # 将原始的观测值转换为向量化的状态
        # :param all_observes: 原始观测值
        # :return: 向量化的状态
        # """
        # if isinstance(all_observes, list):
        #     state = all_observes[0]['observation']
        # else:
        #     state = all_observes['observation']
        # ap0_t0 = state['ap0_t0']
        # # state包含给定因子、买单价格、买单手数、卖单价格、卖单手数
        # state = np.array([state['signal0'], state['signal1'], state['signal2']])
        # # state[3:8] = state[3:8] / ap0_t0
        # # state[13:18] = state[13:18] / ap0_t0
        # return state

    def decouple_action(self, action: int, observation: dict) -> list:
        """
        根据action和当前的订单簿生成具体的订单
        :param action: actor输出的动作
        :param observation: 观测值，其中'bp0'到'bp4'表示从高到低的买家报价，'bv0'到'bv4'表示对应的买家报价手数，
                            'ap0'到'ap4'表示从低到高的卖家报价，'av0'到'av4'表示对应的卖家报价手数，
        :return: 具体的订单
        """
        if isinstance(action, float):
            action = int(action)
        if isinstance(observation, list):
            observation = observation[0]
        if 'observation' in observation.keys():
            observation = observation['observation']

        if action < 5:
            bid_price = observation[f'ap{action}']
            bid_volume = 0
            for i in range(action + 1):
                bid_volume += observation[f'av{i}']
            bid_volume = min(bid_volume, 300 - observation['code_net_position'], 20)
            if bid_volume > 0:
                return [[1, 0, 0], float(bid_volume), bid_price]  # 以bid_price价格买入bid_volume手
        elif action > 5:
            ask_price = observation[f'bp{action - 6}']
            ask_volume = 0
            for i in range(action - 5):
                ask_volume += observation[f'bv{i}']
            ask_volume = min(ask_volume, 300 + observation['code_net_position'], 20)
            if ask_volume > 0:
                return [[0, 0, 1], float(ask_volume), ask_price]  # 以ask_price价格卖出ask_volume手

        return [[0, 1, 0], 0., 0.]  # 什么都不做

    def RL_train(self, save_dir, rl_step, batch_size, init_noise=0.5, noise_dumping=0.99):
        if not os.path.exists(os.path.join(save_dir, 'log')):
            os.makedirs(os.path.join(save_dir, 'log'))
        if not os.path.exists(os.path.join(save_dir, 'models')):
            os.makedirs(os.path.join(save_dir, 'models'))

        logger_writer = SummaryWriter(os.path.join(save_dir, 'log'))

        all_observes = self.env.reset()
        self.test_env.reset()
        state = self.extract_state(all_observes)
        noise = init_noise
        for i in range(int(rl_step) + 1):
            # print(i)
            # if i == -199076:
            #     print(i)
            if random.random() < noise:
                action = random.randint(0, self.action_dim - 1)
            else:
                action = self.get_action(state=torch.FloatTensor(state).to(self.device)).item()
            decoupled_action = self.decouple_action(action=action, observation=all_observes[0])
            all_observes, reward, done, info_before, info_after = self.env.step([decoupled_action])
            next_state = self.extract_state(all_observes)
            self.replay_buffer.push(state, action, reward, next_state, done)
            if self.env.done:
                all_observes = self.env.reset()
                state = self.extract_state(all_observes)
            else:
                state = next_state

            logger = {}
            logger.update(self.critic_train_step(batch_size))
            self.soft_update_target_critic()
            if i % 10000 == 0:
                logger['test_reward_mean'] = self.RL_test(test_length=10000)
                if i > 0:
                    logger['noise'] = noise
                    noise *= noise_dumping
                for key in logger.keys():
                    logger_writer.add_scalar(key, logger[key], i)
                self.save_RL_part(os.path.join(save_dir, 'models', 'RL_part_%dk.pt' % (i / 1000)))
                torch.save(self.replay_buffer, os.path.join(save_dir, 'models', 'replay_buffer_%dk.pt' % (i / 1000)))
                info = 'step: %dk' % (i / 1000)
                # info = 'step: %d' % (i)
                for key in logger.keys():
                    info += ' | %s: %.3f' % (key, logger[key])
                print(info)

    def RL_test(self, test_length=100000):
        reward_sum = 0
        # init_position =
        for _ in range(test_length):
            all_observes = self.test_env.all_observes
            state = self.test_observes_collect.extract_state(all_observes)
            # state = self.extract_state(all_observes)
            # while not self.test_env.done:
            state = torch.FloatTensor(state).to(self.device)
            action = self.get_action(state)
            decoupled_action = self.decouple_action(action=action,
                                                    observation=all_observes[0])
            all_observes, reward, done, info_before, info_after = self.test_env.step([decoupled_action])
            reward_sum += reward
            if done:
                self.test_observes_collect.clear()
            if self.test_env.done:
                self.test_env.reset()
            # state = next_state

        return reward_sum


if __name__ == '__main__':
    from env.chooseenv import make
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, help="the dir to save log and model",
                        default='/devdata1/lhdata/kafang/DQN')
    parser.add_argument("--imitate-step", type=float, help="steps for imitation", default=0)
    parser.add_argument("--rl-step", type=float, help="steps for RL", default=1e8)
    parser.add_argument("--critic-mlp-hidden-size", type=int, help="number of hidden units per layer in critic",
                        default=512)
    parser.add_argument("--actor-mlp-hidden-size", type=int, help="number of hidden units per layer in actor",
                        default=512)
    parser.add_argument("--batch-size", type=int, help="number of samples per minibatch", default=256)
    parser.add_argument("--replay-buffer-capacity", type=int, help="replay buffer size", default=1e6)
    parser.add_argument("--critic-lr", type=float, help="learning rate of critic", default=3e-4)
    parser.add_argument("--soft-tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-cache-len", type=int, default=1)
    parser.add_argument("--SRR", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--noise-dumping", type=float, default=0.99)
    parser.add_argument("--init-noise", type=float, default=0.5)
    parser.add_argument("--priorities-coefficient", type=float, default=0.5)
    parser.add_argument("--priorities-bias", type=float, default=0.05)

    args = parser.parse_args()

    # 设置随机数种子
    setup_seed(args.seed)
    
    env_type = "kafang_stock"
    env = make(env_type, seed=args.seed)
    test_env = make(env_type, seed=args.seed)

    cache_single_dim = 5
    basic_state_dim = 3
    if args.SRR:
        cache_single_dim += 5
        basic_state_dim += 1
    state_dim = args.max_cache_len * cache_single_dim + 2

    trainer = DQNTrainer(state_dim=state_dim,
                         critic_mlp_hidden_size=args.critic_mlp_hidden_size,
                         critic_lr=args.critic_lr,
                         gamma=args.gamma,
                         soft_tau=args.soft_tau,
                         env=env,
                         test_env=test_env,
                         replay_buffer_capacity=args.replay_buffer_capacity,
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                         max_cache_len=args.max_cache_len,
                         priorities_coefficient=args.priorities_coefficient,
                         priorities_bias=args.priorities_bias
                         # device=torch.device("cpu")
                         )

    trainer.RL_train(save_dir=args.save_dir,
                     rl_step=args.rl_step,
                     batch_size=args.batch_size,
                     init_noise=args.init_noise,
                     noise_dumping=args.noise_dumping
                     )
