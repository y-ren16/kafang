import torch
import numpy as np
from RL_train.basicSACTrainer import basicSACMarketmakingTrainer
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import os


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


class MarketmakingTrainer(basicSACMarketmakingTrainer):
    def __init__(self, state_dim, critic_mlp_hidden_size, actor_mlp_hidden_size, log_alpha, critic_lr, actor_lr,
                 alpha_lr, target_entropy, gamma, soft_tau, env, test_env, replay_buffer_capacity, device,
                 max_cache_len, state_keys=('signal0', 'signal1', 'signal2', 'ap0', 'bp0', 'ap1', 'bp1', 'ap2', 'bp2', 'ap3', 'bp3', 'ap4', 'bp4')):
        action_dim = 1  # 动作空间为股票估值
        self.observes_collect = ObservesCollect(maxlen=max_cache_len, keys=state_keys)
        self.test_observes_collect = ObservesCollect(maxlen=max_cache_len, keys=state_keys)

        super().__init__(state_dim, action_dim, critic_mlp_hidden_size, actor_mlp_hidden_size, log_alpha, critic_lr, actor_lr,
                 alpha_lr, target_entropy, gamma, soft_tau, env, test_env, replay_buffer_capacity, device)

    def extract_state(self, all_observes) -> np.ndarray:
        """
        将原始的观测值转换为向量化的状态
        :param all_observes: 原始观测值
        :return: 向量化的状态
        """
        return self.observes_collect.extract_state(all_observes)
        # if isinstance(all_observes, list):
        #     state = all_observes[0]['observation']
        # else:
        #     state = all_observes['observation']
        # ap0_t0 = state['ap0_t0']
        # # state包含给定因子、买单价格、买单手数、卖单价格、卖单手数
        # state = np.array(
        #     [state['signal0'], state['signal1'], state['signal2'], state['bp0'], state['bp1'], state['bp2'],
        #      state['bp3'], state['bp4'],
        #      state['bv0'], state['bv1'], state['bv2'], state['bv3'], state['bv4'], state['ap0'], state['ap1'],
        #      state['ap2'], state['ap3'], state['ap4'],
        #      state['av0'], state['av1'], state['av2'], state['av3'], state['av4']])
        # state[3:8] = state[3:8] / ap0_t0
        # state[13:18] = state[13:18] / ap0_t0
        # return state

    def decouple_action(self, action: np.ndarray, observation: dict) -> list:
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
        bid_price = (observation['bp0'] + observation['ap0']) / 2 * (action[0] * 0.005 + 1) / (1 + 0.00007 + 0.00001)  # 计划买入价
        ask_price = (observation['bp0'] + observation['ap0']) / 2 * (action[0] * 0.005 + 1) * (1 + 0.00007 + 0.00001)  # 计划卖出价
        # 仅考虑以bp0价位卖出，不考虑bp1、bp2、bp3、bp4
        if ask_price <= observation[f'bp0']:
            ask_volume = observation[f'bv0']
            ask_price = observation[f'bp0']
        else:
            ask_volume = 0
        ask_volume = min(ask_volume, 300 + observation['code_net_position'], 20)
        if ask_volume > 0:
            return [[0, 0, 1], float(ask_volume), ask_price]  # 以ask_price价格卖出ask_volume手


        if bid_price >= observation[f'ap0']:
            bid_volume = observation[f'av0']
            bid_price = observation[f'ap0']
        else:
            bid_volume = 0
        bid_volume = min(bid_volume, 300 - observation['code_net_position'], 20)
        if bid_volume > 0:
            return [[1, 0, 0], float(bid_volume), bid_price]  # 以bid_price价格买入bid_volume手

        return [[0, 1, 0], 0., 0.]  # 什么都不做

    def RL_train(self, save_dir, train_step, batch_size):
        if not os.path.exists(os.path.join(save_dir, 'log')):
            os.makedirs(os.path.join(save_dir, 'log'))
        if not os.path.exists(os.path.join(save_dir, 'models')):
            os.makedirs(os.path.join(save_dir, 'models'))

        logger_writer = SummaryWriter(os.path.join(save_dir, 'log'))

        all_observes = self.env.reset()
        self.test_env.reset()
        state = self.observes_collect.extract_state(all_observes)

        for i in range(int(train_step) + 1):
            action = self.actor.get_action(state=torch.FloatTensor(state).to(self.device))
            decoupled_action = self.decouple_action(action=action, observation=all_observes[0])
            all_observes, reward, done, info_before, info_after = self.env.step([decoupled_action])
            next_state = self.observes_collect.extract_state(all_observes)
            self.replay_buffer.push(state, action, reward, next_state, done)
            if done:  # 如果单只股票/单日/全部数据结束，则重置历史观测数据
                self.observes_collect.clear()
            if self.env.done:
                all_observes = self.env.reset()
                state = self.observes_collect.extract_state(all_observes)
            else:
                state = next_state

            logger = {}
            logger.update(self.critic_train_step(batch_size))
            self.soft_update_target_critic()
            logger.update(self.actor_train_step(batch_size))
            if i % 10000 == 0:
                logger['test_reward_mean'] = self.RL_test(test_length=10000)

                for key in logger.keys():
                    logger_writer.add_scalar(key, logger[key], i)
                self.save_RL_part(os.path.join(save_dir, 'models', 'RL_part_%dk.pt' % (i / 1000)))
                info = 'step: %dk' % (i / 1000)
                # info = 'step: %d' % (i)
                for key in logger.keys():
                    info += ' | %s: %.3f' % (key, logger[key])
                print(info)

    def RL_test(self, test_length=1000):
        reward_sum = 0
        for _ in range(test_length):
            all_observes = self.test_env.all_observes
            state = self.test_observes_collect.extract_state(all_observes)
            # while not self.test_env.done:
            state = torch.FloatTensor(state).to(self.device)
            action = self.actor.get_action(state)
            decoupled_action = self.decouple_action(action=action,
                                                    observation=all_observes[0])
            all_observes, reward, done, info_before, info_after = self.test_env.step([decoupled_action])
            reward_sum += reward
            if done:  # 如果单只股票/单日/全部数据结束，则重置历史观测数据
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
                        default='/devdata1/lhdata/kafang/test')
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
    parser.add_argument("--max-cache-len", type=int, default=5)
    parser.add_argument("--basic-state-dim", type=int, default=3)
    parser.add_argument("--cache-single-dim", type=int, default=3)
    parser.add_argument("--state-keys", type=list, default=('signal0', 'signal1', 'signal2', 'ap0', 'bp0', 'ap1', 'bp1', 'ap2', 'bp2', 'ap3', 'bp3', 'ap4', 'bp4'))

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

    trainer = MarketmakingTrainer(state_dim=(args.max_cache_len - 1) * args.cache_single_dim + args.basic_state_dim + 1,
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

    trainer.RL_train(save_dir=args.save_dir,
                     train_step=args.rl_step,
                     batch_size=args.batch_size
                     )