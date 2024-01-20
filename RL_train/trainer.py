import torch
import numpy as np
from RL_train.basicSACTrainer import basicSACMarketmakingTrainer


class MarketmakingTrainer(basicSACMarketmakingTrainer):
    def __init__(self, state_dim, critic_mlp_hidden_size, actor_mlp_hidden_size, log_alpha, critic_lr, actor_lr,
                 alpha_lr, target_entropy, gamma, soft_tau, env, test_env, replay_buffer_capacity, device):
        action_dim = 2  # 动作空间为较低的计划买入价和较高的计划卖出价

        super().__init__(state_dim, action_dim, critic_mlp_hidden_size, actor_mlp_hidden_size, log_alpha, critic_lr, actor_lr,
                 alpha_lr, target_entropy, gamma, soft_tau, env, test_env, replay_buffer_capacity, device)

    def extract_state(self, all_observes) -> np.ndarray:
        """
        将原始的观测值转换为向量化的状态
        :param all_observes: 原始观测值
        :return: 向量化的状态
        """
        if isinstance(all_observes, list):
            state = all_observes[0]['observation']
        else:
            state = all_observes['observation']
        ap0_t0 = state['ap0_t0']
        # state包含给定因子、买单价格、买单手数、卖单价格、卖单手数
        state = np.array(
            [state['signal0'], state['signal1'], state['signal2'], state['bp0'], state['bp1'], state['bp2'],
             state['bp3'], state['bp4'],
             state['bv0'], state['bv1'], state['bv2'], state['bv3'], state['bv4'], state['ap0'], state['ap1'],
             state['ap2'], state['ap3'], state['ap4'],
             state['av0'], state['av1'], state['av2'], state['av3'], state['av4']])
        state[3:8] = state[3:8] / ap0_t0
        state[13:18] = state[13:18] / ap0_t0
        return state

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
        bid_price = (action[0] * 0.01 + 1) * (observation['bp0'] + observation['ap0']) / 2  # 计划买入价
        ask_price = (action[0] * 0.01 + (action[1] + 1) / 2 * 0.01 + 1) * (observation['bp0'] + observation['ap0']) / 2
        ask_volume = 0
        for i in range(5):
            if ask_price <= observation[f'bp{i}']:
                ask_volume += observation[f'bv{i}']
            else:
                break
        ask_volume = min(ask_volume, 300 + observation['code_net_position'])
        if ask_volume > 0:
            return [[0, 0, 1], float(ask_volume), ask_price]  # 以ask_price价格卖出ask_volume手

        bid_volume = 0
        for i in range(5):
            if bid_price >= observation[f'ap{i}']:
                bid_volume += observation[f'av{i}']
            else:
                break
        bid_volume = min(bid_volume, 300 - observation['code_net_position'])
        if bid_volume > 0:
            return [[1, 0, 0], float(bid_volume), bid_price]  # 以bid_price价格买入bid_volume手

        return [[0, 1, 0], 0., 0.]  # 什么都不做


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

    args = parser.parse_args()

    env_type = "kafang_stock"
    env = make(env_type, seed=None)
    test_env = make(env_type, seed=None)

    trainer = MarketmakingTrainer(state_dim=23,
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
                                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                  # device=torch.device("cpu")
                                  )

    trainer.RL_train(save_dir=args.save_dir,
                     train_step=args.rl_step,
                     batch_size=args.batch_size
                     )
