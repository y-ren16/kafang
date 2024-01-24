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


class DiscreteTrainer(basicDiscreteTrainer):
    def __init__(self, state_dim, critic_mlp_hidden_size, actor_mlp_hidden_size, critic_lr, actor_lr, gamma, soft_tau, env, test_env,
                 replay_buffer_capacity, device, priorities_coefficient=0.9, priorities_bias=1, log_alpha=1.0):
        action_dim = 11  # 0-4：买入，5：不做：6-10：卖出
        super().__init__(state_dim, action_dim, critic_mlp_hidden_size, critic_lr, gamma, soft_tau, env, test_env,
                         replay_buffer_capacity, device, priorities_coefficient, priorities_bias)
        self.actor = DiscretePolicyNetwork(state_dim, action_dim, actor_mlp_hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.log_alpha = torch.tensor(log_alpha).to(device)
        # self.log_alpha.requires_grad = True

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, state):
        return self.actor.get_action(state)

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
        state = np.array([state['signal0'], state['signal1'], state['signal2']])
        # state[3:8] = state[3:8] / ap0_t0
        # state[13:18] = state[13:18] / ap0_t0
        return state

    def decouple_action(self, action: int, observation: dict) -> list:
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

        if action < 5:
            bid_price = observation[f'ap{action}']
            bid_volume = 0
            for i in range(action + 1):
                bid_volume += observation[f'av{i}']
            bid_volume = min(bid_volume, 300 - observation['code_net_position'])
            if bid_volume > 0:
                return [[1, 0, 0], float(bid_volume), bid_price]  # 以bid_price价格买入bid_volume手
        elif action > 5:
            ask_price = observation[f'bp{action - 6}']
            ask_volume = 0
            for i in range(action - 5):
                ask_volume += observation[f'bv{i}']
            ask_volume = min(ask_volume, 300 + observation['code_net_position'])
            if ask_volume > 0:
                return [[0, 0, 1], float(ask_volume), ask_price]  # 以ask_price价格卖出ask_volume手

        return [[0, 1, 0], 0., 0.]  # 什么都不做

    def get_demonstration(self, state):
        pass

    def imitate_step(self, batch_size):
        batch_size = min(batch_size, len(self.replay_buffer))
        out, indices, weights, priorities = self.replay_buffer.sample(batch_size)
        state, _, _, _, _ = map(np.stack, zip(*out))
        state = torch.FloatTensor(state).to(self.device)
        action = self.get_demonstration(state)
        _, prob, log_prob = self.actor.evaluate(state)
        actor_loss = -torch.log(prob[torch.tensor(range(action.shape[0])), action.view(-1).long()]).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        logger = {
            'actor_loss': actor_loss.item()
        }
        return logger

    def actor_train_step(self, batch_size):
        batch_size = min(batch_size, len(self.replay_buffer))
        out, indices, weights, priorities = self.replay_buffer.sample(batch_size)
        state, _, _, _, _ = map(np.stack, zip(*out))
        state = torch.FloatTensor(state).to(self.device)

        _, prob, log_prob = self.actor.evaluate(state)
        with torch.no_grad():
            expected_q_value = torch.min(self.critic(state), dim=0)[0]
            expected_q_value = expected_q_value - torch.mean(expected_q_value, dim=-1, keepdim=True) \
                .repeat(1, expected_q_value.shape[1])  # using advantage function for normalization
        expected_q_value = self.alpha.detach() * log_prob - expected_q_value
        actor_loss = torch.einsum('ij,ij->i', expected_q_value, prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        logger = {
            'actor_loss': actor_loss.item(),
            'entropy': torch.einsum('ij,ij->i', prob.log(), prob).mean().item()
        }
        return logger

    def RL_train(self, save_dir, imitate_step, rl_step, batch_size, init_noise=0.5, noise_dumping=0.95):
        if not os.path.exists(os.path.join(save_dir, 'log')):
            os.makedirs(os.path.join(save_dir, 'log'))
        if not os.path.exists(os.path.join(save_dir, 'models')):
            os.makedirs(os.path.join(save_dir, 'models'))

        logger_writer = SummaryWriter(os.path.join(save_dir, 'log'))

        all_observes = self.env.reset()
        self.test_env.reset()
        state = self.extract_state(all_observes)
        noise = init_noise
        for i in range(-int(imitate_step), int(rl_step) + 1):
            if random.random() < noise:
                action = random.randint(0, self.action_dim-1)
            else:
                action = self.get_action(state=torch.FloatTensor(state).to(self.device))
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
            if i < 0:
                logger.update(self.imitate_step(batch_size))
            else:
                logger.update(self.actor_train_step(batch_size))
            if i % 10000 == 0:
                logger['test_reward_mean'] = self.RL_test(test_length=10000)
                if i > 0:
                    logger['noise'] = noise
                    noise *= noise_dumping
                for key in logger.keys():
                    logger_writer.add_scalar(key, logger[key], i)
                self.save_RL_part(os.path.join(save_dir, 'models', 'RL_part_%dk.pt' % (i / 1000)))
                info = 'step: %dk' % (i / 1000)
                # info = 'step: %d' % (i)
                for key in logger.keys():
                    info += ' | %s: %.3f' % (key, logger[key])
                print(info)


if __name__ == '__main__':
    from env.chooseenv import make
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, help="the dir to save log and model",
                        default='/devdata1/lhdata/kafang/test')
    parser.add_argument("--imitate-step", type=float, help="steps for imitation", default=1e6)
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

    trainer = DiscreteTrainer(state_dim=3,
                              critic_mlp_hidden_size=args.critic_mlp_hidden_size,
                              actor_mlp_hidden_size=args.actor_mlp_hidden_size,
                              critic_lr=args.critic_lr,
                              actor_lr=args.actor_lr,
                              gamma=args.gamma,
                              soft_tau=args.soft_tau,
                              env=env,
                              test_env=test_env,
                              replay_buffer_capacity=args.replay_buffer_capacity,
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                              # device=torch.device("cpu")
                              )

    trainer.RL_train(save_dir=args.save_dir,
                     imitate_step=args.imitate_step,
                     rl_step=args.rl_step,
                     batch_size=args.batch_size
                     )
