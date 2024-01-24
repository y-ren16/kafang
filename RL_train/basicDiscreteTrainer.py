import torch
from torch import optim
from RL_train.network import MultiDiscreteQNetwork, ReplayBuffer
import copy
from env.stock_raw.utils import Order
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import math
import random


# SunTree和PrioritizedExperienceReplay参考https://github.com/takoika/PrioritizedExperienceReplay
class SumTree(object):
    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1  # 树的层数，树的叶子节点个数不小于max_size
        self.tree_size = 2 ** self.tree_level - 1  # 树的总节点数
        self.tree = [0 for i in range(self.tree_size)]  # 记录每个节点的权重，未填充的叶子节点初始权重为0（即所有节点权重均为0）
        self.data = [None for i in range(self.max_size)]  # 记录每个叶子节点对应的数据。只有叶子节点对应于buffer中的数据
        self.size = 0  # 已填充的数据大小
        self.cursor = 0  # 下一个填充数据的位置

    def add(self, contents, value):
        index = self.cursor
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = 2 ** (self.tree_level - 1) - 1 + index  # 第index个叶子节点对应的节点编号
        diff = value - self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex - 1) / 2)  # 更新父节点，使父节点的权重依然为当前节点与兄弟节点权重之和
            self.reconstruct(tindex, diff)

    def find(self, value, norm=True):
        """
        给定一个随机数，返回对应的数据、权重和位置
        :param value: 随机数
        :param norm: 如果为True，则表示输入的随机数value是正则化的，即（0,1）之间，需要将value乘以根节点的权重
        :return:数据，权重，数据位置
        """
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        if 2 ** (self.tree_level - 1) - 1 <= index:  # index对应的节点已经是叶子节点，则直接返回数据，权重，数据位置
            return self.data[index - (2 ** (self.tree_level - 1) - 1)], self.tree[index], index - (
                    2 ** (self.tree_level - 1) - 1)

        left = self.tree[2 * index + 1]

        if value <= left:  # 如果value小于左孩子的值
            return self._find(value, 2 * index + 1)  # 则在左分支继续寻找
        else:  # 如果value大于左孩子的值
            return self._find(value - left, 2 * (index + 1))  # 则在右分支继续寻找（value要减去左孩子的值）

    def print_tree(self):
        """
        打印每个节点的权重
        :return:
        """
        for k in range(1, self.tree_level + 1):
            for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
                print(self.tree[j], end=' ')
            print()

    def filled_size(self):
        return self.size


class PrioritizedExperienceReplay(object):
    def __init__(self, memory_size=1000000, load_path=None, buffer=None, alpha=1, init_priority=20):
        self.tree = SumTree(memory_size)
        self.memory_size = memory_size
        self.alpha = alpha
        self.init_priority = init_priority
        if load_path != None:
            buffer = torch.load(load_path)
        if buffer != None:
            for data in buffer:
                self.tree.add(data, init_priority ** alpha)

    def push(self, *data):
        self.tree.add(data, self.init_priority ** self.alpha)

    def sample(self, batch_size, beta=0):
        assert self.tree.filled_size() >= batch_size, \
            f'The batch size {batch_size} is too large. The buffer has {self.tree.filled_size()} data.'

        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append(
                (1. / self.memory_size / priority) ** beta if priority > 1e-16 else 0)  # 用于importance sampling
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0])  # To avoid duplicating 避免重复采样！！！

        self.priority_update(indices, priorities)  # Revert priorities

        indices = np.array(indices)
        weights = np.array(weights)
        priorities = np.array(priorities)
        weights /= max(weights)  # 只需要数据而不用重要性采样

        return out, indices, weights, priorities

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        if isinstance(indices, list):
            for i, p in zip(indices, priorities):
                self.tree.val_update(i, p ** self.alpha)
        else:
            assert isinstance(indices, np.ndarray)
            for i in range(len(indices)):
                self.tree.val_update(indices[i], priorities[i] ** self.alpha)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i) ** -old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)

    def __len__(self):
        return self.tree.filled_size()


class basicDQNTrainer:
    def __init__(self, state_dim, action_dim, critic_mlp_hidden_size, critic_lr, gamma, soft_tau, env, test_env,
                 replay_buffer_capacity, device, priorities_coefficient=0.9, priorities_bias=1):
        self.critic = MultiDiscreteQNetwork(state_dim=state_dim,
                                            action_dim=action_dim,
                                            hidden_size=critic_mlp_hidden_size
                                            ).to(device)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.soft_tau = soft_tau
        self.env = env
        self.test_env = test_env
        self.replay_buffer = PrioritizedExperienceReplay(memory_size=replay_buffer_capacity)
        self.priorities_coefficient = priorities_coefficient
        self.priorities_bias = priorities_bias
        self.device = device

    def extract_state(self, all_observes) -> np.ndarray:
        """
        将原始的观测值转换为向量化的状态
        :param all_observes: 原始观测值
        :return: 向量化的状态
        """
        pass

    def decouple_action(self, action: int, observation: dict) -> list:
        """
        根据action和当前的订单簿生成具体的订单
        :param action: actor输出的动作
        :param observation: 观测值，其中'bp0'到'bp4'表示从高到低的买家报价，'bv0'到'bv4'表示对应的买家报价手数，
                            'ap0'到'ap4'表示从低到高的卖家报价，'av0'到'av4'表示对应的卖家报价手数，
        :return: 具体的订单
        """
        pass

    def get_action(self, state):
        q = torch.min(self.critic(state), dim=0)[0]
        return q.argmax(dim=-1)

    def critic_train_step(self, batch_size):
        batch_size = min(batch_size, len(self.replay_buffer))
        out, indices, weights, priorities = self.replay_buffer.sample(batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*out))
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.tensor(action, dtype=torch.int64).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        if len(reward.shape) == 2:
            reward = reward.squeeze(-1)  # shape=(batch_size,)
        done = torch.FloatTensor(done).to(self.device)
        if len(done.shape) == 2:
            done = done.squeeze(-1)  # shape=(batch_size,)
        with torch.no_grad():
            next_action = self.get_action(next_state)
            next_q_value = torch.min(self.target_critic(next_state), dim=0)[0][torch.arange(batch_size), next_action]
            target_q_value = reward + (1 - done) * self.gamma * next_q_value
        assert target_q_value.shape == torch.Size([batch_size])
        critic_loss = torch.zeros_like(reward)
        for q_net in self.critic.Qs:
            critic_loss += (q_net(state)[torch.arange(batch_size), action] - target_q_value) ** 2

        priorities = self.priorities_coefficient * priorities + (1 - self.priorities_coefficient) * (
                np.clip(critic_loss.detach().cpu().numpy() ** 0.5, 0, 100) + self.priorities_bias)
        self.replay_buffer.priority_update(indices, priorities)

        critic_loss = critic_loss.mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        logger = {
            'critic_loss': critic_loss.item()
        }
        return logger

    def soft_update_target_critic(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

    def RL_train(self, save_dir, train_step, batch_size):
        if not os.path.exists(os.path.join(save_dir, 'log')):
            os.makedirs(os.path.join(save_dir, 'log'))
        if not os.path.exists(os.path.join(save_dir, 'models')):
            os.makedirs(os.path.join(save_dir, 'models'))

        logger_writer = SummaryWriter(os.path.join(save_dir, 'log'))

        all_observes = self.env.reset()
        self.test_env.reset()
        state = self.extract_state(all_observes)

        for i in range(int(train_step) + 1):
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
            state = self.extract_state(all_observes)
            # while not self.test_env.done:
            state = torch.FloatTensor(state).to(self.device)
            action = self.get_action(state)
            decoupled_action = self.decouple_action(action=action,
                                                    observation=all_observes[0])
            all_observes, reward, done, info_before, info_after = self.test_env.step([decoupled_action])
            reward_sum += reward
            if self.test_env.done:
                self.test_env.reset()
            # state = next_state

        return reward_sum

    def save_RL_part(self, save_path):
        torch.save({'critic': self.critic.state_dict(),
                    'target_critic': self.target_critic.state_dict(),
                    },
                   save_path)
        return

    def load_RL_part(self, save_path):
        payload = torch.load(save_path)
        self.critic.load_state_dict(payload['critic'])
        self.target_critic.load_state_dict(payload['target_critic'])
        return
