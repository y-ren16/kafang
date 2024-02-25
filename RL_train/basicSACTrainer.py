import torch
from torch import optim
from RL_train.network import MultiQNetwork, GaussianPolicyNetwork, ReplayBuffer
import copy
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os


class basicSACMarketmakingTrainer:
    def __init__(self, state_dim, action_dim, critic_mlp_hidden_size, actor_mlp_hidden_size, log_alpha, critic_lr, actor_lr,
                 alpha_lr, target_entropy, gamma, soft_tau, env, test_env, replay_buffer_capacity, device):
        self.critic = MultiQNetwork(state_dim=state_dim,
                               action_dim=action_dim,
                               hidden_size=critic_mlp_hidden_size).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        # for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
        #     target_param.data.copy_(param.data)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.actor = GaussianPolicyNetwork(state_dim=state_dim,
                                      action_dim=action_dim,
                                      hidden_dim=actor_mlp_hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.log_alpha = torch.FloatTensor([log_alpha]).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.env = env
        self.test_env = test_env
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

        self.device = device

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def extract_state(self, all_observes) -> np.ndarray:
        """
        将原始的观测值转换为向量化的状态
        :param all_observes: 原始观测值
        :return: 向量化的状态
        """
        pass

    def decouple_action(self, action: np.ndarray, observation: dict) -> list:
        """
        根据action和当前的订单簿生成具体的订单
        :param action: actor输出的动作
        :param observation: 观测值，其中'bp0'到'bp4'表示从高到低的买家报价，'bv0'到'bv4'表示对应的买家报价手数，
                            'ap0'到'ap4'表示从低到高的卖家报价，'av0'到'av4'表示对应的卖家报价手数，
        :return: 具体的订单
        """
        pass

    def critic_train_step(self, batch_size):
        state, action, reward, next_state, done = self.replay_buffer.sample(min(batch_size, len(self.replay_buffer)))
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        if len(reward.shape) == 1:
            reward = reward.unsqueeze(-1)  # shape=(batch_size, 1)
        done = torch.FloatTensor(done).to(self.device)
        if len(done.shape) == 1:
            done = done.unsqueeze(-1)  # shape=(batch_size, 1)
        with torch.no_grad():
            next_action, next_log_prob, _, _, _ = self.actor.evaluate(next_state)
            next_q_value = torch.min(self.target_critic(next_state, next_action), dim=0)[
                               0] - self.alpha * next_log_prob
            target_q_value = reward + (1 - done) * self.gamma * next_q_value

        critic_loss = self.critic.qLoss(target_q_value, state, action, F.mse_loss)
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

    def actor_train_step(self, batch_size, mean_lambda=0, std_lambda=1e-3, z_lambda=0.0):
        state, _, _, _, _ = self.replay_buffer.sample(min(batch_size, len(self.replay_buffer)))
        state = torch.FloatTensor(state).to(self.device)

        new_action, log_prob, z, mean, log_std = self.actor.evaluate(state)
        expected_new_q_value = torch.min(self.critic(state, new_action), dim=0)[0]

        # log_prob_target = expected_new_q_value - expected_value
        # actor_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        actor_loss = (self.alpha * log_prob - expected_new_q_value).mean()

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        actor_loss += mean_loss + std_loss + z_loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            new_action, log_prob, _, _, _ = self.actor.evaluate(state)
            alpha_loss = log_prob + self.target_entropy
        entropy = -alpha_loss.mean() + self.target_entropy
        alpha_loss = -self.alpha * alpha_loss.mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        logger = {
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'entropy': entropy.item(),
            'alpha': self.alpha.item(),
            'log_prob': log_prob.mean().item(),
            'expected_new_q_value': expected_new_q_value.mean().item()
        }
        return logger

    def RL_train(self, save_dir, train_step, imitate_step, batch_size):
        if not os.path.exists(os.path.join(save_dir, 'log')):
            os.makedirs(os.path.join(save_dir, 'log'))
        if not os.path.exists(os.path.join(save_dir, 'models')):
            os.makedirs(os.path.join(save_dir, 'models'))

        logger_writer = SummaryWriter(os.path.join(save_dir, 'log'))

        all_observes = self.env.reset()
        self.test_env.reset()
        state = self.extract_state(all_observes)

        for i in range(int(train_step) + 1):
            action = self.actor.get_action(state=torch.FloatTensor(state).to(self.device))
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
            logger.update(self.actor_train_step(batch_size))
            if i % 10000 == 0:
                logger['test_reward_mean'] = self.RL_test(test_length=10000)

                for key in logger.keys():
                    logger_writer.add_scalar(key, logger[key], i)
                self.save_RL_part(os.path.join(save_dir, 'models', 'RL_part_%dk.pt' % (i / 1000)))
                torch.save(self.replay_buffer, os.path.join(save_dir, 'models', 'replay_buffer_%dk.pt' % (i / 1000)))
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
            action = self.actor.get_action(state)
            decoupled_action = self.decouple_action(action=action,
                                                    observation=all_observes[0])
            all_observes, reward, done, info_before, info_after = self.test_env.step([decoupled_action])
            reward_sum += reward
            if self.test_env.done:
                self.test_env.reset()
            # state = next_state

        return reward_sum

    def save_RL_part(self, save_path):
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    'target_critic': self.target_critic.state_dict(),
                    'log_alpha': self.log_alpha,
                    },
                   save_path)
        return

    def load_RL_part(self, save_path):
        payload = torch.load(save_path)
        self.actor.load_state_dict(payload['actor'])
        self.critic.load_state_dict(payload['critic'])
        self.target_critic.load_state_dict(payload['target_critic'])
        self.log_alpha.data.copy_(payload['log_alpha'].data)
        return
