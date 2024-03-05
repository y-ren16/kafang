import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import torch
import numpy as np
from torch.distributions import Categorical


# 状态空间：连续的向量
# 动作空间：二维向量，计划买入价和较高的计划卖出价
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class MultiQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, ensembles=2, init_w=3e-3):
        super(MultiQNetwork, self).__init__()
        self.Qs = nn.ModuleList([SoftQNetwork(state_dim, action_dim, hidden_size, init_w) for _ in range(ensembles)])

    def forward(self, state, action):
        out = [q_net(state, action) for q_net in self.Qs]
        return torch.stack(out, dim=0)

    def qLoss(self, target, state, action, criterion):
        loss = 0
        for q_net in self.Qs:
            loss += criterion(q_net(state, action), target)
        return loss

    # def parameters(self, recurse: bool = True):
    #     p = []
    #     for q_net in self.Qs:
    #         p += q_net.parameters()
    #     return p

class GaussianPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3, log_std_min=-25, log_std_max=10):
        super(GaussianPolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def get_log_prob(self, state, action, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = 0.5 * torch.log((1 + action + epsilon) / (1 - action + epsilon))
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        return log_prob

    def get_action(self, state, dtype='ndarray', random_flag=True):
        # if not isinstance(state, torch.Tensor):
        #     state = torch.FloatTensor(state).to(self.device)
        # if len(state.shape) == 1:
        #     state = state.unsqueeze(0)
        mean, log_std = self.forward(state)
        if random_flag:
            std = log_std.exp()
            normal = Normal(mean, std)

            action = torch.tanh(normal.sample())
        else:
            action = torch.tanh(mean)
        if dtype == 'ndarray':
            action = action.detach().cpu().numpy()
        return action


class DiscreteQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3):
        super(DiscreteQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DiscretePolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, init_w=3e-3):
        super(DiscretePolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, action_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return F.softmax(x, dim=-1)

    def evaluate(self, state, epsilon=1e-6):
        prob = self.forward(state)
        dist = Categorical(prob)
        action = dist.sample().view(-1, 1)
        z = (prob == 0.0).float() * epsilon
        log_prob = torch.log(prob + z)

        return action, prob, log_prob

    def get_action(self, state, dtype='ndarray'):
        # if not isinstance(state, torch.Tensor):
        #     state = torch.FloatTensor(state).to(self.device)
        # if len(state.shape) == 1:
        #     state = state.unsqueeze(0)

        prob = self.forward(state)

        dist = Categorical(prob)
        # action = dist.sample().view(-1, 1)
        action = dist.sample()
        if dtype == 'ndarray':
            action = action.detach().cpu().numpy()
        return action


class MultiDiscreteQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size, ensembles=2, init_w=3e-3):
        super(MultiDiscreteQNetwork, self).__init__()
        self.Qs = nn.ModuleList([DiscreteQNetwork(state_dim, action_dim, hidden_size, init_w) for _ in range(ensembles)])

    def forward(self, state):
        out = [q_net(state) for q_net in self.Qs]
        return torch.stack(out, dim=0)

    def qLoss(self, target, state, action, criterion):
        loss = 0
        for q_net in self.Qs:
            loss += criterion(q_net(state)[action], target)
        return loss


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def clear(self):
        self.buffer = []
        self.position = 0
        return

    def __len__(self):
        return len(self.buffer)