import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], 1)


class conv_net(nn.Module):
    def __init__(self, observation_dim):
        super(conv_net, self).__init__()
        self.observation_dim = observation_dim
        self.net = nn.Sequential(
            nn.Conv2d(self.observation_dim[0], 8, 4, 2),
            CReLU(),
            nn.Conv2d(16, 8, 5, 1),
            CReLU(),
            nn.Conv2d(16, 8, 3, 1),
            CReLU()
        )

    def forward(self, observation):
        observation = observation / 255.
        x = self.net(observation).view(observation.size(0), -1)
        return x


class dueling_ddqn(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(dueling_ddqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.conv_net = conv_net(self.observation_dim)
        self.advantage_net = nn.Sequential(
            nn.Linear(self.conv_dim(), 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.conv_dim(), 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )

    def conv_dim(self):
        return self.conv_net(torch.zeros([1, * self.observation_dim])).view(1, -1).size(-1)

    def forward(self, observation):
        conv_feature = self.conv_net(observation)
        adv = self.advantage_net(conv_feature)
        val = self.value_net(conv_feature)
        return adv + val - adv.mean()

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(observation)
            action = q_value.max(1)[1].data[0].item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action


class policy(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(policy, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.conv_net = conv_net(self.observation_dim)
        self.fc_net = nn.Sequential(
            nn.Linear(self.conv_dim(), 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def conv_dim(self):
        return self.conv_net(torch.zeros([1, * self.observation_dim])).view(1, -1).size(-1)

    def forward(self, observation):
        return self.fc_net(self.conv_net(observation))

    def act(self, observation):
        probs = self.forward(observation)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().detach().item()
        return action