import gym
import lasertag
import numpy as np
from model import dueling_ddqn, policy
from buffer import reservoir_buffer, n_step_replay_buffer, replay_buffer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random


class nfsp(object):
    def __init__(self, env, epsilon_init, decay, epsilon_min, update_freq, sl_lr, rl_lr, sl_capa, rl_capa, n_step, gamma, eta, max_episode, negative, rl_start, sl_start, train_freq, rl_batch_size, sl_batch_size, render, device):
        self.env = env
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.update_freq = update_freq
        self.sl_lr = sl_lr
        self.rl_lr = rl_lr
        self.sl_capa = sl_capa
        self.rl_capa = rl_capa
        self.n_step = n_step
        self.gamma = gamma
        self.eta = eta
        self.max_episode = max_episode
        self.negative = negative
        self.sl_start = sl_start
        self.rl_start = rl_start
        self.train_freq = train_freq
        self.rl_batch_size = rl_batch_size
        self.sl_batch_size = sl_batch_size
        self.render = render
        self.device = device

        self.observation_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        self.sl_p1_buffer = reservoir_buffer(self.sl_capa)
        self.sl_p2_buffer = reservoir_buffer(self.sl_capa)
        if self.n_step > 1:
            self.rl_p1_buffer = n_step_replay_buffer(self.rl_capa, self.n_step, self.gamma)
            self.rl_p2_buffer = n_step_replay_buffer(self.rl_capa, self.n_step, self.gamma)
        else:
            self.rl_p1_buffer = replay_buffer(self.rl_capa)
            self.rl_p2_buffer = replay_buffer(self.rl_capa)
        self.p1_dqn_eval = dueling_ddqn(self.observation_dim, self.action_dim).to(self.device)
        self.p1_dqn_target = dueling_ddqn(self.observation_dim, self.action_dim).to(self.device)
        self.p2_dqn_eval = dueling_ddqn(self.observation_dim, self.action_dim).to(self.device)
        self.p2_dqn_target = dueling_ddqn(self.observation_dim, self.action_dim).to(self.device)
        self.p1_dqn_target.load_state_dict(self.p1_dqn_eval.state_dict())
        self.p2_dqn_target.load_state_dict(self.p2_dqn_eval.state_dict())

        self.p1_policy = policy(self.observation_dim, self.action_dim).to(self.device)
        self.p2_policy = policy(self.observation_dim, self.action_dim).to(self.device)

        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(-1. * x / self.decay)

        self.p1_dqn_optimizer = torch.optim.Adam(self.p1_dqn_eval.parameters(), lr=self.rl_lr)
        self.p2_dqn_optimizer = torch.optim.Adam(self.p2_dqn_eval.parameters(), lr=self.rl_lr)
        self.p1_policy_optimizer = torch.optim.Adam(self.p1_policy.parameters(), lr=self.sl_lr)
        self.p2_policy_optimizer = torch.optim.Adam(self.p2_policy.parameters(), lr=self.sl_lr)

    def rl_train(self, buffer, target_model, eval_model, optimizer, count):
        observation, action, reward, next_observation, done = buffer.sample(self.rl_batch_size)

        observation = torch.FloatTensor(observation).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_observation = torch.FloatTensor(next_observation).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = eval_model.forward(observation)
        next_q_values = target_model.forward(next_observation)
        argmax_actions = eval_model.forward(next_observation).max(1)[1].detach()
        next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * (1 - done) * next_q_value

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if count % self.update_freq == 0:
            target_model.load_state_dict(eval_model.state_dict())

    def sl_train(self, buffer, model, optimizer):
        observation, action = buffer.sample(self.sl_batch_size)

        observation = torch.FloatTensor(observation).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        probs = model.forward(observation)
        prob = probs.gather(1, action.unsqueeze(1)).squeeze(1)
        log_prob = prob.log()
        loss = -log_prob.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def run(self):
        count = 0
        for episode in range(self.max_episode):
            p1_total_reward = 0
            p2_total_reward = 0
            p1_obs, p2_obs = self.env.reset()
            if self.render:
                self.env.render()
            while True:
                is_best_response = False if random.random() > self.eta else True
                count += 1
                if is_best_response:
                    p1_action = self.p1_dqn_eval.act(torch.FloatTensor(np.expand_dims(p1_obs, 0)).to(self.device), self.epsilon(count))
                    p2_action = self.p2_dqn_eval.act(torch.FloatTensor(np.expand_dims(p2_obs, 0)).to(self.device), self.epsilon(count))
                else:
                    p1_action = self.p1_policy.act(torch.FloatTensor(np.expand_dims(p1_obs, 0)).to(self.device))
                    p2_action = self.p2_policy.act(torch.FloatTensor(np.expand_dims(p1_obs, 0)).to(self.device))
                actions = {"1": p1_action, "2": p2_action}
                (p1_next_obs, p2_next_obs), reward, done, info = self.env.step(actions)

                if self.render:
                    self.env.render()

                p1_reward = reward[0] - 1 if self.negative else reward[0]
                p2_reward = reward[1] - 1 if self.negative else reward[1]
                p1_total_reward += p1_reward
                p2_total_reward += p2_reward

                self.rl_p1_buffer.store(p1_obs, p1_action, p1_reward, p1_next_obs, done)
                self.rl_p2_buffer.store(p2_obs, p2_action, p2_reward, p2_next_obs, done)
                if is_best_response:
                    self.sl_p1_buffer.store(p1_obs, p1_action)
                    self.sl_p2_buffer.store(p2_obs, p2_action)

                if len(self.rl_p1_buffer) > self.rl_start and len(self.sl_p1_buffer) > self.sl_start and count % self.train_freq == 0:
                    self.rl_train(self.rl_p1_buffer, self.p1_dqn_target, self.p1_dqn_eval, self.p1_dqn_optimizer, count)
                    self.rl_train(self.rl_p2_buffer, self.p2_dqn_target, self.p2_dqn_eval, self.p2_dqn_optimizer, count)

                    self.sl_train(self.sl_p1_buffer, self.p1_policy, self.p1_policy_optimizer)
                    self.sl_train(self.sl_p2_buffer, self.p2_policy, self.p2_policy_optimizer)

                p1_obs = p1_next_obs
                p2_obs = p2_next_obs

                if done:
                    print('episode: {}  p1_total_reward: {:.1f}  p2_total_reward: {:.1f}  epsilon: {:.3f}'.format(episode + 1, p1_total_reward, p2_total_reward, self.epsilon(count)))
                    break
