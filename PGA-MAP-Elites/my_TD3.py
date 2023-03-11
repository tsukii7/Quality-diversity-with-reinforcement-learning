'''
Copyright (c) 2020 Scott Fujimoto
Based on Twin Delayed Deep Deterministic Policy Gradients (TD3)
Implementation by Scott Fujimoto https://github.com/sfujim/TD3 Paper: https://arxiv.org/abs/1802.09477,
                    Olle Nilsson: olle.nilsson19@imperial.ac.uk
'''

import copy

import gym
import collections
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import torch.optim as optim
# import matplotlib.pyplot as plt

LEARNING_RATE = 3e-4
# GAMMA = 0.99
MEMORY_CAPACITY = int(1e6)
BATCH_SIZE = 256
TARGET_REPLACE_CNT = 2
# TRAIN_STEP_CNT = 500
# STATES_DIM = 4
# ACTIONS_DIM = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # if m.bias is not None:
        #     nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.LayerNorm):
        pass
    # if isinstance(m, nn.BatchNorm1d):
    #     nn.init.constant_(m.weight, 1.0)
    #     nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, neurons_list=None):
        super(ActorNetwork, self).__init__()
        if neurons_list is None:
            neurons_list = [128, 128]
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, neurons_list[0], bias=True)
        # self.ln1 = nn.LayerNorm(neurons_list[0])
        self.fc2 = nn.Linear(neurons_list[0], neurons_list[1], bias=True)
        # self.ln2 = nn.LayerNorm(neurons_list[1])
        self.fc3 = nn.Linear(neurons_list[1], action_dim, bias=True)
        # self.ln3 = nn.LayerNorm(action_dim)
        # self.out = nn.Linear(256, action_dim)
        # self.out.weight.data.normal_(0, 0.1)

        # self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state):
        # action_value = F.relu(self.ln1(self.fc1(state)))
        # action_value = F.relu(self.ln2(self.fc2(action_value)))
        # action_value = torch.tanh(self.ln3(self.fc3(action_value)))
        action_value = F.relu(self.fc1(state))
        action_value = F.relu(self.fc2(action_value))
        action_value = torch.tanh(self.fc3(action_value))
        action_value = self.max_action * action_value
        return action_value

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self(state).detach().cpu().data.numpy().flatten()
        return action

    def save(self, filename):
        torch.save(self.state_dict(), filename + "_actor")

    def load(self, filename):
        self.load_state_dict(torch.load(filename))


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, neurons_list=None):
        super(CriticNetwork, self).__init__()

        if neurons_list is None:
            neurons_list = [256, 256]

        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, neurons_list[0])
        self.fc2 = nn.Linear(neurons_list[0], neurons_list[1])
        self.out1 = nn.Linear(neurons_list[1], 1)

        # Q2
        self.fc3 = nn.Linear(state_dim + action_dim, neurons_list[0])
        self.fc4 = nn.Linear(neurons_list[0], neurons_list[1])
        self.out2 = nn.Linear(neurons_list[1], 1)

        # self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        # self.apply(weight_init)
        self.to(device)

    def forward(self, state, action, get_q2=True):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.out1(q1)

        if get_q2:
            q2 = F.relu(self.fc3(sa))
            q2 = F.relu(self.fc4(q2))
            q2 = self.out2(q2)
            return q1, q2

        return q1

    def save(self, filename):
        torch.save(self.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))


class TD3(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 policy_noise,
                 noise_clip,
                 learning_rate=LEARNING_RATE,
                 discount=0.99,
                 tau=0.005,
                 action_noise=0.1,
                 batch_size=BATCH_SIZE,
                 policy_freq=TARGET_REPLACE_CNT):

        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        # self.critic_target = CriticNetwork(state_dim, action_dim)
        # self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau  # weight of current to network update the target network
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip  # range of noise
        self.policy_freq = policy_freq
        self.batch_size = batch_size

        self.actors_set = set()
        self.actors = []
        self.actor_targets = []
        self.actor_optimisers = []

        self.learn_iterator = 0
        self.memory_iterator = 0
        self.memory = collections.deque(maxlen=MEMORY_CAPACITY)
        self.loss_func = F.mse_loss
        self.device = device

    # Select action according to policy and add clipped noise
    # def choose_action(self, state, actor, train=True):
    #     actor.eval()
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    #     action = actor.forward(state).squeeze()
    #
    #     if train:
    #         noise = torch.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
    #                              dtype=torch.float).to(device)
    #         action = torch.clamp(action + noise, -1, 1)
    #     actor.train()
    #
    #     return action.detach().cpu().numpy()

    def store_transition(self, s, a, r, s_, done_mask):
        transition = (s, a, r, s_, done_mask)
        self.memory.append(transition)
        self.memory_iterator += 1
        # if self.memory_iterator % 100000 == 0:
        #     print("memory_iterator:")
        #     print(self.memory_iterator)

    def sample_state(self, nr_of_steps_act, batch_size):
        states = []
        for _ in range(nr_of_steps_act):
            state = []
            mini_batch = random.sample(self.memory, batch_size)
            for transition in mini_batch:
                s, a, r, s_prime, done_mask = transition
                state.append(s)
            state = torch.FloatTensor(np.array(state)).to(self.device)
            states.append(state)
        return states

    # randomly sample mini_batch transitions from memory
    def sample_transition(self, batch_size=BATCH_SIZE):
        mini_batch = random.sample(self.memory, batch_size)
        batch_s, batch_a, batch_r, batch_s_prime, batch_done = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            batch_s.append(s)
            batch_a.append(a)
            batch_r.append([r])
            batch_s_prime.append(s_prime)
            batch_done.append([done_mask])

        batch_s = torch.FloatTensor(np.array(batch_s)).to(self.device)
        batch_a = torch.FloatTensor(np.array(batch_a)).to(self.device)
        batch_r = torch.FloatTensor(np.array(batch_r)).to(self.device)
        batch_s_prime = torch.FloatTensor(np.array(batch_s_prime)).to(self.device)
        batch_done = torch.FloatTensor(np.array((batch_done))).to(self.device)

        return batch_s, batch_a, batch_r, batch_s_prime, batch_done

    def add_species(self, archive):
        # 检查是否存在新的种群
        diff = set(archive.keys()) - self.actors_set
        for desc in diff:
            self.actors_set.add(desc)
            a = archive[desc].x
            new_actor = copy.deepcopy(a)
            # for param in new_actor.parameters():
            #     param.requires_grad = True
            actor_target = copy.deepcopy(new_actor)
            optimizer = torch.optim.Adam(new_actor.parameters(), lr=3e-4)
            self.actors.append(new_actor)
            self.actor_targets.append(actor_target)
            self.actor_optimisers.append(optimizer)

    def update_network_parameters(self, state, tau=None):
        if tau is None:
            tau = self.tau
        for idx, actor in enumerate(self.actors):
            # 计算 actor loss， 批量梯度下降
            actor_loss = -self.critic(state, actor(state), get_q2=False).mean()
            # Optimize the actor
            self.actor_optimisers[idx].zero_grad()

            actor_loss.backward()
            self.actor_optimisers[idx].step()

            # 更新 actor_target
            for actor_params, actor_target_params in zip(self.actors[idx].parameters(),
                                                         self.actor_targets[idx].parameters()):
                actor_target_params.data.copy_(tau * actor_params + (1 - tau) * actor_target_params)

        # 更新 critic_target
        for critic_params, critic_target_params in zip(self.critic.parameters(),
                                                       self.critic_target.parameters()):
            critic_target_params.data.copy_(tau * critic_params + (1 - tau) * critic_target_params)

    def train(self, archive, n_crit, batch_size=256):
        # if ddpg.memory_iterator > MEMORY_CAPACITY:
        if self.memory_iterator < BATCH_SIZE:
            return
        self.add_species(archive)

        critic_loss = 0
        for _ in range(n_crit):
            self.learn_iterator += 1
            state, action, reward, next_state, done = self.sample_transition(batch_size)
            all_target_Q = torch.zeros(batch_size, len(self.actors), device=device)
            # question
            with torch.no_grad():
                noise = (
                        torch.randn_like(action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                for idx, actor in enumerate(self.actors):
                    next_action = (self.actor_targets[idx](next_state) + noise).clamp(-self.max_action, self.max_action)
                    target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                    target_Q = torch.min(target_Q1, target_Q2)
                    all_target_Q[:, idx] = target_Q.squeeze()

                target_Q = torch.max(all_target_Q, dim=1, keepdim=True)[0]
                # target_Q = reward +  done * self.discount * target_Q
                target_Q = reward + (1.0 - done) * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            # critic_loss = self.loss_func(current_Q1, target_Q) + self.loss_func(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.learn_iterator % self.policy_freq == 0:
                self.update_network_parameters(state)
        return critic_loss

# def plot(name, rewards):
#     plt.title(name + "curve of DQN for CartPole-v1")
#     # plt.title("learning curve of Sarsa for Taxi-v3")
#     plt.xlabel('epsiodes')
#     plt.ylabel('rewards')
#     plt.plot(rewards, label='rewards')
#     plt.show()
