import copy

import gym
import collections
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

LEARNING_RATE = 0.0003
GAMMA = 0.95
EPSILON = 0.98  # greedy policy
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 32
TARGET_REPLACE_CNT = 2
EPISIODE_CNT = 500
TRAIN_STEP_CNT = 500
# STATES_DIM = 4
# ACTIONS_DIM = 2

TIME_LIMIT = 600
TEST_EPISODE_CNT = 50

losses = []

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, neurons_list=None):
        super(ActorNetwork, self).__init__()
        if neurons_list is None:
            neurons_list = [128, 128]
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, neurons_list[0])
        # self.fc1.weight.data.normal_(0, 0.1)
        self.ln1 = nn.LayerNorm(neurons_list[0])
        self.fc2 = nn.Linear(neurons_list[0], neurons_list[1])
        # self.fc2.weight.data.normal_(0, 0.1)
        self.ln2 = nn.LayerNorm(neurons_list[1])
        self.fc3 = nn.Linear(neurons_list[1], action_dim)
        # self.fc2.weight.data.normal_(0, 0.1)
        self.ln3 = nn.LayerNorm(action_dim)
        # self.out = nn.Linear(256, action_dim)
        # self.out.weight.data.normal_(0, 0.1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state):
        # print(state)
        action_value = F.relu(self.ln1(self.fc1(state)))
        action_value = F.relu(self.ln2(self.fc2(action_value)))
        action_value = torch.tanh(self.ln3(self.fc3(action_value)))
        action_value = self.max_action * action_value
        # action_value = self.out(state)
        return action_value

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self(state).detach().cpu().data.numpy().flatten()
        return action

    def save(self, filename):
        torch.save(self.state_dict(), filename)

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

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.apply(weight_init)
        self.to(device)

    def forward(self, state, action, get_q2=True):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.out1(q1)

        if get_q2:
            q2 = F.relu(self.fc1(sa))
            q2 = F.relu(self.fc2(q2))
            q2 = self.out1(q2)
            return q1, q2

        return q1
        # return q1

    def save(self, filename):
        torch.save(self.state_dict(), filename)
        torch.save(self.critic_optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.load_state_dict(torch.load(filename))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))


class TD3(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.005,
                 action_noise=0.1,
                 batch_size=BATCH_SIZE,
                 policy_freq=TARGET_REPLACE_CNT,
                 policy_noise=0.2,
                 noise_clip=0.5):
        # self.actor = ActorNetwork(state_dim, action_dim, max_action)
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau  # weight of current to network update the target network
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip  # range of noise
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.total_it = 0

        self.actors_set = set()
        self.actors = []
        self.actor_targets = []
        self.actor_optimisers = []

        self.learn_iterator = 0
        self.memory_iterator = 0
        self.memory = collections.deque(maxlen=MEMORY_CAPACITY)
        self.loss_func = nn.MSELoss()

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

    def sample_state(self, batch_size, steps):
        states = []
        mini_batch = random.sample(self.memory, steps)
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            states.append(s)
        states = torch.FloatTensor(np.array(states))
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

        batch_s = torch.FloatTensor(np.array(batch_s))
        batch_a = torch.FloatTensor(np.array(batch_a))
        batch_r = torch.FloatTensor(np.array(batch_r))
        batch_s_prime = torch.FloatTensor(np.array(batch_s_prime))
        batch_done = torch.FloatTensor(np.array((batch_done)))

        return batch_s, batch_a, batch_r, batch_s_prime, batch_done

    def add_species(self, archive):
        # check if found new species
        diff = set(archive.keys()) - self.actors_set
        for desc in diff:
            # add new species to the critic training pool
            self.actors_set.add(desc)
            a = archive[desc].x
            new_actor = copy.deepcopy(a)
            for param in new_actor.parameters():
                param.requires_grad = True
            # new_actor.parent_1_id = a.id
            # new_actor.parent_2_id = None
            # new_actor.type = "critic_training"
            actor_target = copy.deepcopy(new_actor)
            optimizer = torch.optim.Adam(new_actor.parameters(), lr=3e-4)
            self.actors.append(new_actor)
            self.actor_targets.append(actor_target)
            self.actor_optimisers.append(optimizer)

    def update_network_parameters(self, state, tau=None):
        if tau is None:
            tau = self.tau
        for idx, actor in enumerate(self.actors):
            # Compute actor loss
            actor_loss = -self.critic(state, actor(state), get_q2=False).mean()
            # Optimize the actor
            self.actor_optimisers[idx].zero_grad()

            actor_loss.backward()
            self.actor_optimisers[idx].step()
            for actor_params, actor_target_params in zip(self.actors[idx].parameters(),
                                                         self.actor_targets[idx].parameters()):
                actor_target_params.data.copy_(tau * actor_params + (1 - tau) * actor_target_params)

        for critic_params, critic_target_params in zip(self.critic.parameters(),
                                                       self.critic_target.parameters()):
            critic_target_params.data.copy_(tau * critic_params + (1 - tau) * critic_target_params)

    def train(self, archive, n_crit, batch_size=256):
        # if ddpg.memory_iterator > MEMORY_CAPACITY:
        if self.memory_iterator < BATCH_SIZE:
            return
        self.add_species(archive)
        for _ in range(n_crit):
            self.learn_iterator += 1
            state, action, reward, next_state, not_done = self.sample_transition(batch_size)
            all_target_Q = torch.zeros(batch_size, len(self.actors), device=device)
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            for idx, actor in enumerate(self.actors):
                next_action = (self.actor_targets[idx](next_state) + noise).clamp(-self.max_action, self.max_action)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                all_target_Q[:, idx] = target_Q.squeeze()

            target_Q = torch.max(all_target_Q, dim=1, keepdim=True)[0]
            target_Q = reward + not_done * self.discount * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = self.loss_func(current_Q1, target_Q) + self.loss_func(current_Q2, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.learn_iterator % self.policy_freq == 0:
                self.update_network_parameters(state)
        return critic_loss

    # def learn(self):
    #     # update target network with evaluation network
    #     if len(self.memory) < BATCH_SIZE:
    #         return
    #
    #     if self.learn_iterator % TARGET_REPLACE_CNT == 0:
    #         self.critic_target.load_state_dict(self.critic.state_dict())
    #     self.learn_iterator += 1
    #     cnt = 0
    #     for i in range(10):
    #         batch_s, batch_a, batch_r, batch_s_prime, batch_done = self.sample_transition(BATCH_SIZE)
    #         # train evaluation network
    #         q_eval = self.critic(batch_s).gather(1, batch_a)  # shape (batch, 1)
    #         q_next = self.critic_target(batch_s_prime).detach()  # detach from graph, don't backpropagate
    #         # target = batch_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)    # shape (batch, 1)
    #         target = batch_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) * batch_done  # shape (batch, 1)
    #         loss = self.loss_func(q_eval, target)
    #         if cnt % 1000 == 0:
    #             losses.append(loss.detach().numpy())
    #         cnt += 1
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

# def plot(name, rewards):
#     plt.title(name + "curve of DQN for CartPole-v1")
#     # plt.title("learning curve of Sarsa for Taxi-v3")
#     plt.xlabel('epsiodes')
#     plt.ylabel('rewards')
#     plt.plot(rewards, label='rewards')
#     plt.show()
#
#
# if __name__ == '__main__':
#     # initialize
#     time_start = time.time()
#     env = gym.make('CartPole-v1')
#     ddpg = PGA_MAP_Elites()
#     rewards = []
#     print('\nCollecting experience...')
#     for i_episode in range(EPISIODE_CNT):
#         if time.time() - time_start > TIME_LIMIT:
#             break
#         s = env.reset()
#         ep_r = 0
#         while True:
#             # env.render()
#             action = ddpg.choose_action(s)
#             s_prime, r, done, info, = env.step(action)
#             done_mask = 0.0 if done else 1.0
#             ddpg.store_transition(s, action, r, s_prime, done_mask)
#             ep_r += r
#             if ddpg.memory_iterator > MEMORY_CAPACITY:
#                 ddpg.learn()
#                 if done:
#                     print('Ep: ', i_episode,
#                           '| Ep_r: ', round(ep_r, 2))
#
#             if done:
#                 rewards.append(ep_r)
#                 break
#             s = s_prime
#     plot('learning ', rewards)
#     plot('loss ', losses)
#     rewards = []
#     for i in range(TEST_EPISODE_CNT):
#         s = env.reset()
#         env.render()
#         done = False
#         score = 0.0
#         while not done:
#             action = ddpg.choose_action(s)
#             s_prime, r, done, info = env.step(action)
#             s = s_prime  # s进入下一个state
#             score += r
#             if done:
#                 rewards.append(score)
#                 break
#         print("test  score:{:.1f}".format(score))
#     plot('testing ', rewards)
#     # torch.save(dqn.target_net, 'target_net.pkl')  # 保存整个网络
#
#     ddpg.critic_target.save('target_net.pkl')
