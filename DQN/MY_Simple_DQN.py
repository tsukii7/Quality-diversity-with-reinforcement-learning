import gym
import collections
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON = 0.98  # greedy policy
MEMORY_CAPACITY = 2000
BATCH_SIZE = 32
TARGET_REPLACE_CNT = 200
EPISIODE_CNT = 500
N_STATES = 4
N_ACTIONS = 2

TIME_LIMIT = 600
TEST_EPISODE_CNT = 50

losses = []
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        # self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 128)
        # self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128, N_ACTIONS)
        # self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net = Net()
        self.target_net = Net()
        self.learn_iterator = 0
        self.memory_iterator = 0
        self.memory = collections.deque(maxlen=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.loss_func = nn.MSELoss()
        self.target_net.load_state_dict(self.eval_net.state_dict())  # q_target

    def choose_action(self, x):
        # E-greedy
        coin = random.random()
        if coin > EPSILON:
            return random.randint(0, 1)
        else:
            x = torch.from_numpy(x).float()
            actions_value = self.eval_net.forward(x)
            action = actions_value.argmax().item()
            return action

    def store_transition(self, s, a, r, s_, done_mask):
        transition = (s, a, r, s_, done_mask)
        self.memory.append(transition)
        self.memory_iterator += 1

    def learn(self):
        # update target network with evaluation network
        if self.learn_iterator % TARGET_REPLACE_CNT == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_iterator += 1
        cnt = 0
        for i in range(10):
            # randomly sample mini_batch transitions from memory
            mini_batch = random.sample(self.memory, BATCH_SIZE)
            batch_s, batch_a, batch_r, batch_s_prime, batch_done = [], [], [], [], []

            for transition in mini_batch:
                s, a, r, s_prime, done_mask = transition
                batch_s.append(s)
                batch_a.append([a])
                batch_r.append([r])
                batch_s_prime.append(s_prime)
                batch_done.append([done_mask])

            batch_s = torch.tensor(batch_s, dtype=torch.float)
            batch_a = torch.tensor(batch_a)
            batch_r = torch.tensor(batch_r)
            batch_s_prime = torch.tensor(batch_s_prime, dtype=torch.float)
            batch_done = torch.tensor(batch_done, dtype=torch.float)

            # train evaluation network
            q_eval = self.eval_net(batch_s).gather(1, batch_a)  # shape (batch, 1)
            q_next = self.target_net(batch_s_prime).detach()  # detach from graph, don't backpropagate
            # target = batch_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)    # shape (batch, 1)
            target = batch_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1) * batch_done  # shape (batch, 1)
            loss = self.loss_func(q_eval, target)
            if cnt % 1000 == 0:
                losses.append(loss.detach().numpy())
            cnt += 1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def plot(name, rewards):
    plt.title(name + "curve of DQN for CartPole-v1")
    # plt.title("learning curve of Sarsa for Taxi-v3")
    plt.xlabel('epsiodes')
    plt.ylabel('rewards')
    plt.plot(rewards, label='rewards')
    plt.show()


if __name__ == '__main__':
    # initialize
    time_start = time.time()
    env = gym.make('CartPole-v1')  # 加载环境
    dqn = DQN()
    rewards = []
    print('\nCollecting experience...')
    for i_episode in range(EPISIODE_CNT):
        if time.time() - time_start > TIME_LIMIT:
            break
        s = env.reset()
        ep_r = 0
        while True:
            # env.render()
            action = dqn.choose_action(s)
            s_prime, r, done, info, = env.step(action)
            done_mask = 0.0 if done else 1.0
            dqn.store_transition(s, action, r, s_prime, done_mask)
            ep_r += r
            if dqn.memory_iterator > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                rewards.append(ep_r)
                break
            s = s_prime
    plot('learning ', rewards)
    plot('loss ', losses)
    rewards = []
    for i in range(TEST_EPISODE_CNT):
        s = env.reset()
        env.render()
        done = False
        score = 0.0
        while not done:
            action = dqn.choose_action(s)
            s_prime, r, done, info = env.step(action)
            s = s_prime  # s进入下一个state
            score += r
            if done:
                rewards.append(score)
                break
        print("test  score:{:.1f}".format(score))
    plot('testing ', rewards)
    # torch.save(dqn.target_net, 'target_net.pkl')  # 保存整个网络

    torch.save(dqn.target_net.state_dict(), 'target_net.pkl')
