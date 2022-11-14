import gym
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
N_STATES = 4
N_ACTIONS = 2

# data=pd.read_pickle('target_net.pkl')
# print('data:\n',data)


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

def restore_net():#提取整个网络
    env = gym.make('CartPole-v1')  # 加载环境
    net = Net()
    net.load_state_dict(torch.load('target_net.pkl')) # 仅加载参数

    rewards = []
    for i in range(100):
        s = env.reset()
        env.render()
        done = False
        score = 0.0
        while not done:
            s = torch.from_numpy(s).float()
            actions_value = net.forward(s)
            action = actions_value.argmax().item()
            s_prime, r, done, info = env.step(action)
            s = s_prime  # s进入下一个state
            score += r
            if done:
                rewards.append(score)
                break
        print("test  score:{:.1f}".format(score))
    env.close()
    plt.title("learning curve of DQN for CartPole-v1")
    # plt.title("learning curve of Sarsa for Taxi-v3")
    plt.xlabel('epsiodes')
    plt.ylabel('rewards')
    plt.plot(rewards, label='rewards')
    plt.show()

if __name__ == '__main__':
    restore_net()

