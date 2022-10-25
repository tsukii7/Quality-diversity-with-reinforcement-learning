import numpy as np
import random as rand


class QLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,  # float, 更新Q-table时的学习率，范围 0.0 ~ 1.0, 常用值 0.2
                 gamma=0.9,  # float, 更新Q-table时的衰减率，范围 0.0 ~ 1.0, 常用值 0.9
                 rar=0.5,  # float, 随机行为比例, 每一步随机选择行为的概率。范围 0.0（从不随机） ~ 1.0（永远随机）, 常用值 0.5
                 radr=0.99,  # float, 随机行为比例衰减率, 每一步都更新 rar = rar * radr. 0.0（直接衰减到0） ~ 1.0（从不衰减）, 常用值 0.99
                 verbose=True):  # boolean, 如果为真，你的类可以打印调试语句，否则，禁止所有打印语句

        self.q_table = np.zeros((num_states, num_actions))
        self.verbose = verbose
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

    def querysetstate(self, s):
        self.s = s
        q_table = self.q_table
        self.rar = self.rar * self.radr
        action = 0
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            for i in range(self.num_actions):
                if q_table[s][i] > q_table[s][action]:
                    action = i
        if self.verbose: print("s =", s, "a =", action)
        return action

    def query(self, s_prime, r):
        # TODO
        q_table = self.q_table
        self.rar = self.rar * self.radr
        action = 0
        # max_q = np.max(q_table[s_prime,:])
        max_q = q_table[s_prime][action]
        if rand.random() < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            for i in range(self.num_actions):
                if q_table[s_prime][i] > q_table[s_prime][action]:
                    action = i
                    max_q = q_table[s_prime][action]
        q_table[self.s][self.a] += self.alpha*(r + self.gamma*max_q-q_table[self.s][self.a])   # q-learning off-policy
        self.s = s_prime
        self.a = action
        if self.verbose: print("s =", s_prime, "a =", action, "r =", r)
        return action
