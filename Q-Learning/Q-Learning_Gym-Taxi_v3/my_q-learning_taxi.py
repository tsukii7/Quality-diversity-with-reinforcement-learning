import gym
import my_QLearner as ql
import seaborn as sns
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")  # 创建出租车游戏环境
# env = gym.make("CartPole-v1")  # 创建cartpole游戏环境
state = env.reset()  # 初始化环境
envspace = env.observation_space.n  # 状态空间的大小
# envspace = env.observation_space.shape[0]  # 对于 Box 类型的连续空间具有有限数量的元素 n 是无效的，因此env.observation_space.n属性不存在。 观测空间的大小
actspace = env.action_space.n  # 动作空间的大小
print('状态空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('动作数 = {}'.format(env.action_space.n))
print('初始状态 = {}'.format(state))

# Q-learning
learner = ql.QLearner(num_states=envspace, num_actions=actspace, verbose=False)
# Q = np.zeros([envspace,actspace]) #创建一个Q-table

alpha = 0.5  # 学习率
rewards = []
for episode in range(1, 2001):
    done = False
    reward = 0  # 瞬时reward
    epi_reward = 0  # 累计reward
    state = env.reset()  # 状态初始化
    action = learner.querysetstate(state)
    learner.rar = 0.3
    while done != True:
        newpos, reward, done, info = env.step(action)
        state = newpos
        action = learner.query(state, reward)
        epi_reward += reward
        # env.render()
    if episode % 50 == 0:
        print('episode:{};\t\ttotal reward:{}'.format(episode,  epi_reward))
    if episode % 10 == 0:
        rewards.append(epi_reward)

# wandb.log({
#             'accumulated_reward': sum(rewards),
#             'loss': loss,
#             'avg log_likelihood': np.mean([log_l.detach().numpy() for log_l in action_log_likelihoods])
#         })

plt.title("learning curve of Q-learning for Taxi-v3")
# plt.title("learning curve of Sarsa for Taxi-v3")
plt.xlabel('epsiodes/10')
plt.ylabel('rewards')
plt.plot(rewards, label='rewards')
plt.show()

print('The Q table is:{}'.format(learner.q_table))

# 测试阶段
conter = 0
reward = None
state = env.reset()  # 状态初始化
done = False
rewards = []
# learner.rar = 0.3
while done != True:
    action = learner.querysetstate(state)
    state, reward, done, info = env.step(action)
    rewards.append(reward)
    conter = conter + 1
    env.render()
    # print(reward)
plt.title("learning curve of Q-learning for Taxi-v3")
# plt.title("learning curve of Sarsa for Taxi-v3")
plt.ylabel('rewards')
plt.xlabel('step')
plt.plot(rewards, label='reward')
plt.show()
