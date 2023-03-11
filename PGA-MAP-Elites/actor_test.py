import gym
import numpy as np
import torch
import QDgym

from my_TD3 import ActorNetwork


def make_env(env_id):
    env = gym.make(env_id, render=True)
    # env = gym.make(env_id, render=False)
    return env


def evaluate(env, actor):
    state = env.reset()
    done = False
    # eval loop
    while not done:
        # env.render()
        # time.sleep(0.01)
        action = actor.select_action(np.array(state))
        next_state, reward, done, _ = env.step(action)
        state = next_state
    result = (env.tot_reward, env.desc, env.alive)
    return result


if __name__ == '__main__':
    temp_env = make_env("QDHalfCheetahBulletEnv-v0")
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    max_action = float(temp_env.action_space.high[0])
    temp_env.close()
    env = make_env("QDHalfCheetahBulletEnv-v0")
    actor = ActorNetwork(state_dim, action_dim, max_action)
    actor.load_state_dict(torch.load(r"C:\Users\Ksco\ProgramProjects\PycharmProjects\QDRL\PGA-MAP-Elites-master\models\PGA-MAP-Elites_QDHalfCheetahBulletEnv-v0_0_2_actor_98315"))

    # actor.load(
    #     "C:\Users\Ksco\ProgramProjects\PycharmProjects\QDRL\PGA-MAP-Elites-master\models\PGA-MAP-Elites_QDHalfCheetahBulletEnv-v0_0_2_actor_98315")
    result = evaluate(env, actor)
    print(f"Max fitness: {result[0]}")
