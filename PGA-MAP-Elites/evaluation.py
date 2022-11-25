import copy
import time

import numpy as np


def evaluate(env, actors, agent, test_mode=False):
    results = []
    for actor in actors:
        state = env.reset()
        done = False
        # eval loop
        # state_array, action_array, next_state_array, reward_array, done_bool_array = [], [], [], [], []
        while not done:
            # env.render()
            # time.sleep(0.01)
            action = actor.select_action(np.array(state))
            next_state, reward, done, _ = env.step(action)
            done_bool = float(done) if env.T < env._max_episode_steps else 0
            if not test_mode:
                agent.store_transition(state, action, reward, next_state, done_bool)
        result = (env.tot_reward, env.desc, env.alive)
        results.append(result)
    return results


def train_critic(agent, archive, nr_of_steps_act, nr_of_steps_crit):
    # start critic training
    if len(agent.memory) > agent.batch_size and archive:  # hack as well
        t1 = time.time()
        # train critic
        critic_loss = agent.train(archive, nr_of_steps_crit)
        train_time = time.time() - t1
        out_actors = []
        print("critic.actors len: ", len(agent.actors))
        for actor in agent.actors:
            a = copy.deepcopy(actor)
            for param in a.parameters():
                param.requires_grad = False
            out_actors.append(a)
        states = agent.sample_state(agent.batch_size, nr_of_steps_act)
        print(f"Train Time: {train_time}")
        print(f"Critic Loss: {critic_loss.detach()}")
        return agent.critic_target, out_actors, states

        # state_array.append(state)
        # action_array.append(action)
        # next_state_array.append(next_state)
        # reward_array.append(reward)
        # done_bool_array.append(done_bool)
        # state = next_state

        # if env.T == 1:
        #     state_array = state
        #     action_array = action
        #     next_state_array = next_state
        #     reward_array = reward
        #     done_bool_array = done_bool
        # else:
        #     state_array = np.vstack((state, state_array))
        #     action_array = np.vstack((action, action_array))
        #     next_state_array = np.vstack((next_state, next_state_array))
        #     reward_array = np.vstack((reward, reward_array))
        #     done_bool_array = np.vstack((done_bool, done_bool_array))
