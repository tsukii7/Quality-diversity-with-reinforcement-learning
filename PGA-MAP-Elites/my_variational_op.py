import copy

import numpy as np
import torch


def variation(archive,
              batch_size,
              proportion_evo,
              learning_rate,
              critic,
              states,
              nr_of_steps_act):
    keys = list(archive.keys())

    actors_z = []
    actors_x_evo = []
    actors_y_evo = []

    # 产生actor父母
    rand_evo_1 = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
    rand_evo_2 = np.random.randint(len(keys), size=int(batch_size * proportion_evo))
    for n in range(0, len(rand_evo_1)):
        actors_x_evo += [archive[keys[rand_evo_1[n]]]]
        actors_y_evo += [archive[keys[rand_evo_2[n]]]]

    for n in range(len(actors_x_evo)):
        actor_x = actors_x_evo[n].x
        actor_y = actors_y_evo[n].x
        actor_z = copy.deepcopy(actor_x)
        actor_z_state_dict = crossover(actor_x.state_dict(), actor_y.state_dict())  # 将actor x和y的网络参数crossover
        actor_z.load_state_dict(actor_z_state_dict)
        actors_z.append(actor_z)

        # PG
    actors_x_grad = []
    rand_grad = np.random.randint(len(keys), size=(batch_size - int(batch_size * proportion_evo)))
    for n in range(0, len(rand_grad)):
        actors_x_grad += [archive[keys[rand_grad[n]]]]  # 取individual

    # 输入所有actor，一个critic， states，
    for individual in actors_x_grad:
        actor_z = individual.x
        actor_z_copy = copy.deepcopy(actor_z)
        # for param in actor_z_copy.parameters():
        #     param.requires_grad = True
        optimizer = torch.optim.Adam(actor_z_copy.parameters(), lr=learning_rate)
        for i in range(nr_of_steps_act):
            state = states[i]
            actor_loss = -critic(state, actor_z(state), get_q2=False).mean()
            optimizer.zero_grad()
            actor_loss.backward()
            optimizer.step()
        # for param in actor_z.parameters():
        #     param.requires_grad = False
        actors_z.append(actor_z)
    return actors_z


def crossover(actorx_state_dict, actory_state_dict):
    actor_z_state_dict = copy.deepcopy(actorx_state_dict)
    for tensor in actorx_state_dict:
        if "weight" or "bias" in tensor:
            actor_z_state_dict[tensor] = iso_dd(actorx_state_dict[tensor], actory_state_dict[tensor])
    return actor_z_state_dict


def iso_dd(x, y):
    '''
            Iso+Line
            Ref:
            Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
            GECCO 2018
            '''
    a = torch.zeros_like(x).normal_(mean=0, std=0.005)  # 生成一个与x维度一致的随机数矩阵，均值是0， 标准差是iso_sigma
    b = np.random.normal(0, 0.05)
    z = x.clone() + a + b * (y - x)
    return z
