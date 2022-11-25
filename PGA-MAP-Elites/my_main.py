import argparse
import sys

import gym
import QDgym

from my_TD3 import TD3, ActorNetwork
from my_utils import *
from functools import partial
from sklearn.neighbors import KDTree
# from my_DDPG import *
from my_variational_op import *
from evaluation import train_critic, evaluate


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as f:
            parser.parse_args([s.strip("\n") for s in f.readlines()], namespace)


def make_env(env_id):
    env = gym.make(env_id)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=open,
                        action=LoadFromFile)  # Config file to load args (Typically you would only specifiy this arg)
    parser.add_argument("--env", default="QDAntBulletEnv-v0")  # Environment name (only QDRL envs will run)
    parser.add_argument("--seed", default=0, type=int)  # Seed
    parser.add_argument("--save_path", default=".")  # Path where to save results
    ##########################################################################################################
    ########################## QD PARAMS #####################################################################
    ##########################################################################################################
    parser.add_argument("--dim_map", default=4, type=int)  # Dimentionality of behaviour space
    parser.add_argument("--n_niches", default=1296, type=int)  # nr of niches/cells of behaviour
    parser.add_argument("--n_species", default=1,
                        type=int)  # nr of species/cells in species archive (The species archive is disabled in the GECCO paper by setting n_species=1. See readme for details)
    parser.add_argument("--max_evals", default=1e6, type=int)  # nr of evaluations (I)
    parser.add_argument("--mutation_op", default=None)  # Mutation operator to use (Set to None in GECCO paper)
    parser.add_argument("--crossover_op",
                        default="iso_dd")  # Crossover operator to use (Set to iso_dd aka directional variation in GECCO paper which uses mutation and crossover in one)
    parser.add_argument("--min_genotype",
                        default=False)  # Minimum value a gene in the genotype can take (if False no limit) (Set to False in GECCO paper)
    parser.add_argument("--max_genotype",
                        default=False)  # Maximum value a gene in the genotype can take (if False no limit) (Set to False in GECCO paper)
    parser.add_argument("--mutation_rate", default=0.05,
                        type=float)  # Probablity of a gene to be mutated (Not used in GECCO paper. iso_dd mutates all genes unconditionally)
    parser.add_argument("--crossover_rate", default=0.75,
                        type=float)  # Probablity of genotypes being crossed over (Not used in GECCO paper. iso_dd crosses all genes unconditionally)
    parser.add_argument("--eta_m", default=5.0,
                        type=float)  # Parameter for polynomaial mutation (Not used in GECCO paper)
    parser.add_argument("--eta_c", default=10.0,
                        type=float)  # Parameter for Simulated Binary Crossover (Not used in GECCO paper)
    parser.add_argument("--sigma", default=0.2,
                        type=float)  # Sandard deviation for gaussian muatation (Not used in GECCO paper)
    parser.add_argument("--iso_sigma", default=0.01,
                        type=float)  # Gaussian parameter in iso_dd/directional variation (sigma_1)
    parser.add_argument("--line_sigma", default=0.2,
                        type=float)  # Line parameter in iso_dd/directional variation (sigma_2)
    parser.add_argument("--max_uniform", default=0.1,
                        type=float)  # Max mutation for uniform muatation (Not used in GECCO paper)
    parser.add_argument("--cvt_samples", default=100000,
                        type=int)  # Nr. of samples to use when approximating archive cell-centroid locations
    parser.add_argument("--eval_batch_size", default=100,
                        type=int)  # Batch size for parallel evaluation of policies (b)
    parser.add_argument("--random_init", default=500, type=int)  # Number of random evaluations to inililise (G)
    parser.add_argument("--init_batch_size", default=100,
                        type=int)  # Batch size for parallel evaluation during random init (b)
    parser.add_argument("--save_period", default=10000, type=int)  # How many evaluations between saving archives
    parser.add_argument("--num_cpu", default=32, type=int)  # Nr. of CPUs to use in parallel evaluation
    parser.add_argument("--num_cpu_var", default=32, type=int)  # Nr. of CPUs to use in parallel variation
    parser.add_argument("--use_cached_cvt",
                        action="store_true")  # Use cached centroids for creating archive if avalable
    parser.add_argument("--not_discard_dead",
                        action="store_true")  # Don't discard solutions that does not survive the entire simulation (Set to not dicard in GECCO paper)
    parser.add_argument("--neurons_list", default="128 128",
                        type=str)  # List of neurons in actor network layers. Network will be of form [neurons_list + [action dim]]
    #########################################################################################################
    ######################### RL PARAMS #####################################################################
    #########################################################################################################
    parser.add_argument("--train_batch_size", default=256, type=int)  # Batch size for both actors and critic (N)
    parser.add_argument("--discount", default=0.99)  # Discount factor for critic (gamma)
    parser.add_argument("--tau", default=0.005, type=float)  # Target networks update rate (tau)
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target during critic update (sigma_p)
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target noise (c)
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed actor updates (d)
    parser.add_argument('--nr_of_steps_crit', default=300,
                        type=int)  # Nr of. training steps for critic traning (n_crit)
    parser.add_argument('--nr_of_steps_act', default=10, type=int)  # Nr of. training steps for PG varaiation (n_grad)
    parser.add_argument("--proportion_evo", default=0.5,
                        type=float)  # Proportion of batch to use GA variation (n_evo = proportion_evo * b. Set to 0.5 in GECCO paper)
    parser.add_argument("--normalise", action="store_true")  # Use layer norm (Not used in GECCO paper)
    parser.add_argument("--affine",
                        action="store_true")  # Use affine transormation with layer norm (Not used in GECCO paper)
    parser.add_argument("--gradient_op", action="store_true")  # Use PG variation
    parser.add_argument("--lr", default=0.001, type=float)  # Learning rate PG variation

    args = parser.parse_args()
    args.neurons_list = [int(x) for x in args.neurons_list.split()]

    # 获取state_dim action_dim max_action:动作空间的最大取值
    temp_env = gym.make(args.env)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    max_action = float(temp_env.action_space.high[0])
    temp_env.close()

    # Compute CVT for main and species archive
    c = cvt(args.n_niches, args.dim_map, args.cvt_samples, args.use_cached_cvt)
    sc = cvt(args.n_species, args.dim_map, args.cvt_samples, args.use_cached_cvt)
    # k-nn for achive addition. The nearest centroid is found by this by setting k=1.
    kdt = KDTree(c, leaf_size=30, metric='euclidean')  # main k-nn
    s_kdt = KDTree(sc, leaf_size=30, metric='euclidean')  # species k-nn

    # create empty archive
    archive = {}
    s_archive = {}

    # initialize critic network
    agent = TD3(state_dim, action_dim, max_action)

    # 创建环境
    env = make_env(args.env)

    # map-elites loop
    n_evals = 0
    while n_evals < args.max_evals:
        print(f"Number of solutions:{len(archive)}")
        print(f"Number of species:{len(s_archive)}")
        to_evaluate = []
        if n_evals < args.random_init:
            print("random loop")
            for i in range(args.init_batch_size):
                actor = ActorNetwork(state_dim, action_dim, max_action)
                to_evaluate.append(actor)

        else:
            print("selection/variation")
            # TODO: train an actor
            critic, actors, states = train_critic(agent, s_archive, args.nr_of_steps_act, args.nr_of_steps_crit)
            to_evaluate += actors

            # variation
            to_evaluate += variation(archive, args.eval_batch_size - len(actors), 1)  # TODO:实现完pg改proportion

        # solution: (fitness, desc, alive)
        solutions = evaluate(env, to_evaluate, agent)
        n_evals += len(to_evaluate)

        # add to
        to_archive = []
        to_s_archive = []
        for index, solution in enumerate(solutions):
            fitness = solution[0]
            desc = solution[1]
            isAlive = solution[2]
            if isAlive or args.not_discard_dead:
                s = Individual(to_evaluate[index], desc, fitness)
                to_archive.append(s)
                to_s_archive.append(s)

        add_to_archive(to_archive, archive, kdt)
        add_to_archive(to_s_archive, s_archive, s_kdt, main=False)


        max_fitness = -sys.maxsize
        sum_fit = 0
        for x in archive.values():
            sum_fit += x.fitness
            if x.fitness > max_fitness:
                max_fitness = x.fitness
        print(f"[{n_evals}/{int(args.max_evals)}]",  flush=True)
        print(f"Max fitness: {max_fitness}")
        print(f"Mean fitness: {sum_fit/len(archive)}")

    env.close()

