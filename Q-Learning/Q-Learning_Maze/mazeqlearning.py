"""
Train a Q Learner in a navigation problem.
"""

import numpy as np
import random as rand
import QLearner as ql
import Maze

import time

# convert the position to a single integer state
def to_state(maze, pos):
    # TODO
    s = pos[0] * maze.data.shape[1] + pos[1]
    return s


# train learner to go through maze multiple epochs
# each epoch involves one trip from start to the goal or timeout before reaching the goal
# return list of rewards of each trip
def train(maze, learner, epochs=2000, timeout=5, verbose=False):
    # TODO
    # rewards = np.zeros(epochs)
    total_reward = 0
    # while not at goal and not timeout:
    time_start = time.time()  # 开始计时
    for i in range(epochs):
        if time.time() - time_start > timeout:
            print("timeout")
            break

        learner.rar = 0.3
        robopos = maze.get_start_pos()
        action = learner.querysetstate(to_state(maze, robopos))

        while robopos != maze.get_goal_pos():
            if  time.time() - time_start > timeout:
                break
            newpos, reward = maze.move(robopos, action)
            robopos = newpos
            action = learner.query(to_state(maze, robopos), reward)
            total_reward += reward
    return total_reward /epochs


# run the code to train a learner on a maze
def maze_qlearning(filename):
    # TODO
    # initialize maze object
    data = np.genfromtxt(filename, delimiter=',')
    maze = Maze.Maze(data)
    # initialize learner object
    learner = ql.QLearner(verbose=False)
    # execute train(maze, learner)
    reward = train(maze, learner)
    # return median of all rewards
    return reward


if __name__ == "__main__":
    rand.seed(5)
    maze_qlearning('testworlds/world01.csv')
