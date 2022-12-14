"""
Template for implementing Maze
"""

import numpy as np
import random as rand


class Maze(object):
    def __init__(self,
                 data,
                 reward_walk=-1,
                 reward_init=-10,
                 reward_obstacle=-1,
                 reward_quicksand=-100,
                 reward_goal=1,
                 random_walk_rate=0.2,
                 verbose=False):

        # TODO
        self.reward_init = reward_init
        self.data = data
        self.reward_walk = reward_walk
        self.reward_obstacle = reward_obstacle
        self.reward_quicksand = reward_quicksand
        self.reward_goal = reward_goal
        self.random_walk_rate = random_walk_rate
        self.verbose = verbose

    # return the start position of the robot
    def get_start_pos(self):
        # TODO
        # return (0,0)
        data = self.data
        for r in range(0, data.shape[0]):
            for c in range(0, data.shape[1]):
                if self.data[r][c] == 2:
                    return (r, c)

    # return the goal position of the robot
    def get_goal_pos(self):
        # TODO
        # return (0,0)
        data = self.data
        for r in range(0, data.shape[0]):
            for c in range(0, data.shape[1]):
                if self.data[r][c] == 3:
                    return (r, c)

    # move the robot and report new position and reward
    # Note that robot cannot step into obstacles nor step out of the map
    # Note that robot may ignore the given action and choose a random action
    def move(self, oldpos, a):
        # TODO
        data = self.data
        if rand.random() < self.random_walk_rate:
            a = rand.randint(0, 4 - 1)
        if a == 0:
            newpos = (oldpos[0] - 1, oldpos[1])
        elif a == 1:
            newpos = (oldpos[0] + 1, oldpos[1])
        elif a == 2:
            newpos = (oldpos[0], oldpos[1] - 1)
        else:
            newpos = (oldpos[0], oldpos[1] + 1)
        if not (0 <= newpos[0] < data.shape[0]
                and 0 <= newpos[1] < data.shape[1]) or data[newpos[0]][newpos[1]] == 1:
            newpos = oldpos
            reward = self.reward_obstacle
        elif data[newpos[0]][newpos[1]] == 0:
            reward = self.reward_walk
        elif data[newpos[0]][newpos[1]] == 2:
            reward = self.reward_init
        elif data[newpos[0]][newpos[1]] == 5:
            reward = self.reward_quicksand
        else:
            reward = 1

        # return the new, legal location and reward
        return newpos, reward

    # print out the map
    def print_map(self):
        data = self.data
        print("--------------------")
        for row in range(0, data.shape[0]):
            for col in range(0, data.shape[1]):
                if data[row, col] == 0:  # Empty space
                    print(" ", end="")
                if data[row, col] == 1:  # Obstacle
                    print("X", end="")
                if data[row, col] == 2:  # Start
                    print("S", end="")
                if data[row, col] == 3:  # Goal
                    print("G", end="")
                if data[row, col] == 5:  # Quick sand
                    print("~", end="")
            print()
        print("--------------------")

    # print the map and the trail of robot
    def print_trail(self, trail):
        data = self.data
        trail = data.copy()
        for pos in trail:

            # check if position is valid
            if not (0 <= pos[0] < data.shape[0]
                    and 0 <= pos[1] < data.shape[1]):
                print("Warning: Invalid position in trail, out of the world")
                return

            if data[pos] == 1:  # Obstacle
                print("Warning: Invalid position in trail, step on obstacle")
                return

            # mark the trail
            if data[pos] == 0:  # mark enter empty space
                trail[pos] = "."
            if data[pos] == 5:  # make enter quicksand
                trail[pos] = "@"

        print("--------------------")
        for row in range(0, trail.shape[0]):
            for col in range(0, trail.shape[1]):
                if trail[row, col] == 0:  # Empty space
                    trail[row, col] = " "
                if trail[row, col] == 1:  # Obstacle
                    trail[row, col] = "X"
                if trail[row, col] == 2:  # Start
                    trail[row, col] = "S"
                if trail[row, col] == 3:  # Goal
                    trail[row, col] = "G"
                if trail[row, col] == 5:  # Quick sand
                    trail[row, col] = "~"

                print(trail[row, col], end="")
            print()
        print("--------------------")
