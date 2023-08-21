import random

import numpy as np
import matplotlib.pyplot as plt

import agents
from utils import Path
import solver


def get_random_boolean_weighted(weight):
    return (np.random.random_sample(1) < weight)[0]


class Grid:
    def __init__(self, height, width, obstacle_ratio=0.1, conglomeration_ratio=0.5):
        # Generates an empty grid
        self.grid = np.zeros((height, width))
        self.obstacle_ratio = obstacle_ratio
        self.conglomeration_ratio = conglomeration_ratio

        # Calculates how many obstacle cells we need
        self.num_obstacle_cells = int(self.grid.size * obstacle_ratio)

        obstacles = self.num_obstacle_cells
        # Adds obstacles
        while obstacles > 0:
            rand_i, rand_j = self.get_random_empty_cell()
            self.grid[rand_i][rand_j] = 1
            obstacles -= 1
            obstacles = self.__add_neighbor_obstacle((rand_i, rand_j), obstacles, conglomeration_ratio)

    # FIXME: should we allow movement across "diagonal obstacles"?
    def neighbors(self, el, also_diagonals):
        (i, j) = el
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if also_diagonals:
            directions += [(1, 1), (-1, -1), (-1, 1), (1, -1), (0, 0)]
        neighbors = []
        for curr_dir in directions:
            i_d, j_d = curr_dir
            if i + i_d < 0 or i + i_d >= self.grid.shape[0] or j + j_d < 0 or j + j_d >= self.grid.shape[1]:
                continue
            neighbors.append((i + i_d, j + j_d))
        return neighbors

    def empty_neighbors(self, el, also_diagonals=False):
        return [n for n in self.neighbors(el, also_diagonals) if self.grid[n] != 1]

    def get_random_empty_cell(self):
        rand_i, rand_j = np.random.randint(self.grid.shape[0]), np.random.randint(self.grid.shape[1])
        while self.grid[rand_i][rand_j] == 1:
            rand_i, rand_j = np.random.randint(self.grid.shape[0]), np.random.randint(self.grid.shape[1])
        return rand_i, rand_j

    def __add_neighbor_obstacle(self, starting_from, num_obstacle_cells, conglomeration_ratio):
        candidate_neighbors = self.empty_neighbors(starting_from, also_diagonals=False)
        while num_obstacle_cells > 0 and get_random_boolean_weighted(conglomeration_ratio) and len(
                candidate_neighbors) > 0:
            random.shuffle(candidate_neighbors)
            selected_neighbor = candidate_neighbors.pop()
            candidate_neighbors += self.empty_neighbors(selected_neighbor, also_diagonals=False)
            self.grid[selected_neighbor] = 1
            num_obstacle_cells -= 1
        else:
            return num_obstacle_cells

    def print(self):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] == 0:
                    print('.', end='')
                else:
                    print('█', end='')
            print()

    def plot(self, show=True):
        plt.title(
            f'{self.grid.shape[1]} x {self.grid.shape[0]}\nObstacles: {self.num_obstacle_cells} / {self.grid.size}')
        plt.pcolormesh(1 - self.grid, edgecolors='#777', linewidth=0.5, cmap='gray')
        plt.xticks(range(0, self.grid.shape[1], 5))
        plt.yticks(range(0, self.grid.shape[0], 2))
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect('equal')
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_verticalalignment("top")
        if show:
            plt.show()

    def get_weight(self, from_cell, to_cell):
        distance = abs(from_cell[0] - to_cell[0]) + abs(from_cell[1] - to_cell[1])
        return 1 if distance <= 1 else np.sqrt(2)

    def get_path_cost(self, path: Path):
        cost = 0
        for t in range(len(path) - 1):
            cost += self.get_weight(path[t], path[t + 1])
        return cost

    def to_adj(self):
        adj = {}
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                # Do not put obstacles "vertices" (they are not vertices) into the adjacency lists
                if self.grid[i, j] == 1:
                    continue
                empty_neighbors = self.empty_neighbors((i, j), also_diagonals=True)

                empty_neighbors = [(n, self.get_weight((i, j), n)) for n in empty_neighbors]
                adj[(i, j)] = empty_neighbors
        return adj

    def idxes_to_idx(self, idxes):
        i, j = idxes
        return i * self.grid.shape[0] + j


class Instance:
    def __init__(self, width, height,
                 num_agents=3,
                 conglomeration_ratio=0.5,
                 obstacle_ratio=0.1,
                 max_length=20,
                 agent_path_length=10,
                 agent_generator: agents.AgentGenerator = agents.RandomAgentGenerator(max_length=10)):

        self.grid = Grid(width, height, conglomeration_ratio=conglomeration_ratio, obstacle_ratio=obstacle_ratio)
        self.adj = self.grid.to_adj()

        self.init = self.grid.get_random_empty_cell()
        self.goal = self.grid.get_random_empty_cell()

        self.max_length = max_length
        if max_length > self.maximum_max_length(agent_path_length):
            print(f"Warning: max_length is too big. Setting max_length to {self.maximum_max_length(agent_path_length)}")
            self.max_length = self.maximum_max_length(agent_path_length)

        self.num_agents = num_agents
        self.paths, self.starting_positions = agent_generator.build_paths(num_agents, instance=self)

    def plot_instant(self, t, additional_path: Path):
        self.grid.plot(show=False)
        plt.title(f"t={t}")

        for pos in self.starting_positions:
            plt.plot(pos[1] + 0.5, pos[0] + 0.5, 'x', markersize=18)

        plt.plot(self.init[1] + 0.5, self.init[0] + 0.5, 'x', markersize=18, color='#ccc')
        plt.plot(self.goal[1] + 0.5, self.goal[0] + 0.5, '+', markersize=18, color='#ccc')

        plt.gca().set_prop_cycle(None)
        for path in self.paths:
            plt.plot(path[t][1] + 0.5, path[t][0] + 0.5, 's', markersize=12)

        plt.plot(additional_path[t][1] + 0.5, additional_path[t][0] + 0.5, '.', markersize=20, color='#aaa')

        plt.show()

    def plot(self, additional_path: Path):
        for t in range(max([len(path) for path in self.paths] + [len(additional_path)])):
            self.plot_instant(t, additional_path)

    def maximum_max_length(self, agent_path_length):
        return agent_path_length + self.grid.num_obstacle_cells

    def solve(self, heuristic):
        return solver.reach_goal(self.grid, self.adj, self.paths,
                                 self.init, self.goal, self.max_length,
                                 self.starting_positions, heuristic)
