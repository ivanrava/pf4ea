import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Callable

from generator.grid import Grid
from generator.utils import Path


# TODO: add more heuristics


def extract_min(structure: set, function: Callable[[(int, int)], float]):
    min_el = next(iter(structure))
    min_score = function(min_el)
    for el in structure:
        score = function(el)
        if score < min_score:
            min_score = score
            min_el = el
    return min_el


class Heuristic(ABC):
    def __init__(self, raw_grid):
        self.grid = raw_grid

    @abstractmethod
    def heuristic(self, vertex: (int, int)) -> float:
        pass

    @abstractmethod
    def costs(self) -> dict:
        pass

    def plot(self):
        new_grid = np.copy(self.grid)
        for cell, cost in self.costs().items():
            new_grid[cell] = cost
        plt.pcolormesh(new_grid, edgecolors='#777', linewidth=0.5, cmap='gray')
        plt.xticks(range(0, new_grid.shape[1], 5))
        plt.yticks(range(0, new_grid.shape[0], 2))
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect('equal')
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_verticalalignment("top")
        plt.show()

    def relaxed_path_from(self, starting_cell: (int, int)):
        raise NotImplementedError("This strategy cannot relax paths. Please choose another")


def diagonal_heuristic(n: (int, int), goal: (int, int)):
    dx = abs(n[1] - goal[1])
    dy = abs(n[0] - goal[0])
    return dx + dy + (np.sqrt(2) - 2) * min(dx, dy)


class Diagonal(Heuristic):
    def __init__(self, grid: Grid, goal: (int, int)):
        super().__init__(grid)
        self.diagonals = {}
        for vertex in self.grid.adj.keys():
            self.diagonals[vertex] = diagonal_heuristic(vertex, goal)

    def heuristic(self, vertex: (int, int)):
        return self.diagonals[vertex]

    def costs(self) -> dict:
        return self.diagonals


class DijkstraRelaxer(Heuristic):
    def __init__(self, instance):
        super().__init__(instance.grid.grid)
        self.pi = {}
        self.d = {}
        self.instance = instance
        self.__relax_dijkstra(instance)

    def __relax_dijkstra(self, instance):
        """
        Dijkstra algorithm
        :param instance: problem instance
        :return: the shortest path tree
        """
        # The source of Dijkstra is the final goal
        d = {instance.goal: 0}
        pi = {}
        S = set()
        Q = set(instance.grid.adj.keys())
        while len(Q) > 0:
            u = extract_min(Q, lambda x: d[x] if x in d else np.inf)
            Q = Q - {u}
            S = S | {u}
            for v in instance.grid.adj[u]:
                v = v[0]
                if (d[v] if v in d else np.inf) > (d[u] if u in d else np.inf) + instance.grid.get_weight(u, v):
                    d[v] = d[u] + instance.grid.get_weight(u, v)
                    pi[v] = u
        self.pi = pi
        self.d = d

    # FIXME #1: it can be memoized
    def relaxed_path_from(self, starting_cell: (int, int)):
        path = Path([starting_cell])
        while path[-1] != self.instance.goal:
            try:
                path.append(self.pi[path[-1]])
            except KeyError:
                return None
        return path

    # FIXME #1: it can be memoized
    def heuristic(self, v: (int, int)):
        path = [v]
        cost = 0
        while path[-1] != self.instance.goal:
            try:
                path.append(self.pi[path[-1]])
                cost += self.d[path[-1]]
            except KeyError:
                return np.inf
        return cost

    def costs(self) -> dict:
        return self.d
