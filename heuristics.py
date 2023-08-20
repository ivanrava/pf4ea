import utils
from generator import Instance, Path
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class Heuristic(ABC):
    def __init__(self, instance: Instance):
        self.instance = instance

    @abstractmethod
    def heuristic(self, vertex: (int, int)) -> float:
        pass

    @abstractmethod
    def costs(self) -> dict:
        pass

    def plot(self):
        new_grid = np.copy(self.instance.grid.grid)
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

    def relaxed_path_from(self, starting_cell):
        raise NotImplementedError("This strategy cannot relax paths. Please choose another")


def diagonal_heuristic(n, goal):
    dx = abs(n[1] - goal[1])
    dy = abs(n[0] - goal[0])
    return dx + dy + (np.sqrt(2) - 2) * min(dx, dy)


class Diagonal(Heuristic):
    def __init__(self, instance: Instance):
        super().__init__(instance)
        self.diagonals = {}
        for vertex in instance.adj.keys():
            self.diagonals[vertex] = diagonal_heuristic(vertex, instance.goal)

    def heuristic(self, vertex: (int, int)):
        return self.diagonals[vertex]

    def costs(self) -> dict:
        return self.diagonals


class DijkstraRelaxer(Heuristic):
    def __init__(self, instance: Instance):
        super().__init__(instance)
        self.pi = {}
        self.d = {}
        self.instance = instance
        self.__relax_dijkstra(instance)

    def __relax_dijkstra(self, instance: Instance):
        """
        Dijkstra algorithm
        :param instance: problem instance
        :return: the shortest path tree
        """
        # The source of Dijkstra is the final goal
        d = {instance.goal: 0}
        pi = {}
        S = set()
        Q = set(instance.adj.keys())
        while len(Q) > 0:
            u = utils.extract_min(Q, lambda x: d[x] if x in d else np.inf)
            # TODO: very quick and dirty, can be optimized
            Q = Q.difference({u})
            S = S.union({u})
            for v in instance.adj[u]:
                v = v[0]
                if (d[v] if v in d else np.inf) > (d[u] if u in d else np.inf) + instance.grid.get_weight(u, v):
                    d[v] = d[u] + instance.grid.get_weight(u, v)
                    pi[v] = u
        self.pi = pi
        self.d = d

    # FIXME #1: it can be memoized
    def relaxed_path_from(self, starting_cell):
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
