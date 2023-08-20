import utils
from generator import Instance, Path
import numpy as np
import matplotlib.pyplot as plt


# TODO #1: precalculate
def h_diagonal(n, goal):
    dx = abs(n[1] - goal[1])
    dy = abs(n[0] - goal[0])
    return dx + dy + (np.sqrt(2) - 2) * min(dx, dy)


class Relaxer:
    def __init__(self, instance: Instance):
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

    def plot(self):
        new_grid = np.copy(self.instance.grid.grid)
        for cell, cost in self.d.items():
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

    # FIXME #1: it can be memoized
    def relaxed_path_from(self, starting_cell):
        path = Path([starting_cell])
        while path[-1] != self.instance.goal:
            path.append(self.pi[path[-1]])
        return path

    # FIXME #1: it can be memoized
    def relaxed_heuristic(self, v: (int, int)):
        path = [v]
        cost = 0
        while path[-1] != self.instance.goal:
            path.append(self.pi[path[-1]])
            cost += self.d[path[-1]]
        return cost

