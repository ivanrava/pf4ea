from generator import Instance, Path
import numpy as np
import matplotlib.pyplot as plt


def extract_min(structure, function):
    min_el = None
    min_score = np.inf
    for el in structure:
        score = function(el)
        if score < min_score:
            min_score = score
            min_el = el
    return min_el


def relaxed_path(instance: Instance):
    """
    Reversed Dijkstra algorithm
    :param instance: problem instance
    :return: the shortest path tree
    """
    # The source of Dijkstra is the final goal
    d = {instance.goal: 0}
    pi = {}
    S = set()
    Q = set(instance.adj.keys())
    while len(Q) > 0:
        u = extract_min(Q, lambda x: d[x] if x in d else np.inf)
        # TODO: very quick and dirty, can be optimized
        Q = Q.difference({u})
        S = S.union({u})
        for v in instance.adj[u]:
            v = v[0]
            if d[v] if v in d else np.inf > d[u] if u in d else np.inf + instance.grid.get_weight(u, v):
                d[v] = d[u] + instance.grid.get_weight(u, v)
                pi[v] = u
    return pi, d


def plot_dijkstra(instance, pi, d):
    new_grid = np.copy(instance.grid.grid)
    for cell, cost in d.items():
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


def is_collision_free(path: Path, other_paths: [Path], debug=False):
    for t in range(max([len(p) for p in other_paths] + [len(path)])):
        for other_path in other_paths:
            if path[t] == other_path[t]:
                if debug:
                    print(f"COLLISION: 2 agents both found on tile {path[t]} at instant {t}")
                return False
            if t == 0:
                continue
            if path[t] == other_path[t - 1] and other_path[t] == path[t - 1]:
                if debug:
                    print(f"COLLISION: swapped places on adjacent tiles {path[t]} and {other_path[t]}, at instant {t}")
                return False
            # Diagonal collisions
            delta1 = tuple(abs(np.subtract(path[t - 1], other_path[t])))
            delta2 = tuple(abs(np.subtract(path[t], other_path[t - 1])))
            delta3 = tuple(abs(np.subtract(path[t - 1], path[t])))
            delta4 = tuple(abs(np.subtract(other_path[t - 1], other_path[t])))
            if delta3 == delta4 == (1, 1) and delta1 == delta2 and delta1 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if debug:
                    print(f"COLLISION: crossing while going for tiles {path[t]} and {other_path[t]}, at instant {t}")
                return False
    return True


def is_pathset_collision_free(pathset):
    is_free = True
    for i in range(len(pathset)):
        free = is_collision_free(pathset[i], pathset[:i] + pathset[i + 1:], True)
        if not free:
            is_free = False
    return is_free
