from generator import Instance
import numpy as np


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
        print(len(Q))
        u = extract_min(Q, lambda x: d[x] if x in d else np.inf)
        # TODO: very quick and dirty, can be optimized
        Q = Q.difference({u})
        S = S.union({u})
        for v in instance.adj[u]:
            v = v[0]
            if d[v] if v in d else np.inf > d[u] if u in d else np.inf + instance.grid.get_weight(u, v):
                d[v] = d[u] + instance.grid.get_weight(u, v)
                pi[v] = u
    return pi


# TODO: apply OOP Python shenanigans
def query_path(path, t):
    try:
        return path[t]
    except IndexError:
        return path[-1]


def is_collision_free(path, other_paths, debug=False):
    for t in range(max([len(p) for p in other_paths] + [len(path)])):
        for other_path in other_paths:
            if query_path(path, t) == query_path(other_path, t):
                if debug:
                    print(f"COLLISION: 2 agents both found on tile {query_path(path, t)} at instant {t}")
                return False
            if t == 0:
                continue
            if query_path(path, t) == query_path(other_path, t-1) and query_path(other_path, t) == query_path(path, t-1):
                if debug:
                    print(f"COLLISION: 2 agents swapped places on adjacent tiles {query_path(path, t)} and {query_path(other_path, t)}, at instant {t}")
                return False
            # Diagonal collisions
            delta1 = tuple(abs(np.subtract(query_path(path, t - 1), query_path(other_path, t))))
            delta2 = tuple(abs(np.subtract(query_path(path, t), query_path(other_path, t - 1))))
            if delta1 == delta2 and delta1 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if debug:
                    print(f"COLLISION: 2 agents crossed paths simultaneously while going for tiles {query_path(path, t)} and {query_path(other_path, t)}, at instant {t}")
                return False
    return True


def is_pathset_collision_free(pathset):
    is_free = True
    for i in range(len(pathset)):
        free = is_collision_free(pathset[i], pathset[:i] + pathset[i + 1:], True)
        if not free:
            is_free = False
    return is_free
