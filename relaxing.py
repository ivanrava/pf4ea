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