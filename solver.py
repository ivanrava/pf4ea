import generator
import relaxing
from generator import Instance
import numpy as np


# TODO: precalculate
def h(n, goal):
    dx = abs(n[1] - goal[1])
    dy = abs(n[0] - goal[0])
    return dx + dy + (np.sqrt(2) - 2) * min(dx, dy)


def reconstruct_path(init: (int, int), goal: (int, int), P, t: int) -> generator.Path:
    path = generator.Path([goal])
    while path[-1] != init:
        previous = P[(path[-1], t)]
        path.append(previous[0])
        t -= 1
    path.reverse()
    return path


# TODO: add "alternative strategy"
def reach_goal(instance: Instance):
    closed_states = set()
    open_states = {(instance.init, 0)}
    # FIXME: better options?
    g = {(instance.init, 0): 0}
    # TODO: P is equal to OPEN U CLOSED (p. 55). Maybe we can "delete" it?
    P = {}

    # TODO: precalculate
    def f(state: ((int, int), int)):
        v, t = state
        return g[state] + h(v, instance.goal) if state in g else np.inf

    while len(open_states) > 0:
        # Find the state in open_states with the lowest f-score
        min_state = relaxing.extract_min(open_states, f)

        v, t = min_state
        open_states = open_states.difference({(v, t)})
        closed_states = closed_states.union({(v, t)})
        if v == instance.goal:
            return reconstruct_path(instance.init, instance.goal, P, t)
        if t < instance.max_length:
            for n in instance.adj[v]:
                n, _ = n
                if (n, t + 1) not in closed_states:
                    traversable = True
                    for path in instance.paths:
                        if path[t + 1] == n or (path[t + 1] == v and path[t] == n):
                            traversable = False
                    if traversable:
                        if (n, t + 1) not in g or g[min_state] + instance.grid.get_weight(v, n) < g[(n, t + 1)]:
                            P[(n, t + 1)] = min_state
                            g[(n, t + 1)] = g[min_state] + instance.grid.get_weight(v, n)
                            # ...precalculate f
                        if (n, t + 1) not in open_states:
                            open_states = open_states.union({(n, t + 1)})

    return None
