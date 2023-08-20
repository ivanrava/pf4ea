import generator
import relaxing
from generator import Instance
import numpy as np
from collections.abc import Callable


# TODO #1: precalculate
def h_diagonal(n, goal):
    dx = abs(n[1] - goal[1])
    dy = abs(n[0] - goal[0])
    return dx + dy + (np.sqrt(2) - 2) * min(dx, dy)


def reconstruct_path(init: (int, int), goal: (int, int), P, t: int) -> generator.Path:
    path = generator.Path([goal])
    while path[-1] != init or t != 0:
        previous = P[(path[-1], t)]
        path.append(previous[0])
        t -= 1
    path.reverse()
    return path


def reach_goal(instance: Instance, h: Callable[[(int, int), (int, int)], float], relaxer=None):
    if relaxer is not None:
        h = lambda from_vertex, to_vertex: relaxer.relaxed_heuristic(from_vertex)
    else:
        # TODO: where to put this?
        # Safety check for starting position collisions. Should be moved elsewhere?
        if instance.init in instance.starting_positions:
            return None

    closed_states = set()
    open_states = {(instance.init, 0)}
    # FIXME: better options for this data structure?
    g = {(instance.init, 0): 0}
    # TODO: P is equal to OPEN U CLOSED (p. 55). Maybe we can "delete" it?
    P = {}

    f = {(instance.init, 0): h(instance.init, instance.goal)}

    while len(open_states) > 0:
        # Find the state in open_states with the lowest f-score
        min_state = relaxing.extract_min(open_states, lambda vertex: f[vertex] if vertex in f else np.inf)

        v, t = min_state
        open_states = open_states.difference({(v, t)})
        closed_states = closed_states.union({(v, t)})
        if v == instance.goal:
            return reconstruct_path(instance.init, instance.goal, P, t)
        if relaxer is not None:
            # Bulk of the alternative strategy
            relaxed_path = relaxer.relaxed_path_from(v)
            if relaxing.is_collision_free(relaxed_path, instance.paths):
                # [1:] or there is a double vertex at the middle (the first reaches v, and the second restarts from v)
                return reconstruct_path(instance.init, v, P, t) + relaxed_path[1:]
        if t < instance.max_length:
            for n in instance.adj[v]:
                n, _ = n
                if (n, t + 1) not in closed_states:
                    traversable = True
                    # Check collisions with other agents
                    for path in instance.paths:
                        # 1. An agent is going to the same cell (n) on next tick (t+1)
                        # 2. An agent is going to my cell (v) on next tick (t+1), and previously was on my next cell (n)
                        if path[t + 1] == n or (path[t + 1] == v and path[t] == n):
                            traversable = False
                        # 3. The agents are "crossing" paths ("diagonal collision")
                        else:
                            delta1 = tuple(abs(np.subtract(v, path[t + 1])))
                            delta2 = tuple(abs(np.subtract(n, path[t])))
                            delta3 = tuple(abs(np.subtract(path[t], path[t + 1])))
                            delta4 = tuple(abs(np.subtract(v, n)))
                            if delta3 == delta4 == (1, 1) and delta1 == delta2 and delta1 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                traversable = False
                    if traversable:
                        if (n, t + 1) not in g or g[min_state] + instance.grid.get_weight(v, n) < g[(n, t + 1)]:
                            P[(n, t + 1)] = min_state
                            g[(n, t + 1)] = g[min_state] + instance.grid.get_weight(v, n)
                            f[(n, t + 1)] = g[(n, t + 1)] + h(n, instance.goal)
                        if (n, t + 1) not in open_states:
                            open_states = open_states.union({(n, t + 1)})

    return None
