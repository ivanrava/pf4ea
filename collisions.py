import numpy as np

from utils import Path


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
