import collisions
import heuristics
import solver
from generator import Instance
import numpy as np
from pprint import pprint

# FIXME: what to do here?
# Seed 2,11,16,18,24,31,37,42,47,48,51,67,71,73,74,80,87,90,91,92: grey stops and another overlaps it.
# Seed 29,30: grey stop and purple stop collide.
# Seed 48: start and goal are the same (allowed, not allowed?)
if __name__ == '__main__':
    np.random.seed(19)
    instance = Instance(10, 8, conglomeration_ratio=0.4, obstacle_ratio=0.3, num_agents=5)

    # path, _, _ = solver.reach_goal(instance, heuristics.Diagonal(instance))
    path, _, _ = solver.reach_goal(instance, heuristics.DijkstraRelaxer(instance))
    if path is not None:
        print(":) REACHED!")
        instance.plot(path)
        pathset = [path] + instance.paths
        collisions.is_pathset_collision_free(pathset)
    else:
        print(":( Unreachable")
