import numpy as np
from timeit import default_timer as timer

import solver
from generator import Instance
import generator.agents as agents

# FIXME: what to do here?
# Seed 2,11,16,18,24,31,37,42,47,48,51,67,71,73,74,80,87,90,91,92 (4): grey stops and another overlaps it.
# Seed 29,30: grey stop and purple stop collide.
# Seed 48: start and goal are the same (allowed, not allowed?)
if __name__ == '__main__':
    np.random.seed(4)

    start = timer()
    instance = Instance(10, 8,
                        conglomeration_ratio=0.4,
                        obstacle_ratio=0.3,
                        num_agents=5,
                        agent_generator=agents.RandomAgentGenerator(max_length=10, avoid_backtracking=False))
    # instance = Instance(10, 8,
    #                     conglomeration_ratio=0.4,
    #                     obstacle_ratio=0.3,
    #                     num_agents=5,
    #                     agent_generator=agents.OptimalAgentGenerator(max_length=10))
    end = timer()
    print(f"Instance generation: {end-start} s")

    import heuristics
    start = timer()
    path, closed_states, inserted_states = solver.reach_goal(instance.grid, instance.paths, instance.init, instance.goal, instance.max_length, instance.starting_positions, heuristic=heuristics.Diagonal(instance.grid, instance.goal))
    end = timer()
    print(f"Instance resolution: {end-start} s")

    if path is not None:
        print(len(path), instance.grid.get_path_cost(path), closed_states, inserted_states, path.waits())
        print(":) REACHED!")
        instance.plot(path)
        pathset = [path] + instance.paths
        import collisions
        collisions.is_pathset_collision_free(pathset)
    else:
        instance.grid.plot(True)
        print(":( Unreachable")
