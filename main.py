import numpy as np
from timeit import default_timer as timer
import sys

import solver
from generator import Instance
import generator.agents as agents
import heuristics
import timeout

# FIXME: should we do a final check after resolution?
# Seed 2,11,16,18,24,31,37,42,47,48,51,67,71,73,74,80,87,90,91,92 (4): grey stops and another overlaps it.
# FIXME: should we allow the possibility of start = end
# Seed 48: start and goal are the same (allowed, not allowed?)
if __name__ == '__main__':
    def get_agent_generator(name: str, path_length: int):
        agent_generators = {
            'random': agents.RandomAgentGenerator(max_length=path_length),
            'optimal': agents.OptimalAgentGenerator(max_length=path_length)
        }
        return agent_generators[name]


    def get_heuristic(name: str, instance: Instance):
        heuristic_dict = {
            'dijkstra': heuristics.DijkstraRelaxer(instance),
            'diagonal': heuristics.Diagonal(instance.grid, instance.goal)
        }
        return heuristic_dict[name]


    if len(sys.argv) > 1:
        width = int(sys.argv[1])
        height = int(sys.argv[2])
        num_agents = int(sys.argv[3])
        obstacle_ratio = float(sys.argv[4])
        conglomeration_ratio = float(sys.argv[5])
        agent_path_length = int(sys.argv[6])
        max_length = int(sys.argv[7])
        agent_generator = sys.argv[8]
        h = sys.argv[9]
        seed = int(sys.argv[10])

        np.random.seed(seed)

        agent_generator_obj = get_agent_generator(agent_generator, agent_path_length)

        try:
            with timeout.time_limit(1):
                instance = Instance(width, height,
                                    num_agents=num_agents,
                                    obstacle_ratio=obstacle_ratio,
                                    conglomeration_ratio=conglomeration_ratio,
                                    max_length=max_length,
                                    agent_generator=agent_generator_obj
                                    )
                heuristic_obj = get_heuristic(h, instance)
                start = timer()
                path, closed_states, inserted_states = solver.solve_instance(instance, heuristic=heuristic_obj)
                end = timer()
                instance_resolution = end-start

                print('success' if path is not None else 'failure',
                      len(path) if path is not None else 0, instance.grid.get_path_cost(path) if path is not None else 0,
                      closed_states, inserted_states, path.waits() if path is not None else 0,
                      instance.grid.elapsed_time, instance.elapsed_time, instance_resolution,
                      sep=';', end='')
        except timeout.TimeoutException:
            print('timeout',
                  0, 0,
                  0, 0, 0,
                  0, 0, 0,
                  sep=';', end='')
    else:
        np.random.seed(4)

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
        print(f"Grid generation: {instance.grid.elapsed_time} s")
        print(f"Paths generation: {instance.elapsed_time} s")

        start = timer()
        # path, closed_states, inserted_states = solver.solve_instance(instance, heuristic=heuristics.Diagonal(instance.grid, instance.goal))
        path, closed_states, inserted_states = solver.solve_instance(instance,
                                                                     heuristic=heuristics.DijkstraRelaxer(instance))
        end = timer()
        print(f"Instance resolution: {end - start} s")

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
