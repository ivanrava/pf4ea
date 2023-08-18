import relaxing
import solver
from generator import Grid, Instance
import numpy as np
from pprint import pprint

np.random.seed(40)
instance = Instance(10, 8, conglomeration_ratio=0.4, obstacle_ratio=0.3, num_agents=5)
path = solver.reach_goal(instance)
# print(relaxing.relaxed_path(instance))

pathset = [path] + instance.paths
relaxing.is_pathset_collision_free(pathset)