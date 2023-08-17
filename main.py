from generator import Grid, Instance
import numpy as np
from pprint import pprint

np.random.seed(11)
instance = Instance(10, 8, conglomeration_ratio=0.4, obstacle_ratio=0.3)
instance.plot()
print(instance.adj)
# pprint(instance.grid.to_adj())
