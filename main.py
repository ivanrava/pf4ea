from generator import Grid, Instance
from pprint import pprint

instance = Instance(30, 40, conglomeration_ratio=0.4, obstacle_ratio=0.3)
instance.plot()
# pprint(instance.grid.to_adj())
