from generator import Grid
from pprint import pprint

grid = Grid(30, 40, conglomeration_ratio=0.1)
grid.plot()
pprint(grid.to_adj())
