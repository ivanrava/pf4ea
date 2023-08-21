import matplotlib.pyplot as plt

import generator.agents as agents
from generator.grid import Grid
from generator.utils import Path


class Instance:
    def __init__(self, width, height,
                 num_agents=3,
                 conglomeration_ratio=0.5,
                 obstacle_ratio=0.1,
                 max_length=20,
                 agent_path_length=10,
                 agent_generator: agents.AgentGenerator = agents.RandomAgentGenerator(max_length=10)):

        self.grid = Grid(width, height, conglomeration_ratio=conglomeration_ratio, obstacle_ratio=obstacle_ratio)

        self.init = self.grid.get_random_empty_cell()
        self.goal = self.grid.get_random_empty_cell()

        self.max_length = max_length
        if max_length > self.maximum_max_length(agent_path_length):
            print(f"Warning: max_length is too big. Setting max_length to {self.maximum_max_length(agent_path_length)}")
            self.max_length = self.maximum_max_length(agent_path_length)

        self.num_agents = num_agents
        self.paths, self.starting_positions = agent_generator.build_paths(num_agents, self.grid)

    def plot_instant(self, t, additional_path: Path):
        self.grid.plot(show=False)
        plt.title(f"t={t}")

        for pos in self.starting_positions:
            plt.plot(pos[1] + 0.5, pos[0] + 0.5, 'x', markersize=18)

        plt.plot(self.init[1] + 0.5, self.init[0] + 0.5, 'x', markersize=18, color='#ccc')
        plt.plot(self.goal[1] + 0.5, self.goal[0] + 0.5, '+', markersize=18, color='#ccc')

        plt.gca().set_prop_cycle(None)
        for path in self.paths:
            plt.plot(path[t][1] + 0.5, path[t][0] + 0.5, 's', markersize=12)

        plt.plot(additional_path[t][1] + 0.5, additional_path[t][0] + 0.5, '.', markersize=20, color='#aaa')

        plt.show()

    def plot(self, additional_path: Path):
        for t in range(max([len(path) for path in self.paths] + [len(additional_path)])):
            self.plot_instant(t, additional_path)

    def maximum_max_length(self, agent_path_length):
        return agent_path_length + self.grid.num_obstacle_cells

    def is_valid_start_stop(self):
        # init collides
        if self.init in self.starting_positions:
            return False
        # goal collides
        for path in self.paths:
            if self.goal == path[-1]:
                return False

        return True

