from abc import ABC, abstractmethod
import numpy as np

from .utils import Path
from .grid import Grid
import solver


class AgentGenerator(ABC):
    def __init__(self, max_length: int):
        self.max_length = max_length
        self.starting_positions = []
        self.paths = []

    @abstractmethod
    def build_path(self, grid: Grid) -> Path:
        raise NotImplementedError("Abstract method 'build_path' not implemented")

    def get_new_random_start(self, grid: Grid) -> (int, int):
        cell = grid.get_random_empty_cell()
        while cell in self.starting_positions:
            cell = grid.get_random_empty_cell()
        return cell

    def build_paths(self, num_agents: int, grid: Grid):
        for _ in range(num_agents):
            # Tries to build a collision-free path
            new_path = self.build_path(grid)
            import collisions
            while not collisions.is_collision_free(new_path, self.paths):
                new_path = self.build_path(grid)
            # Appends the path
            self.starting_positions.append(new_path[0])
            self.paths.append(new_path)
        return self.paths, self.starting_positions


class RandomAgentGenerator(AgentGenerator):
    def __init__(self, max_length: int, avoid_backtracking=False):
        super().__init__(max_length)
        self.avoid_backtracking = avoid_backtracking

    def build_path(self, grid: Grid) -> Path:
        path = Path([self.get_new_random_start(grid)])
        while len(path) < self.max_length:
            neighbors = grid.adj[path[-1]][:]
            if len(neighbors) == 0:
                break
            idx = np.random.choice(range(len(neighbors)))
            next_neighbor = neighbors[idx][0]
            # Optional: forbids agent from walking again down the same cells
            if self.avoid_backtracking:
                while next_neighbor in path:
                    neighbors.pop(idx)
                    if len(neighbors) == 0:
                        return path
                    idx = np.random.choice(range(len(neighbors)))
                    next_neighbor = neighbors[idx][0]
            path.append(next_neighbor)
        return path


class OptimalAgentGenerator(AgentGenerator):
    def __init__(self, max_length: int):
        super().__init__(max_length)

    def ask_reach_goal_for_path(self, grid: Grid) -> Path:
        path, _, _ = solver.reach_goal(
            grid,
            # This list is incremented each time, starting from an empty state
            self.paths,
            self.get_new_random_start(grid),
            self.get_new_random_start(grid),
            max_length=self.max_length,
        )
        return path

    def build_path(self, grid: Grid) -> Path:
        path = self.ask_reach_goal_for_path(grid)
        while path is None:
            path = self.ask_reach_goal_for_path(grid)
        return path
