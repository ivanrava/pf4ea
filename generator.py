import numpy as np
import matplotlib.pyplot as plt

import relaxing


def get_random_boolean_weighted(weight):
    return (np.random.random_sample(1) < weight)[0]


class Path(list):
    def __getitem__(self, t):
        try:
            return super().__getitem__(t)
        except IndexError:
            return super().__getitem__(-1)

    def __add__(self, other):
        return Path([x for x in self] + [x for x in other])


class Grid:
    def __init__(self, height, width, obstacle_ratio=0.1, conglomeration_ratio=0.5):
        # Generates an empty grid
        self.grid = np.zeros((height, width))
        self.obstacle_ratio = obstacle_ratio
        self.conglomeration_ratio = conglomeration_ratio

        # Calculates how many obstacle cells we need
        self.num_obstacle_cells = int(self.grid.size * obstacle_ratio)

        obstacles = self.num_obstacle_cells
        # Adds obstacles
        while obstacles > 0:
            rand_i, rand_j = self.get_random_empty_cell()
            self.grid[rand_i][rand_j] = 1
            obstacles -= 1
            obstacles = self.__add_neighbor((rand_i, rand_j), obstacles, conglomeration_ratio)

    # FIXME: should we allow movement across "diagonal obstacles"?
    def neighbors(self, el, also_diagonals):
        (i, j) = el
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if also_diagonals:
            directions += [(1, 1), (-1, -1), (-1, 1), (1, -1), (0, 0)]
        neighbors = []
        for curr_dir in directions:
            i_d, j_d = curr_dir
            if i + i_d < 0 or i + i_d >= self.grid.shape[0] or j + j_d < 0 or j + j_d >= self.grid.shape[1]:
                continue
            neighbors.append((i + i_d, j + j_d))
        return neighbors

    def empty_neighbors(self, el, also_diagonals=False):
        return [n for n in self.neighbors(el, also_diagonals) if self.grid[n] != 1]

    def get_random_empty_cell(self):
        rand_i, rand_j = np.random.randint(self.grid.shape[0]), np.random.randint(self.grid.shape[1])
        while self.grid[rand_i][rand_j] == 1:
            rand_i, rand_j = np.random.randint(self.grid.shape[0]), np.random.randint(self.grid.shape[1])
        return rand_i, rand_j

    # FIXME: conglomeration=1 should be 1 big block, careful to the borders
    def __add_neighbor(self, starting_from, num_obstacle_cells, conglomeration_ratio):
        if num_obstacle_cells == 0:
            return 0
        add_neighbor = get_random_boolean_weighted(conglomeration_ratio)
        if add_neighbor:
            neighbors = self.empty_neighbors(starting_from, also_diagonals=False)
            if len(neighbors) == 0:
                return num_obstacle_cells
            random_neighbor = neighbors[np.random.choice(range(len(neighbors)))]
            self.grid[random_neighbor] = 1

            return self.__add_neighbor(random_neighbor, num_obstacle_cells - 1, conglomeration_ratio)
        else:
            return num_obstacle_cells

    def print(self):
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                if self.grid[i][j] == 0:
                    print('.', end='')
                else:
                    print('█', end='')
            print()

    def plot(self, show=True):
        plt.title(
            f'{self.grid.shape[1]} x {self.grid.shape[0]}\nObstacles: {self.num_obstacle_cells} / {self.grid.size}')
        plt.pcolormesh(1 - self.grid, edgecolors='#777', linewidth=0.5, cmap='gray')
        plt.xticks(range(0, self.grid.shape[1], 5))
        plt.yticks(range(0, self.grid.shape[0], 2))
        ax = plt.gca()
        ax.invert_yaxis()
        ax.set_aspect('equal')
        for tick in ax.xaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_verticalalignment("top")
        if show:
            plt.show()

    def get_weight(self, from_cell, to_cell):
        distance = abs(from_cell[0] - to_cell[0]) + abs(from_cell[1] - to_cell[1])
        return 1 if distance <= 1 else np.sqrt(2)

    def to_adj(self):
        adj = {}
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                # Do not put obstacles "vertices" (they are not vertices) into the adjacency lists
                if self.grid[i, j] == 1:
                    continue
                empty_neighbors = self.empty_neighbors((i, j), also_diagonals=True)

                empty_neighbors = [(n, self.get_weight((i, j), n)) for n in empty_neighbors]
                adj[(i, j)] = empty_neighbors
        return adj

    def idxes_to_idx(self, idxes):
        i, j = idxes
        return i * self.grid.shape[0] + j


class Instance:
    def __init__(self, width, height,
                 num_agents=3,
                 conglomeration_ratio=0.5,
                 obstacle_ratio=0.1,
                 max_length=20,
                 agent_path_length=10,
                 avoid_backtracking=False):
        self.grid = Grid(width, height, conglomeration_ratio=conglomeration_ratio, obstacle_ratio=obstacle_ratio)
        self.adj = self.grid.to_adj()
        self.max_length = max_length
        self.num_agents = num_agents
        self.starting_positions = []
        self.paths = []

        # FIXME: add check for max_length parameter (it has an upper bound)

        for _ in range(num_agents):
            # Tries to build a collision-free path
            new_path = self.build_path_from(self.grid.get_random_empty_cell(), max_length=agent_path_length, avoid_backtracking=avoid_backtracking)
            while not relaxing.is_collision_free(new_path, self.paths):
                new_path = self.build_path_from(self.grid.get_random_empty_cell(), max_length=agent_path_length, avoid_backtracking=avoid_backtracking)
            # Appends the path
            self.starting_positions.append(new_path[0])
            self.paths.append(new_path)
        self.init = self.grid.get_random_empty_cell()
        self.goal = self.grid.get_random_empty_cell()

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

    def build_path_from(self, starting_position, max_length: int, avoid_backtracking: bool = False) -> Path:
        path = Path([starting_position])
        while len(path) < max_length:
            neighbors = self.adj[path[-1]][:]
            if len(neighbors) == 0:
                break
            idx = np.random.choice(range(len(neighbors)))
            next_neighbor = neighbors[idx][0]
            # Optional: forbids agent from walking again down the same cells
            if avoid_backtracking:
                while next_neighbor in path:
                    neighbors.pop(idx)
                    if len(neighbors) == 0:
                        return path
                    idx = np.random.choice(range(len(neighbors)))
                    next_neighbor = neighbors[idx][0]
            path.append(next_neighbor)
        return path
