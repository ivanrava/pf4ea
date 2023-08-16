import numpy as np
import matplotlib.pyplot as plt


def get_random_boolean_weighted(weight):
    return (np.random.random_sample(1) < weight)[0]


def idxes_to_key(idxes):
    i, j = idxes
    return f'({i},{j})'


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

    def neighbors(self, el, also_diagonals):
        (i, j) = el
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if also_diagonals:
            directions += [(1, 1), (-1, -1), (-1, 1), (1, -1)]
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
        plt.title(f'{self.grid.shape[1]} x {self.grid.shape[0]}\nObstacles: {self.num_obstacle_cells} / {self.grid.size}')
        # plt.imshow(1-self.grid, cmap='gray')
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

    def __get_weight(self, from_cell, to_cell):
        distance = abs(from_cell[0] - to_cell[0]) + abs(from_cell[1] - to_cell[1])
        return 1 if distance == 1 else np.sqrt(2)

    def to_adj(self):
        # adj = np.zeros(self.grid.size - self.num_obstacle_cells)
        adj = {}
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                # Do not put obstacles "vertices" (they are not vertices) into the adjacency lists
                if self.grid[i, j] == 1:
                    continue
                empty_neighbors = self.empty_neighbors((i, j), also_diagonals=True)

                empty_neighbors = [(n, self.__get_weight((i, j), n)) for n in empty_neighbors]
                adj[idxes_to_key((i, j))] = empty_neighbors
        return adj

    def idxes_to_idx(self, idxes):
        i, j = idxes
        return i * self.grid.shape[0] + j


class Instance:
    def __init__(self, width, height, num_agents=3, conglomeration_ratio=0.5, obstacle_ratio=0.1, max_length=20):
        self.grid = Grid(width, height, conglomeration_ratio=conglomeration_ratio, obstacle_ratio=obstacle_ratio)
        self.adj = self.grid.to_adj()
        self.num_agents = num_agents
        self.starting_positions = []
        self.paths = []
        while num_agents > 0:
            starting_position = self.grid.get_random_empty_cell()
            while starting_position in self.starting_positions:
                starting_position = self.grid.get_random_empty_cell()
            num_agents -= 1
            self.starting_positions.append(starting_position)
            self.paths.append(self.build_path_from(starting_position, max_length=max_length))

    def plot(self):
        self.grid.plot(show=False)

        for path in self.paths:
            for idx, node in enumerate(path):
                plt.plot(node[1]+0.5, node[0]+0.5, 's', markersize=8, color='#aaa')
                plt.text(node[1]+0.1, node[0]+1, idx+1, fontsize=5)

        for pos in self.starting_positions:
            plt.plot(pos[1] + 0.5, pos[0] + 0.5, 's', markersize=8)
        plt.show()

    def build_path_from(self, starting_position, max_length):
        path = [starting_position]
        while len(path) < max_length:
            neighbors = self.adj[idxes_to_key(path[-1])]
            if len(neighbors) == 0:
                break
            idx = np.random.choice(range(len(neighbors)))
            next_neighbor = neighbors[idx][0]
            while next_neighbor in path:
                neighbors.pop(idx)
                if len(neighbors) == 0:
                    return path
                idx = np.random.choice(range(len(neighbors)))
                next_neighbor = neighbors[idx][0]
            path.append(next_neighbor)
        return path
