import gym
from gym import spaces
import numpy as np
from enum import IntEnum


class Cell(IntEnum):
    EMPTY, WALL, LAVA = range(3)

class Grid(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, board, start_pos, start_orientation, goal_pos):
        super().__init__()
        self.board = board
        self.start_pos = start_pos
        self.start_orientation = start_orientation
        self.goal_pos = goal_pos
        self.action_space = spaces.Discrete(3)
        self.n_x, self.n_y = self.board.shape
        self.observation_space = spaces.Discrete(self.n_x * self.n_y * 4)
        rot = lambda x: (np.array([[0, -1], [1, 0]]) @ x[:, None]).reshape(-1)
        drc = np.array([0, 1])
        self.orientation_lookup = {0: drc, 1: rot(drc), 2: rot(rot(drc)), 3: rot(rot(rot(drc)))}

    def step(self, action):
        agent_pos, orientation = self._decode_state(self.state)
        assert not self.done
        assert action in range(3)
        self.state, reward, done = self.dynamics(self.state, action)
        if done:
            self.done = True
        return self.state, reward, done, {}

    def get_agent_pos(self):
        agent_pos, orientation = self._decode_state(self.state)
        return agent_pos

    def _encode_state(self, agent_pos, orientation):
        x, y = agent_pos
        return int(orientation * self.n_x * self.n_y + x * self.n_y + y)

    def _decode_state(self, state):
        orientation = state // (self.n_x * self.n_y)
        other = state % (self.n_x * self.n_y) 
        agent_pos = other // self.n_y, other % self.n_y
        return agent_pos, orientation

    def reset(self):
        agent_pos = self.start_pos
        orientation = self.start_orientation
        self.state = self._encode_state(agent_pos, orientation)
        self.done = False
        return self.state

    def dynamics(self, state, action):
        agent_pos, orientation = self._decode_state(state)
        delta = self.orientation_lookup[orientation]
        if (agent_pos == self.goal_pos).all():
            return state, 0, True
        if action == 0:
            try:
                if (agent_pos + delta < 0).any():
                    raise IndexError()
                if self.board[tuple(agent_pos + delta)] == int(Cell.EMPTY):
                    agent_pos = agent_pos + delta
                elif self.board[tuple(agent_pos + delta)] == int(Cell.LAVA):
                    agent_pos = agent_pos + delta
                    state = self._encode_state(agent_pos, orientation)
                    reward = 0
                    done = True
                    return state, reward, done
            except IndexError:
                pass
        elif action == 1:
            orientation += 1
            orientation %= 4
        elif action == 2:
            orientation -= 1
            orientation %= 4
        reward = 1 if (agent_pos == self.goal_pos).all() else 0
        state = self._encode_state(agent_pos, orientation)
        done = reward == 1
        return state, reward, done

    def render(self, mode='human', close=False):
        agent_pos, orientation = self._decode_state(self.state)
        for i in range(self.n_x):
            for j in range(self.n_y):
                if list(agent_pos) == [i, j]:
                    print('o', end='')
                elif list(self.goal_pos) == [i, j]:
                    print('x', end='')
                elif self.board[i, j] == int(Cell.EMPTY):
                    print('.', end='')
                elif self.board[i, j] == int(Cell.LAVA):
                    print('~', end='')
                elif self.board[i, j] == int(Cell.WALL):
                    print('-', end='')
            print()
        print()

class EmptyGrid(Grid):
    board = np.zeros([8, 8])
    pot_goals = []
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 0 and i > 0 or j > 0:
                pot_goals.append(np.array([i, j]))
    variants = len(pot_goals)

    def __init__(self, variant):
        assert variant in range(self.variants)
        start_pos = np.zeros(2)
        start_orientation = 0
        goal_pos = self.pot_goals[variant]
        super().__init__(self.board, start_pos, start_orientation, goal_pos)


class MazeGrid(Grid):
    board = np.array([
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 2, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [2, 0, 1, 0, 1, 0, 0, 0],
        [2, 0, 1, 0, 1, 0, 1, 1],
        [0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 2, 0, 0, 0],
    ])
    pot_goals = []
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 0 and i > 0 or j > 0:
                pot_goals.append(np.array([i, j]))
    variants = len(pot_goals)

    def __init__(self, variant):
        assert variant in range(self.variants)
        start_pos = np.zeros(2)
        start_orientation = 0
        goal_pos = self.pot_goals[variant]
        super().__init__(self.board, start_pos, start_orientation, goal_pos)


class RoomsGrid(Grid):
    board = np.array([
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
        [0, 2, 2, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 2, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 0, 0],
    ])
    pot_goals = []
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 0 and i > 0 or j > 0:
                pot_goals.append(np.array([i, j]))
    variants = len(pot_goals)

    def __init__(self, variant):
        assert variant in range(self.variants)
        start_pos = np.zeros(2)
        start_orientation = 0
        goal_pos = self.pot_goals[variant]
        super().__init__(self.board, start_pos, start_orientation, goal_pos)
