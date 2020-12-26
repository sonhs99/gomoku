from .. import board
from . import type

import copy

__all__ = [
    'Action',
    'Board',
    'State'
]

class Action(board.Action):
    def __init__(self, point=None, is_pass=False, resign=False):
        assert (point is not None) ^ is_pass ^ resign
        self.point = point
        self.is_play = (point is not None)
        self.is_pass = is_pass
        self.is_resign = resign

    @classmethod
    def play(cls, point):
        assert isinstance(point, type.Point)
        return Action(point=point)

    @classmethod
    def pass_turn(cls):
        return Action(is_pass=True)

    @classmethod
    def resign(cls):
        return Action(resign=True)

neighbors = [
        [type.Point( 1,  1), type.Point(-1, -1)],
        [type.Point( 1,  0), type.Point(-1,  0)],
        [type.Point( 1, -1), type.Point(-1,  1)],
        [type.Point( 0,  1), type.Point( 0, -1)],
]

class Board(board.Board):
    
    def __init__(self, num_cols, num_rows):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self._grid = {}

    def is_on_grid(self, point):
        return 0 <= point.col <= self.num_cols and \
            0 <= point.row <= self.num_rows

    def place(self, player, action):
        super().place(player, action)
        assert action.point is not None
        point = action.point
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None
        
        self._grid[point] = player

    def get(self, point):
        return self._grid.get(point)

    def get_info(self, player, point):
        global neighbors
        info = [
            [0, False],
            [0, False],
            [0, False],
            [0, False],
        ]
        for idx, neighbor in enumerate(neighbors):
            for n in neighbor:
                neigh = point + n
                while(self._grid.get(neigh) == player):
                    neigh = neigh + n
                    info[idx][0] += 1
                info[idx][1]  = self._grid.get(neigh) == None and \
                                self._grid.get(neigh) == player
        return info

class State(board.State):
    def __init__(self, board, next_player, prev, action, over=None):
        self.board = board
        self.next_player = next_player
        self.prev = prev
        self.last = action
        self.over = over

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)
        board = Board(*board_size)
        return State(board, type.Player.black, None, None)

    def apply(self, action):
        super().apply(action)
        flag = None
        if action.is_play:
            next_board = copy.deepcopy(self.board)
            next_board.place(self.next_player, action)
            if self.test(self.next_player, action): flag = self.next_player
        else:
            next_board = self.board
        return State(next_board, self.next_player.next, self.last, action, flag)

    def is_33(self, player, action):
        if not action.is_play: return False
        count = 0
        for i in self.board.get_info(player, action.point):
            if i[0] + int(i[1]) == 2: count += 1
        return count >= 2

    def test(self, player, action):
        if not action.is_play: return False
        for i in self.board.get_info(player, action.point):
            if i[0] == 4: return True
        return False

    def is_valid_action(self, action):
        super().apply(action)
        if self.is_over():
            return False
        if action.is_pass or action.is_resign:
            return True
        return (
            self.board.get(action.point) is None and
            not self.is_33(self.next_player, action))
        
    def is_over(self):
        if self.over is not None: return True
        if self.last is None or self.prev is None: return False
        if self.last.is_resign: return True
        return self.last.is_pass and self.prev.is_pass

    def result(self):
        if not self.is_over():
            return None
        if self.last.is_resign:
            return self.next_player
        return self.over