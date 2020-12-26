from re import A
from game.omok.type import *
from game.omok.board import *

import numpy as np

class ZeroEncoder:
    def __init__(self, board_size):
        self.board_size = board_size
        # 0 - 3. our stones with 1, 2, 3, 4+ liberties
        # 4 - 7. opponent stones with 1, 2, 3, 4+ liberties
        # 8. 1 if we get komi
        # 9. 1 if opponent gets komi
        # 10. action would be illegal due to ko
        self.num_planes = 5

    def name(self):
        return 'zero'

    def encode(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player = game_state.next_player

        for r in range(self.board_size):
            for c in range(self.board_size):
                p = Point(row=r + 1, col=c + 1)
                b = game_state.board.get(p)

                if b is None:
                    if game_state.is_33(next_player, Action.play(p)):
                        board_tensor[2][r][c] = 1
                else:
                    info = game_state.board.get_info(b, p)
                    n = max(info, key=lambda x: x[0])[0]
                    if b == next_player:
                        board_tensor[0][r][c] = 1
                        board_tensor[3][r][c] = n / 5
                    else:
                        board_tensor[1][r][c] = 1
                        board_tensor[4][r][c] = n / 5

        return board_tensor

# tag::encode_action[]
    def encode_action(self, action):
        if action.is_play:
            return (self.board_size * (action.point.row - 1) +   # <1>
                (action.point.col - 1))                          # <1>
        elif action.is_pass:
            return self.board_size * self.board_size           # <2>
        raise ValueError('Cannot encode resign action')          # <3>

    def decode_action_index(self, index):
        if index == self.board_size * self.board_size:
            return Action.pass_turn()
        row = index // self.board_size
        col = index % self.board_size
        return Action.play(Point(row=row + 1, col=col + 1))

    def num_actions(self):
        return self.board_size * self.board_size + 1
# end::encode_action[]

    def shape(self):
        return self.num_planes, self.board_size, self.board_size,

def create(board_size):
    return ZeroEncoder(board_size)