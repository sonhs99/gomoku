import numpy as np

from .base import Agent
from game.omok.board import Action
from game.omok.type import Point

__all__ = ['RandomBot']


class RandomBot(Agent):
    def __init__(self):
        self.dim = None
        self.point_cache = []

    def _update_cache(self, dim):
        self.dim = dim
        rows, cols = dim
        self.point_cache = []
        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                self.point_cache.append(Point(row=r, col=c))

    def select_action(self, game_state):
        """Choose a random valid move that preserves our own eyes."""
        super().select_action(game_state)
        dim = (game_state.board.num_rows, game_state.board.num_cols)
        if dim != self.dim:
            self._update_cache(dim)

        idx = np.arange(len(self.point_cache))
        np.random.shuffle(idx)
        for i in idx:
            p = self.point_cache[i]
            if game_state.is_valid_action(Action.play(p)):
                return Action.play(p)
        return Action.pass_turn()

    def save(self, h5file):
        pass

    @classmethod
    def load(h5file):
        pass