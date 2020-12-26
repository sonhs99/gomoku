from .. import type

import enum
from collections import namedtuple

__all__ = [
    'Player',
    'Point'
]

class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def next(self):
        return Player.black if self == Player.white else Player.white

class Point(namedtuple('Point', 'col row')):
    def neighbor(self):
        return [
            Point(self.col + 1, self.row),
            Point(self.col - 1, self.row),
            Point(self.col, self.row + 1),
            Point(self.col, self.row - 1),
        ]

    def __add__(self, other):
        return Point(self.col + other.col, self.row + other.row)
