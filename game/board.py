from . import type
from abc import ABCMeta, abstractmethod

__all__ = [
    'Board',
    'State',
    'Action'
]

class Board(metaclass=ABCMeta):
    def __init__(self):
        pass
    
    @abstractmethod
    def place(self, player, action):
        assert isinstance(action, Action)

    @abstractmethod
    def get(self, point):
        pass


class Action():
    def __init__(self):
        raise NotImplementedError()

class State():
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def new_game(cls):
        pass

    @abstractmethod
    def apply(self, action):
        assert isinstance(action, Action)

    @abstractmethod
    def is_valid_action(self, action):
        assert isinstance(action, Action)
        
    @abstractmethod
    def is_over(self):
        pass

    @abstractmethod
    def result(self):
        pass