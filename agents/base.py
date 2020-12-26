from game import board

from abc import ABCMeta, abstractmethod
import h5py

class Agent(metaclass=ABCMeta):
    def __init__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def select_action(self, state):
        assert isinstance(state, board.State)
    
    def save(self, h5file):
        raise NotImplementedError()

    @classmethod
    def load(h5file):
        raise NotImplementedError()