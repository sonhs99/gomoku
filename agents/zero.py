from . import base
from .experience import *
from .encoders import zeroencoder
from game import kerasutil

import numpy as np
from tensorflow.keras.optimizers import SGD

class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0

class ZeroTreeNode:
    def __init__(self, state, value, priors, parent, last_move):
        self.state = state
        self.value = value
        self.parent = parent
        self.last_move = last_move
        self.total_visit_count = 1
        self.branches = {}

        for action, p in priors.items():
            if state.is_valid_action(action):
                self.branches[action] = Branch(p)
        self.children = {}
    
    def actions(self):
        return self.branches.keys()

    def add_child(self, action, child_node):
        self.children[action] = child_node

    def get_child(self, action):
        return self.children.get(action)

    def has_child(self, action):
        return action in self.children

    def expected_value(self, action):
        branch = self.branches[action]
        if branch.visit_count == 0:
            return 0.0
        return branch.total_value / branch.visit_count

    def prior(self, action):
        return self.branches[action].prior

    def record_visit(self, action, value):
        self.total_visit_count += 1
        self.branches[action].visit_count += 1
        self.branches[action].total_value += value

    def visit_count(self, move):
        if move in self.branches:
            return self.branches[move].visit_count
        return 0
        
class ZeroAgent(base.Agent):
    def __init__(self, model, encoder, num_rounds=1600, c=2.0):
        self._model = model
        self._encoder = encoder
        self._collector = None
        self.num_rounds = num_rounds
        self.c = c
        self.temperature = 0.0

    def set_collector(self, collector):
        self._collector = collector
    
    def set_temperature(self, temperature):
        self.temperature = temperature

    def select_branch(self, node):
        total_n = node.total_visit_count
        if np.random.rand() < self.temperature:
            actions = list(node.actions())
            return actions[np.random.randint(0, len(actions))]

        def score_branch(action):
            q = node.expected_value(action)
            p = node.prior(action) #+ np.random.rand() if e else 0
            n = node.visit_count(action)
            return (q + self.c * p * np.sqrt(total_n) / (n + 1))
        return max(node.actions(), key=score_branch)

    def create_node(self, game_state, action=None, parent=None):
        state_tensor = self._encoder.encode(game_state)
        model_input = np.array([state_tensor])
        priors, values = self._model.predict(model_input)
        priors = priors[0]
        value = values[0][0]
        
        action_priors = {
            self._encoder.decode_action_index(idx): p
            for idx, p in enumerate(priors)
        }
        new_node = ZeroTreeNode(
            game_state, value,
            action_priors,
            parent, action
        )
        if parent is not None: parent.add_child(action, new_node)
        return new_node

    def select_action(self, game_state):
        root = self.create_node(game_state)

        for i in range(self.num_rounds):
            node = root
            next_action = self.select_branch(node)
            while node.has_child(next_action):
                node = node.get_child()
                next_action = self.select_branch(node)
            new_state = node.state.apply(next_action)
            child_node = self.create_node(new_state, parent=node)

            action = next_action
            value = -1 * child_node.value
            while node is not None:
                node.record_visit(action, value)
                action = node.last_move
                node = node.parent
                value = -1 * value

        if self._collector is not None:
            root_state_tensor = self._encoder.encode(game_state)
            visit_counts = np.array([
                root.visit_count(self._encoder.decode_action_index(idx))
                for idx in range(self._encoder.num_actions())
            ])
            self._collector.record_decision(
                root_state_tensor, visit_counts
            )
        return max(root.actions(), key=root.visit_count)

    def train(self, experience, learning_rate, batch_size):
        num_examples = experience.states.shape[0]

        model_input = experience.states

        visit_sum = np.sum(
            experience.visit_counts, axis=1
        ).reshape((num_examples, 1))
        action_target = experience.visit_counts / visit_sum

        value_target = experience.rewards

        self._model.compile(
            SGD(lr=learning_rate),
            loss=['categorical_crossentropy', 'mse']
        )
        self._model.fit(
            model_input, [action_target, value_target],
            batch_size=batch_size
        )

    def save(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_size'] = self._encoder.board_size
        h5file.create_group('agent')
        h5file['agent'].attrs['num_rounds'] = self.num_rounds
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    @classmethod
    def load(cls, h5file):
        model = kerasutil.load_model_from_hdf5_group(h5file['model'])
        encoder_name = h5file['encoder'].attrs['name']
        board_size = h5file['encoder'].attrs['board_size']
        num_rounds = h5file['agent'].attrs['num_rounds']
        encoder = zeroencoder.ZeroEncoder(board_size)
        return ZeroAgent(model, encoder, num_rounds)