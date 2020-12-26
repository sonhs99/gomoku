import numpy as np

__all__ = [
    'combine_experience',
    'ExperienceBuffer',
    'ExperienceCollector'
]

def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_visit_counts = np.concatenate([np.array(c.visit_counts) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])

    return ExperienceBuffer(
        states=combined_states,
        visit_counts=combined_visit_counts,
        rewards=combined_rewards,
    )

class ExperienceBuffer():
    def __init__(self, states, visit_counts, rewards):
        self.states = states
        self.visit_counts = visit_counts
        self.rewards = rewards

    def save(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('visit_counts', data=self.visit_counts)
        h5file['experience'].create_dataset('rewards', data=self.rewards)

    @classmethod
    def load(cls, h5file):
        return ExperienceBuffer(
            states=h5file['experience']['states'],
            visit_counts=h5file['experience']['visit_counts'],
            rewards=h5file['experience']['rewards'],
        )


class ExperienceCollector():
    def __init__(self, ):
        self.states = []
        self.visit_counts = []
        self.rewards = []
        self._current_episode_states = []
        self._current_episode_visit_counts = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_visit_counts = []
        
    def record_decision(self, state, visit_count):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_count)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards += [reward for _ in range(num_states)]

        self._current_episode_states = []
        self._current_episode_visit_counts = []