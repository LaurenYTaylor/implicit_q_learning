import numpy as np


def dist_fn(self, observation):
    current = np.array([observation[0], observation[1]])
    goal = np.array([observation[-2], observation[-1]])
    return np.linalg.norm(goal - current)
