from typing import Optional, Union

import gym
from wrappers.common import TimeStep
import numpy as np
class DecreaseActionDim(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._action_space = ExpandActionDim(self.action_space)

class ExpandActionDim(gym.spaces.Discrete):
    def __init__(self, action_space):
        super().__init__(action_space.n)

    def sample(self) -> np.ndarray:
        return np.array([super().sample()])
