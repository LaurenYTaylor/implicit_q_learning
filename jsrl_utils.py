import numpy as np

def update_horizon(returns, horizon_idx, prev_best, tolerance=0.05, n=5):
    if len(returns) < n:
        return horizon_idx, prev_best
    rolling_mean = np.mean(returns[-5:])
    if np.isinf(prev_best):
        prev = prev_best
    elif prev_best == 0:
        # prev = prev_best - 0.1
        prev = prev_best
    else:
        prev = prev_best - tolerance * prev_best
    if rolling_mean > prev:
        prev_best = rolling_mean
        return horizon_idx + 1, prev_best
    else:
        return horizon_idx, prev_best


def dist_fn(self, observation):
    current = np.array([observation[0], observation[1]])
    goal = np.array([observation[-2], observation[-1]])
    return np.linalg.norm(goal - current)
