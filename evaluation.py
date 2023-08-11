from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    max_dist = -np.inf
    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            try:
                dist = np.linalg.norm(np.array(env.target_goal) - np.array(env.get_xy()))
                if dist > max_dist:
                    max_dist = dist
            except AttributeError:
                max_dist = 0
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)
    stats['goal_dist'] = max_dist
    return stats

def evaluate_jsrl(learning_agent: nn.Module, env: gym.Env,
             num_episodes: int, pretrained_agent: nn.Module,
            horizon: int, algo: str) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        time_step = 0
        observation, done = env.reset(), False

        while not done:
            agent = learning_agent
            if algo == "jsrlgs":
                h = np.linalg.norm(np.array(env.target_goal) - np.array(env.get_xy()))
            elif algo == "jsrl":
                h = time_step
            if h <= horizon:
                agent = pretrained_agent

            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
            time_step += 1

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats

