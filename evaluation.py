from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    all_dists = []
    all_lens = []
    import time
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        i = 0
        while not done:
            try:
                if "antmaze" in env.unwrapped.spec.id:
                    dist = np.linalg.norm(np.array(env.target_goal) - np.array(env.get_xy()))
                else:
                    dist = np.sqrt(observation[0]**2+observation[1]**2)
                all_dists.append(dist)
            except AttributeError:
                continue
            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
            all_lens.append(i)
            i += 1

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    stats['goal_dist'] = all_dists
    stats['all_lens'] = all_lens
    return stats

def evaluate_jsrl(learning_agent: nn.Module, env: gym.Env,
             num_episodes: int, pretrained_agent: nn.Module,
            horizon: int, algo: str) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    agent_type = []
    for _ in range(num_episodes):
        time_step = 0
        observation, done = env.reset(), False
        while not done:
            agent = learning_agent

            if algo == "jsrlgs":
                if "antmaze" in env.unwrapped.spec.id:
                    h = np.linalg.norm(np.array(env.target_goal) - np.array(env.get_xy()))
                else:
                    h = np.abs(observation[1])
                if h >= horizon:
                    agent = pretrained_agent
                    agent_type.append(0.0)
                else:
                    agent_type.append(1.0)
            elif algo == "jsrl":
                h = time_step
                if h <= horizon:
                    agent = pretrained_agent
                    agent_type.append(0.0)
                else:
                    agent_type.append(1.0)


            action = agent.sample_actions(observation, temperature=0.0)
            observation, _, done, info = env.step(action)
            time_step += 1

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    for k, v in stats.items():
        stats[k] = np.mean(v)

    stats['agent_type'] = np.mean(agent_type)

    return stats

