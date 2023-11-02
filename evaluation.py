from typing import Dict

import flax.linen as nn
import gym
import numpy as np
from copy import deepcopy


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    all_dists = []
    all_lens = []
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
            if "Flappy" in env.unwrapped.spec.id:
                action = agent.sample_deterministic_actions(observation, temperature=0.0)
            else:
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
            horizon: int, algo: str, final_horizon=None,
                  at_threshold=None) -> Dict[str, float]:
    stats = {'return': [], 'length': [], 'agent_type': []}


    for ne in range(num_episodes):
        timesteps_applied = {}
        pos_applied = {}
        time_step = 0
        agent_type = []
        observation, done = env.reset(), False
        while not done:
            if algo == "jsrlgs":
                if "antmaze" in env.unwrapped.spec.id:
                    h = np.linalg.norm(np.array(env.target_goal) - np.array(env.get_xy()))
                else:
                    h = np.abs(observation[1])

                if (h >= horizon and h <= final_horizon) or np.mean(agent_type)>at_threshold:
                    agent = pretrained_agent
                    agent_type.append(0.0)
                else:
                    agent = learning_agent
                    agent_type.append(1.0)
            elif algo == "jsrl":
                h = time_step
                if h <= horizon or np.mean(agent_type)>at_threshold:
                    agent = pretrained_agent
                    agent_type.append(0.0)
                else:

                    agent = learning_agent
                    agent_type.append(1.0)
            if "Flappy" in env.unwrapped.spec.id:
                action = agent.sample_deterministic_actions(observation)
            else:
                action = agent.sample_actions(observation, temperature=0.0)
            timesteps_applied[time_step] = agent_type[-1]
            if "antmaze" in env.unwrapped.spec.id:
                pos_applied[tuple(env.get_xy())] = ["red", "blue"][int(agent_type[-1])]

            observation, _, done, info = env.step(action)
            time_step += 1
        '''
        if ne<5:
            from matplotlib import pyplot as plt
            #plt.scatter(timesteps_applied.keys(), timesteps_applied.values())
            #plt.show()

            fig, ax = plt.subplots(figsize=(5,5))
            #ax.scatter([x[0] for x in pos_applied.keys()], [x[1] for x in pos_applied.keys()], c=pos_applied.values(), s=1)
            #import pdb; pdb.set_trace()
            maze_map = np.array(deepcopy(env.env.env.env.env._maze_map))
            maze_map[maze_map=="g"] = 0.5
            maze_map[maze_map=="r"] = 0.5
            maze_map = maze_map.astype('float')
            min_x, min_y = env.env.env.env.env._rowcol_to_xy((0, 0))
            max_x, max_y = env.env.env.env.env._rowcol_to_xy((4, 4))
            torsox = 0
            torsoy = 0
            scaling = 1

            ax.imshow(maze_map, extent=[min_x, max_x, min_y+0.5, max_y+0.5], cmap="Accent")

            goal = (env.target_goal[0]-(torsox/2)/scaling, env.target_goal[1]-(torsoy/2)/scaling)
            ax.scatter(goal[0], goal[1], s=1, marker="X", c="black")
            ax.text(goal[0], goal[1], "Goal", size=12)
            circ = plt.Circle(goal, horizon, fill=None)
            ax.set_aspect(1)
            ax.add_artist(circ)
            i=0
            for k, v in pos_applied.items():
                ax.scatter(k[0]-(torsox/2)/scaling, k[1]-(torsoy/2)/scaling, c=v, s=1)
                if i%50==1:
                    ax.text(k[0], k[1], str(i), color=v, size=8)
                i+=1

            a = input("save?")
            if len(a)>2:
                plt.savefig(f"plotting/plots/{a}.svg", format='svg', dpi=300, bbox_inches='tight')
            plt.show()
        '''

        for k in stats.keys():
            if k != 'agent_type':
                stats[k].append(info['episode'][k])

        #print(f"eval: {info['episode']}, at: {np.mean(agent_type)}, {horizon}, {final_horizon}")
        stats['agent_type'].append(np.mean(agent_type))


    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats

