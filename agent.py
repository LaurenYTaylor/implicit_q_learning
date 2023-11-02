import numpy as np
import glob
from dataset_utils import ReplayBuffer
from learner import Learner
from env_goals import ENV_GOALS
class Agent:
    def __init__(self, kwargs, obs_space, action_space):
        self.kwargs = kwargs
        self.obs_space = obs_space
        self.action_space = action_space
        self.mode = "offline"
    
    def make_offline_learner(self, config_str):
        if self.kwargs["load_model"]:
            steps = range(1, self.kwargs["max_steps"] + 1)
            self.offline_agent = Learner(self.obs_space.sample()[np.newaxis],
                                       self.action_space.sample()[np.newaxis], **self.kwargs)
            dirs = glob.glob(self.kwargs["load_model"])
            found = False
            for d in dirs:
                if "_".join(config_str.split("_", 5)[1:5]) in d:
                    self.offline_agent.actor = self.offline_agent.actor.load(self.kwargs["load_model"] + "/pretrained_actor")
                    self.offline_agent.critic = self.offline_agent.critic.load(self.kwargs["load_model"] + "/pretrained_critic")
                    self.offline_agent.value = self.offline_agent.value.load(self.kwargs["load_model"] + "/pretrained_value")
                    self.offline_agent.target_critic = self.offline_agent.target_critic.load(
                        self.kwargs["load_model"] + "/pretrained_target_critic")
                    found = True
            assert found, f"Pretrained model was not found. Possibly because " \
                          f"the config string {'_'.join(config_str.split('_', 5)[1:5])} is not in dir name {d}."
    
        else:
            steps = range(1 - self.kwargs["num_pretraining_steps"],
                          self.kwargs["max_steps"] + 1)
    
            self.offline_agent = Learner(self.obs_space.sample()[np.newaxis],
                                       self.action_space.sample()[np.newaxis], **self.kwargs)
        return steps

    def go_online(self):
        self.save_model()
        self.mode == "online"

    def sample(self):
        raise NotImplementedError
    def sample_actions(self, observation):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError

    def setup_learner(self):
        online_agent = Learner(self.obs_space.sample()[np.newaxis],
                                 self.action_space.sample()[np.newaxis], **self.kwargs)
        if self.kwargs["warm_start"]:
            online_agent.actor = online_agent.actor.replace(params=self.offline_agent.actor.params)
            online_agent.critic = online_agent.critic.replace(params=self.offline_agent.critic.params)
            online_agent.value = online_agent.value.replace(params=self.offline_agent.value.params)
            online_agent.target_critic = \
                online_agent.target_critic.replace(params=self.offline_agent.target_critic.params)
        return online_agent
    def save_model(self):
        if self.mode == "offline":
            agent = self.offline_agent
            atype = "pretrained_"
        else:
            agent = self.online_agent
            atype = "final_"
        agent.actor.save(f"{self.kwargs['save_dir']}/{atype}actor")
        agent.critic.save(f"{self.kwargs['save_dir']}/{atype}critic")
        agent.target_critic.save(f"{self.kwargs['save_dir']}/{atype}target_critic")
        agent.value.save(f"{self.kwargs['save_dir']}/{atype}value")

    def evaluate(env, agent, num_episodes):
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
                        dist = np.sqrt(observation[0] ** 2 + observation[1] ** 2)
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


    def evaluate(self, env, agents, num_episodes, horizon_fn=None, use_offline_fn=None):
        stats = {'return': [], 'length': []}
        all_horizons = []
        agent_types = []
        for _ in range(num_episodes):
            observation, done = env.reset(), False
            time_step = 0
            agent_type = []
            while not done:
                if horizon_fn is not None:
                    h = horizon_fn(env, observation, time_step)
                    all_horizons.append(h)
                agent = agents["online"]
                step_agent_type = 1.0
                if use_offline_fn is not None and use_offline_fn(h, np.mean(agent_type)):
                    agent = agents["offline"]
                    step_agent_type = 0.0
                action = agent.sample_deterministic_actions(observation)
                observation, _, done, info = env.step(action)
                time_step += 1
                agent_type.append(step_agent_type)

            for k in stats.keys():
                stats[k].append(info['episode'][k])
            agent_types.append(np.mean(agent_type))

        for k, v in stats.items():
            stats[k] = np.mean(v)

        stats['all_horizons'] = all_horizons
        stats['agent_type'] = np.mean(agent_types
        return stats


class IQLAgent(Agent):
    def __init__(self):

    def make_replay_buffers(self, obs_space, action_dim, dataset):
        self.replay_buffer_online = ReplayBuffer(obs_space, action_dim,
                                            max(self.kwargs["max_steps"], self.kwargs["init_dataset_size"]))
        self.replay_buffer_online.initialize_with_dataset(dataset, self.kwargs["init_dataset_size"])

    def sample(self):
        batch = self.replay_buffer_online.sample(self.kwargs["batch_size"])
        if self.mode == "offline":
            update_info = self.offline_agent.update(batch)
        else:
            update_info = self.online_agent.update(batch)
        return update_info

    def go_online(self):
        super().go_online()
        self.online_agent = self.offline_agent


class JSRLAgent(Agent):
    def __init__(self):
        self.horizon_idx = 0
        self.eval_returns = []
        self.agent_type = []
        self.n_online_samp = int(0.75 * self.kwargs['batch_size'])
        self.n_offline_samp = self.kwargs['batch_size'] - self.n_online_samp
    def make_replay_buffers(self, obs_space, action_dim, dataset):
        self.replay_buffer_online = ReplayBuffer(obs_space, action_dim, 100000)
        replay_buffer_offline = ReplayBuffer(obs_space, action_dim, self.kwargs["init_dataset_size"])
        replay_buffer_offline.initialize_with_dataset(dataset, self.kwargs["init_dataset_size"])

    def go_online(self):
        super().go_online()
        self.online_agent = self.setup_learner()

        pretrained_stats = self.evaluate(eval_env,
                                         {"online": self.offline_agent,
                                          "offline": None},
                                         100)
        prev_best = pretrained_stats['return']
        if self.kwargs["at_thresholds"]:
            at_thresholds = np.linspace(0, 1, self.kwargs["curriculum_stages"])
        else:
            at_thresholds = [np.inf] * self.kwargs["curriculum_stages"]
        horizon = np.mean(pretrained_stats['all_lens'])

        self.horizon_idx = 0
        self.horizons = np.linspace(horizon, 0, self.kwargs['curriculum_stages'])
        self.athresh = self.at_thresholds[self.horizon_idx]
    def evaluate(self, eval_env):
        super().evaluate(eval_env,
                         {"online": self.online_agent,
                            "offline": self.offline_agent},
                         self.use_offline_env,
                         self.horizon_fn)
    def use_offline_fn(self, h, at):
        return (h <= self.horizons[self.horizon_idx] or at > self.athresh)
    def horizon_fn(self, _env, _obs, time_step):
        return time_step

class JSRLGSAgent(Agent):
    def __init__(self):
        super().__init__()
        self.horizon_idx = 0
        self.eval_returns = []
        self.agent_type = []
        self.n_online_samp = int(0.75 * self.kwargs['batch_size'])
        self.n_offline_samp = self.kwargs['batch_size'] - self.n_online_samp
    def make_replay_buffers(self, obs_space, action_dim, dataset):
        self.replay_buffer_online = ReplayBuffer(obs_space, action_dim, 100000)
        self.replay_buffer_offline = ReplayBuffer(obs_space, action_dim, self.kwargs["init_dataset_size"])
        self.replay_buffer_offline.initialize_with_dataset(dataset, self.kwargs["init_dataset_size"])

    def go_online(self):
        super().go_online()
        self.online_agent = self.setup_learner()

        pretrained_stats = self.evaluate(self.offline_agent, eval_env, 100)
        prev_best = pretrained_stats['return']
        if self.kwargs["at_thresholds"]:
            at_thresholds = np.linspace(0, 1, self.kwargs["curriculum_stages"])
        else:
            at_thresholds = [np.inf] * self.kwargs["curriculum_stages"]
        horizon = np.max(pretrained_stats['goal_dist'])
        self.horizons = np.linspace(0, horizon, self.kwargs['curriculum_stages'])
        self.horizon_idx = 0
        self.athresh = self.at_thresholds[self.horizon_idx]

    def evaluate(self, eval_env):
        super().evaluate(eval_env,
                         {"online": self.online_agent,
                            "offline": self.offline_agent},
                         self.use_offline_env,
                         self.horizon_fn)
    def use_offline_fn(self, h, at):
        return (h <= self.horizons[self.horizon_idx] or at > self.athresh)
    def horizon_fn(self, env, obs, _time_step):
        return ENV_GOALS[self.kwargs["env_name"](env, obs)]