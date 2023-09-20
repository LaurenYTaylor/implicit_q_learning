import numpy as np
import glob
from dataset_utils import ReplayBuffer
from learner import Learner
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

    def evaluate(env, num_episodes):
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
    def make_replay_buffers(self, obs_space, action_dim, dataset):
        self.replay_buffer_online = ReplayBuffer(obs_space, action_dim, 100000)
        replay_buffer_offline = ReplayBuffer(obs_space, action_dim, self.kwargs["init_dataset_size"])
        replay_buffer_offline.initialize_with_dataset(dataset, self.kwargs["init_dataset_size"])

    def go_online(self):
        super().go_online()
        self.n_online_samp = int(0.75 * self.kwargs['batch_size'])
        self.n_offline_samp = self.kwargs['batch_size'] - self.n_online_samp
        self.online_agent = self.setup_learner()

        pretrained_stats = evaluate(pretrained_agent, eval_env, 100)
        prev_best = pretrained_stats['return']
        if self.kwargs["at_thresholds"]:
            at_thresholds = np.linspace(0, 1, self.kwargs["curriculum_stages"])
        else:
            at_thresholds = [np.inf] * self.kwargs["curriculum_stages"]
        if "gs" in args.algo:
            horizon = np.max(pretrained_stats['goal_dist'])
            # percentiles = np.percentile(pretrained_stats['goal_dist'], np.linspace(0,100,args.curriculum_stages))[:]
            horizons = np.linspace(0, horizon, self.kwargs['curriculum_stages'])
        else:
            horizon = np.mean(pretrained_stats['all_lens'])
            horizons = np.linspace(horizon, 0, self.kwargs['curriculum_stages'])



class JSRLGSAgent(Agent):
    def __init__(self):
        self.horizon_idx = 0
        self.eval_returns = []
        self.agent_type = []
    def make_replay_buffers(self, obs_space, action_dim, dataset):
        self.replay_buffer_online = ReplayBuffer(obs_space, action_dim, 100000)
        self.replay_buffer_offline = ReplayBuffer(obs_space, action_dim, self.kwargs["init_dataset_size"])
        self.replay_buffer_offline.initialize_with_dataset(dataset, self.kwargs["init_dataset_size"])