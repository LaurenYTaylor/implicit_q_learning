import numpy as np
import glob
from dataset_utils import ReplayBuffer
from learner import Learner
from env_goals import ENV_GOALS
from dataset_utils import Batch
class Agent:
    def __init__(self, params, obs_space, action_space):
        self.params = params
        self.obs_space = obs_space
        self.action_space = action_space
        self.mode = "offline"
    
    def make_offline_learner(self, config_str):
        if self.params["load_model"]:
            steps = range(1, self.params["max_steps"] + 1)
            self.offline_agent = Learner(self.obs_space.sample()[np.newaxis],
                                       self.action_space.sample()[np.newaxis], **self.params)
            dirs = glob.glob(self.params["load_model"])
            found = False
            for d in dirs:
                if "_".join(config_str.split("_", 5)[1:5]) in d and self.params.downloaded_dataset[9:-4] in d:
                    print(f"Using saved model: {d}")
                    if "*" in self.params.load_model:
                        dir = d
                    else:
                        dir = self.params.load_model
                    self.offline_agent.actor = self.offline_agent.actor.load(
                        dir + "/pretrained_actor")
                    self.offline_agent.critic = self.offline_agent.critic.load(
                        dir + "/pretrained_critic")
                    self.offline_agent.value = self.offline_agent.value.load(
                        dir + "/pretrained_value")
                    self.offline_agent.target_critic = self.offline_agent.target_critic.load(
                        dir + "/pretrained_target_critic")
                    found = True
            assert found, f"Pretrained model was not found. Possibly because " \
                          f"the config string {'_'.join(config_str.split('_', 5)[1:5])} is not in dir name {d}."
    
        else:
            steps = range(1 - self.params["num_pretraining_steps"],
                          self.params["max_steps"] + 1)
    
            self.offline_agent = Learner(self.obs_space.sample()[np.newaxis],
                                       self.action_space.sample()[np.newaxis], **self.params)
        return steps

    def go_online(self):
        self.save_model()
        self.mode = "online"

    def setup_learner(self):
        online_agent = Learner(self.obs_space.sample()[np.newaxis],
                                 self.action_space.sample()[np.newaxis], **self.params)
        if self.params["warm_start"]:
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
        agent.actor.save(f"{self.params['save_dir']}/model/{atype}actor")
        agent.critic.save(f"{self.params['save_dir']}/model/{atype}critic")
        agent.target_critic.save(f"{self.params['save_dir']}/model/{atype}target_critic")
        agent.value.save(f"{self.params['save_dir']}/model/{atype}value")

    def evaluate(self, env, num_episodes=100, use_offline_fn=False, horizon_fn=False):
        stats = {'return': [], 'length': []}
        max_horizons = []
        agent_types = []
        for _ in range(num_episodes):
            observation, done = env.reset(), False
            time_step = 0
            agent_type = []
            all_horizons = []
            while not done:
                if horizon_fn:
                    h = self.horizon_fn(env, observation, time_step)
                    all_horizons.append(h)
                step_agent_type = 0.0
                if self.mode == "offline":
                    agent = self.offline_agent
                elif self.mode == "online":
                    at = 0
                    if len(agent_type) > 0:
                        at = np.mean(agent_type)

                    if use_offline_fn and self.use_offline_fn(h, at):
                        agent = self.offline_agent
                    else:
                        step_agent_type = 1.0
                        agent = self.online_agent
                action = agent.sample_deterministic_actions(observation)
                observation, _, done, info = env.step(action)
                time_step += 1
                agent_type.append(step_agent_type)

            for k in stats.keys():
                stats[k].append(info['episode'][k])
            agent_types.append(np.mean(agent_type))
            if len(all_horizons) > 0:
                max_horizons.append(np.max(all_horizons))

        for k, v in stats.items():
            stats[k] = np.mean(v)

        stats['horizon'] = max_horizons
        stats['agent_type'] = np.mean(agent_types)
        return stats

    def sample(self, time_step):
        if self.mode == "offline":
            batch = self.replay_buffers["offline"].sample(self.params['batch_size'])
        elif self.mode == "online":
            #if time_step >= self.n_online_samp and time_step >= self.n_offline_samp:
            if time_step >= 0.75*self.params["batch_size"]:
                online_batch = self.replay_buffers["online"].sample_fifo(self.n_online_samp)
                offline_batch = self.replay_buffers["offline"].sample(self.n_offline_samp)
                batch = {}
                for k in online_batch._asdict().keys():
                    online_v = online_batch._asdict()[k]
                    batch[k] = np.concatenate((online_v, offline_batch._asdict()[k]))
                batch = Batch(**batch)
            else:
                return None
        return batch

    def update(self, batch):
        if self.mode == "offline":
            update_info = self.offline_agent.update(batch)
        else:
            update_info = self.online_agent.update(batch)
        return update_info
    def update_horizon(self, returns, prev_best):
        n = self.params["n_prev_returns"]
        tolerance = self.params["tolerance"]
        if len(self.horizons) > 0 and self.horizon_idx != len(self.horizons) - 1:
            if len(returns) < n:
                return prev_best
            rolling_mean = np.mean(returns[-n:])
            if np.isinf(prev_best):
                prev_tol = prev_best
            else:
                prev_tol = prev_best - tolerance * prev_best
            if rolling_mean >= prev_tol:
                self.horizon_idx += 1
                return rolling_mean
            else:
                return prev_best
    def max_horizon(self, _):
        pass
class IQLAgent(Agent):
    def __init__(self, params, obs_space, act_space):
        super().__init__(params, obs_space, act_space)
        self.n_offline_samp = self.params["batch_size"]
        self.n_online_samp = 0
    def make_replay_buffers(self, dataset):
        replay_buffer_online = ReplayBuffer(self.obs_space, self.action_space,
                                            max(self.params["max_steps"], self.params["init_dataset_size"]))
        replay_buffer_online.initialize_with_dataset(dataset, self.params["init_dataset_size"])
        self.replay_buffers = {"offline": replay_buffer_online,
                               "online": replay_buffer_online}
        return

    def go_online(self, _):
        super().go_online()
        self.online_agent = self.offline_agent
        self.horizons = []
        self.horizon_idx = 0


class JSRLAgent(Agent):
    def __init__(self, params, obs_space, act_space):
        super().__init__(params, obs_space, act_space)
        self.n_online_samp = int(0.75 * self.params['batch_size'])
        self.n_offline_samp = self.params['batch_size'] - self.n_online_samp
    def make_replay_buffers(self, dataset):
        replay_buffer_online = ReplayBuffer(self.obs_space, self.action_space, 100000)
        replay_buffer_offline = ReplayBuffer(self.obs_space, self.action_space, self.params["init_dataset_size"])
        replay_buffer_offline.initialize_with_dataset(dataset, self.params["init_dataset_size"])
        self.replay_buffers = {"online": replay_buffer_online, "offline": replay_buffer_offline}

    def go_online(self, max_horizon):
        super().go_online()
        self.online_agent = self.setup_learner()

        if self.params["at_thresholds"]:
            self.at_thresholds = np.linspace(0, 1, self.params["curriculum_stages"])
        else:
            self.at_thresholds = [np.inf] * self.params["curriculum_stages"]
        self.horizon_idx = 0
        self.horizons = np.linspace(max_horizon, 0, self.params['curriculum_stages'])

    @property
    def athresh(self):
        return self.at_thresholds[self.horizon_idx]

    def evaluate(self, eval_env, num_episodes=100, use_offline_fn=None, horizon_fn=None):
        if self.mode == "offline":
            if use_offline_fn is None:
                use_offline_fn = False
            if horizon_fn is None:
                horizon_fn = False
        else:
            if use_offline_fn is None:
                use_offline_fn = True
            if horizon_fn is None:
                horizon_fn = True
        eval_stats = super().evaluate(eval_env, num_episodes, use_offline_fn, horizon_fn)
        return eval_stats
    def use_offline_fn(self, h, at):
        return (h <= self.horizons[self.horizon_idx] or at >= self.athresh)
    def horizon_fn(self, _env, _obs, time_step):
        return time_step
    def max_horizon(self, stats):
        horizon = np.mean(stats['horizon'])
        return horizon
class JSRLGSAgent(Agent):
    def __init__(self, params, obs_space, act_space):
        super().__init__(params, obs_space, act_space)
        self.n_online_samp = int(0.75 * self.params['batch_size'])
        self.n_offline_samp = self.params['batch_size'] - self.n_online_samp
    def make_replay_buffers(self, dataset):
        replay_buffer_online = ReplayBuffer(self.obs_space, self.action_space, 100000)
        replay_buffer_offline = ReplayBuffer(self.obs_space, self.action_space, self.params["init_dataset_size"])
        replay_buffer_offline.initialize_with_dataset(dataset, self.params["init_dataset_size"])
        self.replay_buffers = {"online": replay_buffer_online, "offline": replay_buffer_offline}

    def go_online(self, max_horizon):
        super().go_online()
        self.online_agent = self.setup_learner()
        if self.params["at_thresholds"]:
            self.at_thresholds = np.linspace(0, 1, self.params["curriculum_stages"])
        else:
            self.at_thresholds = [np.inf] * self.params["curriculum_stages"]
        self.horizons = np.linspace(0, max_horizon, self.params['curriculum_stages'])
        self.horizon_idx = 0

    @property
    def athresh(self):
        return self.at_thresholds[self.horizon_idx]

    def evaluate(self, eval_env, num_episodes=100, use_offline_fn=True, horizon_fn=True):
        if self.mode == "offline":
            if use_offline_fn is None:
                use_offline_fn = False
            if horizon_fn is None:
                horizon_fn = False
        else:
            if use_offline_fn is None:
                use_offline_fn = True
            if horizon_fn is None:
                horizon_fn = True
        eval_stats = super().evaluate(eval_env, num_episodes, use_offline_fn, horizon_fn)
        return eval_stats
    def use_offline_fn(self, h, at):
        return ((h >= self.horizons[self.horizon_idx] and
                 h <= self.horizons[-1]) or
                at >= self.athresh)
    def horizon_fn(self, env, obs, _time_step):
        return ENV_GOALS[self.params["env_name"]](env, obs)
    def max_horizon(self, stats):
        horizon = np.max(stats['horizon'])
        return horizon