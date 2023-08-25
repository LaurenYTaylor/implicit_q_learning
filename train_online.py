import os
from typing import Tuple
import time
import pickle
import argparse
from configs.training_configs import get_config

import gym
#import flappy_bird_gym
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
import glob
import re


import wrappers
from dataset_utils import (Batch, D4RLDataset, Dataset, ReplayBuffer,
                           split_into_trajectories)
from evaluation import evaluate, evaluate_jsrl
from learner import Learner

PARSER = argparse.ArgumentParser(prog="IQL Trainer")
PARSER.add_argument('--env_name', default='halfcheetah-expert-v2', help='Environment name.')
PARSER.add_argument('--save_dir', default='./tmp/', help='Tensorboard logging dir.')
PARSER.add_argument('--seed', default=42, help='Random seed.')
PARSER.add_argument('--eval_episodes', default=25, help='Number of episodes used for evaluation.')
PARSER.add_argument('--load_model', default='', help='Saved model path.')
PARSER.add_argument('--downloaded_dataset', default='', help='Path to pre-downloaded dataset.')
PARSER.add_argument('--log_interval', default=1000, help='Logging interval.')
PARSER.add_argument('--eval_interval', default=10000, help='Eval interval.')
PARSER.add_argument('--batch_size', default=256, help='Mini batch size.')
PARSER.add_argument('--num_pretraining_steps', default=int(1e6), help='Number of pretraining steps.')
PARSER.add_argument('--max_steps', default=int(1e6), help='Number of training steps.')
PARSER.add_argument('--episode_length', default=int(700), help='Length of an episode in env_name.')
PARSER.add_argument('--replay_buffer_size', default=100000, help='Replay buffer size (=max_steps if unspecified).')
PARSER.add_argument('--init_dataset_size', help='Offline data size (uses all data if unspecified).')
PARSER.add_argument('--tqdm', action="store_true", help='Use tqdm progress bar.')
PARSER.add_argument('--opt_decay_schedule', default="", help="Decay schedule.")
PARSER.add_argument('--warm_start', default=False, action="store_true", help='Warm-start actor, critics and '
                                                                             'value function.')
PARSER.add_argument('--tolerance', default=0.05, help="return improvement +\- for moving to next curriculum stage.")
PARSER.add_argument('--n_prev_returns', default=5, help='N previous returns to use to determine improvement.'
                                                                             'value function.')
PARSER.add_argument('--curriculum_stages', default=10, help='Curriculum stages.')
PARSER.add_argument('--algo', default="ft", help='Algorithm.', choices=["ft", "jsrl", "jsrlgs"])


def normalize(dataset):
    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int, downloaded_dataset="") -> Tuple[gym.Env, D4RLDataset]:
    if "flappy" in env_name:
        env = flappy_bird_gym.make(env_name)
    else:
        env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = None
    if downloaded_dataset:
        with open(downloaded_dataset, "rb") as f:
            dataset = pickle.load(f)
    dataset = D4RLDataset(env, dataset=dataset)

    if 'antmaze' in env_name:
        dataset.rewards -= 1.0
        pass  # normalized in the batch instead
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in env_name or 'walker2d' in env_name
          or 'hopper' in env_name):
        normalize(dataset)

    return env, dataset


def update_horizon(returns, horizon_idx, prev_best, tolerance=0.05, n=5):
    if len(returns) < n:
        return horizon_idx, prev_best
    rolling_mean = np.mean(returns[-5:])
    if np.isinf(prev_best):
        prev = prev_best
    else:
        prev = prev_best - tolerance * prev_best
    if rolling_mean > prev:
        prev_best = rolling_mean
        return horizon_idx + 1, prev_best
    else:
        return horizon_idx, prev_best


def make_save_dir(load_model, env_name, algo, test=False):
    if test:
        save_dir = "test_logs"
    else:
        save_dir = "logs"
    if not load_model:
        if os.name == "nt":
            save_dir = save_dir + "\\" + env_name
        else:
            save_dir = save_dir + "/" + env_name
        existing = glob.glob(os.path.dirname(save_dir) + "/*")
        if len(existing) > 0:
            nums = [int(re.search("(.*)_([0-9]+)", name).group(2)) for name in existing
                    if re.search("(.*)_([0-9])+", name).group(1) == save_dir]
            try:
                if len(nums) > 0:
                    save_dir = f"{save_dir}_{max(nums) + 1}"
                else:
                    save_dir = f"{save_dir}_0"
            except ValueError:
                print("If using Windows, use backslash '\' in your save_dir instead of forward slash.")
        else:
            save_dir = save_dir + f"_0"
        save_dir += f"_{algo}_ft"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + "/model/", exist_ok=True)
    else:
        save_dir = os.path.dirname(load_model) + "/continued_training/"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + "/model/", exist_ok=True)

    return save_dir


def setup_learner(env, pretrained_agent, kwargs):
    learning_agent = Learner(env.observation_space.sample()[np.newaxis],
                             env.action_space.sample()[np.newaxis], **kwargs)
    if kwargs["warm_start"]:
        learning_agent.actor = learning_agent.actor.replace(params=pretrained_agent.actor.params)
        learning_agent.critic = learning_agent.critic.replace(params=pretrained_agent.critic.params)
        learning_agent.value = learning_agent.value.replace(params=pretrained_agent.value.params)
        learning_agent.target_critic = \
            learning_agent.target_critic.replace(params=pretrained_agent.target_critic.params)
    return learning_agent


def save_model(agent, save_dir, type=""):
    agent.actor.save(f"{save_dir}/" + type + "actor")
    agent.critic.save(f"{save_dir}/" + type + "critic")
    agent.target_critic.save(f"{save_dir}/" + type + "target_critic")
    agent.value.save(f"{save_dir}/" + type + "value")


def main(args=None):
    ext_configs = get_config("online")
    if isinstance(args, dict):
        ext_configs.update(args)
    args, unknown = PARSER.parse_known_args()

    for k, v in ext_configs.items():
        setattr(args, k, v)

    np.random.seed(args.seed)

    if args.max_steps <= 100:
        test = True
    else:
        test = False

    if args.save_dir is None:
        make_save_dir(args.load_model, args.env_name, args.algo, test=test)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    config_str = f"{timestr}_s{args.seed}_d{args.init_dataset_size}_t{args.tolerance}_nd{args.n_prev_returns}"
    config_str = config_str.replace(".", "-")

    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'tb', config_str), flush_secs=180)

    env, dataset = make_env_and_dataset(args.env_name, args.seed, downloaded_dataset=args.downloaded_dataset)

    try:
        action_dim = env.action_space.shape[0]
    except IndexError:
        #action_dim = env.action_space.n
        action_dim = 0

    if args.algo == "ft":
        replay_buffer_online = ReplayBuffer(env.observation_space, action_dim,
                                            max(args.max_steps, args.init_dataset_size))
        replay_buffer_online.initialize_with_dataset(dataset, args.init_dataset_size)
    else:
        replay_buffer_online = ReplayBuffer(env.observation_space, action_dim, 100000)
        #replay_buffer_online.initialize_with_dataset(dataset, 100000)
        replay_buffer_offline = ReplayBuffer(env.observation_space, action_dim, args.init_dataset_size)
        replay_buffer_offline.initialize_with_dataset(dataset, args.init_dataset_size)

    kwargs = vars(args)

    eval_returns = []
    agent_type = []
    observation, done = env.reset(), False
    time_step = 0
    if args.algo != "ft":
        horizon_idx = 0
        eval_returns = []

    if args.load_model:
        steps = range(1, args.max_steps + 1)
        pretrained_agent = Learner(env.observation_space.sample()[np.newaxis],
                                   env.action_space.sample()[np.newaxis], **kwargs)
        dirs = glob.glob(args.load_model)
        for d in dirs:
            if d.split("_", 1)[1] == config_str.split("_", 1)[1]:
                pretrained_agent.actor = pretrained_agent.actor.load(args.load_model + f"/{d}/pretrained_actor")
                pretrained_agent.critic = pretrained_agent.critic.load(args.load_model + f"/{d}/pretrained_critic")
                pretrained_agent.value = pretrained_agent.value.load(args.load_model + f"/{d}/pretrained_value")
                pretrained_agent.target_critic = pretrained_agent.target_critic.load(
                    args.load_model + f"/{d}/pretrained_target_critic")
    else:
        steps = range(1 - args.num_pretraining_steps,
                      args.max_steps + 1)
        pretrained_agent = Learner(env.observation_space.sample()[np.newaxis],
                                   env.action_space.sample()[np.newaxis], **kwargs)


    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(steps, smoothing=0.1, disable=not args.tqdm):
        if i%10000 == 0:
            print(f"{config_str}: Time Step {i}")
        if i >= 1:
            if i == 1:
                if args.algo == "ft":
                    learning_agent = pretrained_agent
                else:
                    prev_best = -np.inf
                    n_online_samp = int(0.75 * args.batch_size)
                    n_offline_samp = args.batch_size - n_online_samp
                    learning_agent = setup_learner(env, pretrained_agent, kwargs)
                    if "gs" in args.algo:
                        horizon = evaluate(pretrained_agent, env, 100)['goal_dist']
                        horizons = np.linspace(0, horizon, args.curriculum_stages)
                    else:
                        horizon = evaluate(pretrained_agent, env, 100)['length']
                        horizons = np.linspace(horizon, 0, args.curriculum_stages)

                    save_model(pretrained_agent, args.save_dir + f"/model/{config_str}", type="pretrained_")
            if args.algo == "jsrlgs":
                if "antmaze" in args.env_name:
                    h = np.linalg.norm(np.array(env.target_goal) - np.array(env.get_xy()))
                else:
                    h = np.abs(observation[1])
            elif args.algo == "jsrl":
                h = time_step

            if args.algo == "jsrl" and h <= horizons[horizon_idx]:
                action = pretrained_agent.sample_actions(observation, )
                agent_type.append(0.0)
            elif args.algo == "jsrlgs" and h >= horizons[horizon_idx]:
                action = pretrained_agent.sample_actions(observation, )
                agent_type.append(0.0)
            else:
                action = learning_agent.sample_actions(observation, )
                agent_type.append(1.0)

            action = np.clip(action, -1, 1)
            next_observation, reward, done, info = env.step(action)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer_online.insert(observation, action, reward, mask,
                                        float(done), next_observation)
            observation = next_observation

            if done:
                #it, count = np.unique(agent_type, return_counts=True)
                observation, done = env.reset(), False
                time_step = 0
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'training/{k}', v,
                                              info['total']['timesteps'])
                if args.algo != "ft":
                    summary_writer.add_scalar('training/horizon', horizons[horizon_idx], i)
                    summary_writer.add_scalar('training/agent_type', np.mean(agent_type), i)
                    agent_type = []
                summary_writer.flush()
            else:
                time_step += 1
        else:
            info = {}
            info['total'] = {'timesteps': i}

        if args.algo == "ft":
            batch = replay_buffer_online.sample(args.batch_size)
        else:
            if i < 1:
                batch = replay_buffer_offline.sample(args.batch_size)
            else:
                online_batch = replay_buffer_online.sample_fifo(n_online_samp)
                offline_batch = replay_buffer_offline.sample(n_offline_samp)
                batch = {}
                for k in online_batch._asdict().keys():
                    batch[k] = np.concatenate((online_batch._asdict()[k], offline_batch._asdict()[k]))
                batch = Batch(**batch)

        if 'antmaze' in args.env_name:
            batch = Batch(observations=batch.observations,
                          actions=batch.actions,
                          rewards=batch.rewards - 1,
                          masks=batch.masks,
                          next_observations=batch.next_observations)

        if i >= 1:
            agent = learning_agent
        else:
            agent = pretrained_agent

        # don't update until there's a batch size of online data in buffer
        # 0.75*batch size because jsrl takes 75% of batch from online buffer and 25% from offline
        if i < 1 or i >= int(0.75*args.batch_size)+1:
            update_info = agent.update(batch)

        if i % args.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', np.array(v), i)
                # else:
                # summary_writer.add_histogram(f'training/{k}', np.array(v), i)
            summary_writer.flush()

        if i % args.eval_interval == 0:
            if i < 1 or args.algo == "ft":
                eval_stats = evaluate(agent, env, args.eval_episodes)
            else:
                eval_stats = evaluate_jsrl(agent, env, args.eval_episodes, pretrained_agent, horizons[horizon_idx], args.algo)
            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(args.save_dir, config_str+".txt"),
                       eval_returns,
                       fmt=['%d', '%.1f'])
            if args.algo != "ft" and i > 0:
                if len(horizons) > 0 and horizon_idx != len(horizons) - 1:
                    horizon_idx, prev_best = update_horizon([e[1] for e in eval_returns], horizon_idx, prev_best,
                                                            tolerance=args.tolerance, n=args.n_prev_returns)
    summary_writer.close()
    save_model(learning_agent, args.save_dir + f"/model/{config_str}", type="final_")


if __name__ == "__main__":
    main()
