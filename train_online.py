import os
from typing import Tuple
import time
import pickle
import argparse
from configs.training_configs import get_config

import gym
import flappy_bird_gym
import numpy as np
import tqdm
from torch.utils.tensorboard import SummaryWriter
import glob
import re
import json


import wrappers
from dataset_utils import (Batch, D4RLDataset, Dataset, ReplayBuffer,
                           split_into_trajectories)
from evaluation import evaluate, evaluate_jsrl
from learner import Learner
from agent import IQLAgent, JSRLAgent, JSRLGSAgent

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
PARSER.add_argument('--at_thresholds', default=False, help='Use agent type thresholds.')
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

    if "Flappy" in env_name:
        env = wrappers.DecreaseActionDim(flappy_bird_gym.make(env_name))
        clip_to_eps = False
    else:
        env = gym.make(env_name)
        clip_to_eps = True

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = None
    if downloaded_dataset:
        with open(downloaded_dataset, "rb") as f:
            dataset = pickle.load(f)
    dataset = D4RLDataset(env, dataset=dataset, clip_to_eps=clip_to_eps)

    if ('halfcheetah' in env_name or 'walker2d' in env_name
          or 'hopper' in env_name):
        normalize(dataset)

    return env, dataset


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
    config_str = f"{timestr}_s{args.seed}_d{args.init_dataset_size}_t{args.tolerance}_nd{args.n_prev_returns}_{args.downloaded_dataset[9:-4]}"
    config_str = config_str.replace(".", "-")

    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'tb', config_str), flush_secs=180)

    env, dataset = make_env_and_dataset(args.env_name, args.seed, downloaded_dataset=args.downloaded_dataset)
    eval_env, _ = make_env_and_dataset(args.env_name, args.seed, downloaded_dataset=args.downloaded_dataset)
    #eval_env = env

    kwargs = vars(args)

    if isinstance(env.action_space, gym.spaces.Discrete):
        kwargs['distribution'] = 'discrete'
        kwargs['policy_dim'] = env.action_space.n
    else:
        kwargs['distribution'] = 'continuous'

    if args.algo == "ft":
        agent = IQLAgent(kwargs, env.observation_space, env.action_space)
    elif args.algo == "jsrl":
        agent = JSRLAgent(kwargs, env.observation_space, env.action_space)
    elif args.algo == "jsrlgs":
        agent = JSRLGSAgent(kwargs, env.observation_space, env.action_space)


    agent.make_replay_buffers(dataset)

    with open(args.save_dir + "/args.txt", "w") as f:
        json.dump(kwargs, f)

    eval_returns = []
    agent_type = []
    observation, done = env.reset(), False
    time_step = 0

    num_steps = agent.make_offline_learner(config_str)

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(num_steps, smoothing=0.1, disable=not args.tqdm):
        if i >= 1:
            if i == 1:
                offline_stats = agent.evaluate(env, horizon_fn=hasattr(agent, "horizon_fn"))
                prev_best = offline_stats['return']
                #prev_best = -np.inf

                agent.go_online(agent.max_horizon(offline_stats))
                online_eval_returns = []
                eval_agent_types = []

            use_offline_agent = False

            if hasattr(agent, "horizon_fn"):
                h = agent.horizon_fn(env, observation, time_step)

            at = 0
            if len(agent_type) > 0:
                at = np.mean(agent_type)

            if hasattr(agent, "use_offline_fn"):
                use_offline_agent = agent.use_offline_fn(h, at)

            if use_offline_agent:
                action = agent.offline_agent.sample_actions(observation, )
                agent_type.append(0.0)
            else:
                action = agent.online_agent.sample_actions(observation, )
                agent_type.append(1.0)
            #try:
                #print(f"current: {h}, horizons: {agent.horizons}, thresh: {agent.horizons[agent.horizon_idx]},"\
                  #f"agent: {np.mean(agent_type)}, ts: {time_step}, at: {agent.athresh}")
            #except UnboundLocalError:
                #print("IQL")

            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, -1, 1)

            next_observation, reward, done, info = env.step(action)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            agent.replay_buffers["online"].insert(observation, action, reward, mask,
                                        float(done), next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                time_step = 0
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'training/{k}', v,
                                              info['total']['timesteps'])
                summary_writer.add_scalar('training/agent_type', np.mean(agent_type), i)
                agent_type = []
                summary_writer.flush()
            else:
                time_step += 1
        else:
            info = {}
            info['total'] = {'timesteps': i}

        batch = agent.sample(i)
        if batch is not None:
            if 'antmaze' in args.env_name:
                batch = Batch(observations=batch.observations,
                              actions=batch.actions,
                              rewards=batch.rewards - 1,
                              masks=batch.masks,
                              next_observations=batch.next_observations)
            update_info = agent.update(batch)

        if i % args.log_interval == 0 and update_info:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', np.array(v), i)
            summary_writer.flush()

        if i % args.eval_interval == 0:
            eval_stats = agent.evaluate(eval_env, num_episodes=args.eval_episodes)

            for k, v in eval_stats.items():
                if isinstance(v, list):
                    v = np.mean(v)
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(args.save_dir, config_str + ".txt"),
                       eval_returns,
                       fmt=['%d', '%.1f'])

            if i > 1:
                online_eval_returns.append((i, eval_stats['return']))
                eval_agent_types.append((i, eval_stats['agent_type']))
                #print(f"Eval agent type: {eval_stats['agent_type']}, len returns: {len(online_eval_returns)},"
                      #f"horizon_idx: {agent.horizon_idx}, return: {eval_stats['return']}")
                agent.update_horizon(np.array(online_eval_returns)[:, 1], prev_best)
                summary_writer.add_scalar('training/horizon', agent.horizon_idx, i)
                np.savetxt(os.path.join(args.save_dir, config_str + "agent_types.txt"),
                           eval_agent_types, fmt=['%d', '%.1f'])

    agent.save_model()

if __name__ == "__main__":
    main()
