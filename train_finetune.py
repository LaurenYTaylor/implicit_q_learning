import os
from typing import Tuple
import sys

import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from torch.utils.tensorboard import SummaryWriter
import glob
import re
import time

import wrappers
from dataset_utils import (Batch, D4RLDataset, ReplayBuffer,
                           split_into_trajectories)
from evaluation import evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 100,
                     'Number of episodes used for evaluation.')
flags.DEFINE_string('load_model', '', 'Saved model path.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('num_pretraining_steps', int(1e6),
                     'Number of pretraining steps.')
flags.DEFINE_integer('replay_buffer_size', 2000000,
                     'Replay buffer size (=max_steps if unspecified).')
flags.DEFINE_integer('init_dataset_size', None,
                     'Offline data size (uses all data if unspecified).')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('load_critics', False, 'Warm-start critics and value function.')


config_flags.DEFINE_config_file(
    'config',
    'configs/antmaze_finetune_config.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


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
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        # dataset.rewards -= 1.0
        pass  # normalized in the batch instead
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in FLAGS.env_name or 'walker2d' in FLAGS.env_name
          or 'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset

def make_save_dir_remote(load_model, env_name):
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

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + "/model/", exist_ok=True)
    else:
        save_dir = os.path.dirname(load_model) + "/continued_training/"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir + "/model/", exist_ok=True)

    return save_dir
def make_save_dir():
    if not FLAGS.load_model:
        if os.name == "nt":
            FLAGS.save_dir = FLAGS.save_dir + "\\" + FLAGS.env_name
        else:
            FLAGS.save_dir = FLAGS.save_dir + "/" + FLAGS.env_name
        existing = glob.glob(os.path.dirname(FLAGS.save_dir) + "/*")
        if len(existing) > 0:
            nums = [int(re.search("(.*)_([0-9]+)", name).group(2)) for name in existing
                    if re.search("(.*)_([0-9])+", name).group(1) == FLAGS.save_dir]
            try:
                if len(nums) > 0:
                    FLAGS.save_dir = f"{FLAGS.save_dir}_{max(nums) + 1}"
                else:
                    FLAGS.save_dir = f"{FLAGS.save_dir}_0"
            except ValueError:
                print("If using Windows, use backslash '\' in your save_dir instead of forward slash.")
        else:
            FLAGS.save_dir = FLAGS.save_dir + f"_0"

        os.makedirs(FLAGS.save_dir, exist_ok=True)
        os.makedirs(FLAGS.save_dir + "/model/", exist_ok=True)
    else:
        FLAGS.save_dir = os.path.dirname(FLAGS.load_model) + "/continued_training/"
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        os.makedirs(FLAGS.save_dir + "/model/", exist_ok=True)

    return FLAGS.save_dir

def main(seed, data_size, save_dir=None):
    FLAGS([sys.argv[0]])
    FLAGS.env_name = "antmaze-umaze-v0"
    FLAGS.eval_episodes = 25
    FLAGS.eval_interval = 10000
    FLAGS.num_pretraining_steps = 1000000
    FLAGS.max_steps = 1000000
    FLAGS.init_dataset_size = data_size
    FLAGS.seed = seed

    np.random.seed(seed)

    if save_dir is None:
        FLAGS.save_dir = "logs"
        make_save_dir()
    else:
        FLAGS.save_dir = save_dir

    timestr = time.strftime("%Y%m%d-%H%M%S")
    config_str = f"{timestr}_s{FLAGS.seed}_d{FLAGS.init_dataset_size}"

    summary_writer = SummaryWriter(os.path.join(FLAGS.save_dir, 'tb',
                                    timestr+f"_s{FLAGS.seed}_d{FLAGS.init_dataset_size}"),
                                    flush_secs=180)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(env.observation_space, action_dim,
                                 FLAGS.replay_buffer_size or FLAGS.max_steps)
    replay_buffer.initialize_with_dataset(dataset, FLAGS.init_dataset_size)

    kwargs = dict(FLAGS.config)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis], **kwargs)

    eval_returns = []
    observation, done = env.reset(), False

    # Use negative indices for pretraining steps.
    for i in tqdm.tqdm(range(1 - FLAGS.num_pretraining_steps,
                             FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i >= 1:
            action = agent.sample_actions(observation, )
            action = np.clip(action, -1, 1)
            next_observation, reward, done, info = env.step(action)

            if not done or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            replay_buffer.insert(observation, action, reward, mask,
                                 float(done), next_observation)
            observation = next_observation

            if done:
                observation, done = env.reset(), False
                for k, v in info['episode'].items():
                    summary_writer.add_scalar(f'training/{k}', v,
                                              info['total']['timesteps'])
                summary_writer.flush()
        else:
            info = {}
            info['total'] = {'timesteps': i}

        batch = replay_buffer.sample(FLAGS.batch_size)
        if 'antmaze' in FLAGS.env_name:
            batch = Batch(observations=batch.observations,
                          actions=batch.actions,
                          rewards=batch.rewards - 1,
                          masks=batch.masks,
                          next_observations=batch.next_observations)
        update_info = agent.update(batch)


        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', np.array(v), i)
                else:
                    summary_writer.add_histogram(f'training/{k}', np.array(v), i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, timestr+f"_s{FLAGS.seed}_d{FLAGS.init_dataset_size}.txt"),
                       eval_returns,
                       fmt=['%d', '%.1f'])
    summary_writer.close()
    agent.actor.save(f"{FLAGS.save_dir}/model/{config_str}/actor")
    agent.critic.save(f"{FLAGS.save_dir}/model/{config_str}/critic")
    agent.target_critic.save(f"{FLAGS.save_dir}/model/{config_str}/target_critic")
    agent.value.save(f"{FLAGS.save_dir}/model/{config_str}/value")


#if __name__ == '__main__':
#    app.run(main)
