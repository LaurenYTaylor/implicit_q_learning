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

def make_env(env_name: str,
             seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env

def main(data_size, seed):
    FLAGS([sys.argv[0]])
    FLAGS.env_name = "antmaze-umaze-v0"
    FLAGS.eval_episodes = 5
    FLAGS.eval_interval = 10000
    FLAGS.num_pretraining_steps = 1000000
    FLAGS.max_steps = 1000000
    FLAGS.init_dataset_size = data_size
    FLAGS.seed = seed
    FLAGS.load_model = "logs\\antmaze-umaze-v0_8_jsrl_ft\\model\\20230720-155136_s0_d1000000"

    np.random.seed(seed)

    kwargs = dict(FLAGS.config)

    env = make_env(FLAGS.env_name, FLAGS.seed)

    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis], **kwargs)


    agent.actor = agent.actor.load(FLAGS.load_model + "\\actor")
    agent.critic = agent.critic.load(FLAGS.load_model + "\\critic")
    agent.value = agent.value.load(FLAGS.load_model + "\\value")
    agent.target_critic = agent.target_critic.load(FLAGS.load_model + "\\target_critic")

    eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

    print(eval_stats)

if __name__ == "__main__":
    main(1000, 0)