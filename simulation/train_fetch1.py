import os
import gym
import torch
import random
import robosuite as suite
from robosuite.wrappers import GymWrapper
from gym.wrappers.flatten_observation import FlattenObservation

from robosuite.controllers import load_controller_config, ALL_CONTROLLERS

import numpy as np
import make_env
import string
import custom_push

from stable_baselines3 import *
from stable_baselines3.common.callbacks import CheckpointCallback
from gym.wrappers.flatten_observation import FlattenObservation

if __name__ == "__main__":
    run_name = 'logs/fetch-sb3-' + ''.join(random.choice(string.ascii_lowercase) for _ in range(12))

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=run_name)

    env = FlattenObservation(gym.make('FixedFetchPushDense-v0'))
    model = SAC('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    tensorboard_log="./tensorboard/",
                    verbose=1)
    model.learn(int(6e8), callback=checkpoint_callback)

