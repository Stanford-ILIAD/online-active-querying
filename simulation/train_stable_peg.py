from util.rlkit_custom import rollout
import os

from stable_baselines3 import SAC

import torch
import stable_baselines3 as sb

import rlkit
from rlkit.core import logger
import robosuite as suite
from robosuite.wrappers import GymWrapper

from robosuite.controllers import load_controller_config, ALL_CONTROLLERS

import numpy as np
import make_env
from stable_baselines.gail import ExpertDataset




if __name__ == "__main__":
    params = make_env.sample_params()
    env = make_env.make_peg(*params)


    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard")
    model.learn(total_timesteps=int(1e6))

    model.save(f'logs/peg-{params}')
    experiment(variant)

