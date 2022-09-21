import robosuite
import numpy as np
from stable_baselines3 import DQN, SAC
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper
from time import sleep
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
import argparse
from sampler import *


def sample_params():
    x = np.random.uniform(-0.3, 0.1)
    y = np.random.uniform(-0.3, 0.3)
    theta = np.random.uniform(0, 2*np.pi)
    return x, y, theta

def make_env(x, y, theta):

    nut_names = ("SquareNut", "RoundNut")

    table_offset = np.array((0, 0, 0.82))
    placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
    for nut_name, sign in zip(nut_names, (1, -1)):
        placement_initializer.append_sampler(
            sampler=FixedPositionSampler(
                name=f"{nut_name}Sampler",
                x_pos=x,
                y_pos=sign * y,
                rotation=theta,
                rotation_axis="z",
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=table_offset,
                z_offset=0.02,
            )
        )

    env = robosuite.make(
        env_name="NutAssembly", # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        #render_camera="agentview",
        single_object_mode=2, # env has 1 nut instead of 2
        nut_type="round",
        #ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=True,
        use_object_obs=True,
        placement_initializer=placement_initializer,
    )
    env = GymWrapper(env)
    return env

params = 
env = make_env(*sample_params())
# reset the environment
env.reset()
env.render()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
