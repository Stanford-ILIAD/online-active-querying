import robosuite
import numpy as np
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper
from time import sleep
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from sampler import FixedPositionSampler
from robosuite import load_controller_config

import gym
from environments import CustomDoor, CustomNut


def fetch_env():
    return gym.make('FetchPush-v1')


def sample_door_params():
    x = np.random.uniform(-0.3, 0.3)
    y = np.random.uniform(-0.3, 0.3)
    z = np.random.uniform(-0.2, 0.2)
    return x, y, z


def sample_params():
    x = np.random.uniform(-0.3, 0.1)
    y = np.random.uniform(-0.3, 0.3)
    theta = np.random.uniform(0, 2*np.pi)
    return x, y, theta

def make_door(x, y, z):
    env = robosuite.make(
        env_name="CustomDoor", 
        robots="Panda",  
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=True,
        use_object_obs=False,
        base=(x, y, z),
    )
    env = GymWrapper(env)
    return env

def make_peg(x, y, theta):
    nut_names = ("SquareNut", "RoundNut")
    table_offset = np.array((0, 0, 0.82))
    config = load_controller_config(default_controller='OSC_POSE')
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
        env_name="NutAssembly",
        robots="Panda",  
        has_renderer=True,
        has_offscreen_renderer=False,
        single_object_mode=2, 
        nut_type="round",
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=True,
        use_object_obs=False,
        placement_initializer=placement_initializer,
        horizon=500,
        controller_configs=config,
    )
    env = GymWrapper(env)
    return env
