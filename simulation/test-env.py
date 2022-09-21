import robosuite
import numpy as np
from stable_baselines3 import DQN, SAC
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper
from time import sleep

from stable_baselines3.common.logger import configure

tmp_path = "./log"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


controller_config = load_controller_config(default_controller="OSC_POSE")

env = robosuite.make(
    env_name="Door", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True,
)
env = GymWrapper(env)
env.reset()
env.render()
sleep(5)
env.reset()
env.render()
