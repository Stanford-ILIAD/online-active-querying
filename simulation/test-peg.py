import robosuite
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
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
from sampler import FixedPositionSampler
# Save a checkpoint every 1000 steps
callback = CheckpointCallback(save_freq=2000, save_path='./checkpoints/',
                                         name_prefix='peg_model')
nut_names = ("SquareNut", "RoundNut")

tmp_path = "./log"
# set up logger
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])


X=0.21
Y=0.11
T=0.

table_offset = np.array((0, 0, 0.82))
placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
for nut_name, sign in zip(nut_names, (1, -1)):
    print(X, sign*Y)
    placement_initializer.append_sampler(
        sampler=FixedPositionSampler(
            name=f"{nut_name}Sampler",
            x_pos=X,
            y_pos=sign * Y,
            rotation=T,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=table_offset,
            z_offset=0.02,
        )
    )




controller_config = load_controller_config(default_controller="OSC_POSE")

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
if True:
    model = SAC.load('checkpoints/peg_model_572000_steps.zip', env)
else:
    model = SAC('MlpPolicy', env,
                        policy_kwargs=dict(net_arch=[256, 256]),
                        learning_rate=5e-4,
                        buffer_size=15000,
                        learning_starts=200,
                        batch_size=32,
                        gamma=0.8,
                        train_freq=1,
                        gradient_steps=1,
                        target_update_interval=50,
                        verbose=1)

model.set_logger(new_logger)

model.learn(1e8, callback=callback, log_interval=1)
