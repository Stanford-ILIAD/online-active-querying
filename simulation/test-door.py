import robosuite
import numpy as np
from stable_baselines3 import DQN, SAC
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper


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

try:
    model.load('test-door')
except:
    pass

for _ in range(1000):

    env.reset()

    for i in range(1000):
        action = np.random.randn(env.robots[0].dof) # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display

    model.learn(4000)
    model.save('test-door')
