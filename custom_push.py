from gym.envs.registration import registry, register, make, spec
from gym_robotics.envs import rotations, robot_env, utils
from gym_robotics.envs.fetch.push import FetchPushEnv
import numpy as np
from typing import Optional


class FixedFetchPush(FetchPushEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        self.ag = state["achieved_goal"]
        return state

    def step(self, *args, **kwargs):
        state, reward, done, info = super().step(*args, **kwargs)
        state["achieved_goal"] = self.ag
        return state, reward, done, info

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        gripper_state = robot_qpos[-2:]
        gripper_vel = (
            robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric


        obs = np.concatenate(
            [
                grip_pos,
                gripper_state,
                grip_velp,
                gripper_vel,
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": np.squeeze(self.sim.data.get_site_xpos("object0")),
            "desired_goal": self.goal.copy(),
        }

register(
    id='FixedFetchPush-v0',
    entry_point='custom_push:FixedFetchPush',
    max_episode_steps=50,
    kwargs=dict(reward_type='sparse'),
)

register(
    id='FixedFetchPushDense-v0',
    entry_point='custom_push:FixedFetchPush',
    max_episode_steps=50,
    kwargs=dict(reward_type='dense'),
)

class OldFixedFetchPush(FetchPushEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos = None

    def step(self, *args, **kwargs):
        state, reward, done, info = super().step(*args, **kwargs)
        state["achieved_goal"] = self.ag
        return state, reward, done, info

    def get_obs(self):
        return self._get_obs()

    def set_pos(self, pos):
        self.pos = pos

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        self.ag = state["achieved_goal"]
        return state

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        if self.pos is not None:
            object_qpos = self.pos
        else:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
        self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        self.sim.forward()
        return True


register(
    id='OldFixedFetchPush-v0',
    entry_point='custom_push:OldFixedFetchPush',
    max_episode_steps=50,
    kwargs=dict(reward_type='sparse'),
)
