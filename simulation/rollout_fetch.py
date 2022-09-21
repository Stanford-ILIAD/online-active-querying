import pickle, uuid
import numpy as np
import gym
import custom_push
import stable_baselines3 as sb3

joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]

def save(waypoints, filename):
    with open(filename, 'wb') as f:
        pickle.dump(waypoints, f, protocol=2)

def rollout(env):
    state = env.reset()
    policy = sb3.SAC.load('fetch_model.zip')
    new_s = [env.sim.data.get_joint_qpos('robot0:' + name) for name in joint_names]
    waypoints = [new_s]
    env.render()
    while 1:
        action, _ = policy.predict(state)
        state, _, done, _ = env.step(action)
        env.render()
        new_s = [env.sim.data.get_joint_qpos('robot0:' + name) for name in joint_names]
        waypoints.append(new_s)
        if done:
            break
    return waypoints

if __name__ == '__main__':
    env = gym.wrappers.FlattenObservation(gym.make('FixedFetchPushDense-v0'))
    waypoints = rollout(env)
    save(waypoints, 'traj.pkl')
            
