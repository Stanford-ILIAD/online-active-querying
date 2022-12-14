import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import tqdm
import os
import pickle5 as pickle
from stable_baselines3 import SAC
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from functools import partial
from IPython import embed
import multiprocessing
import contextlib
import os
from itertools import starmap
import re
import make_env
import pandas as pd
import abc
import gym.spaces as spaces
import custom_push
from IPython import embed
from gym.wrappers import FlattenObservation
from rollout_fetch import joint_names
from typing import Optional

MAX_T = 500
device = torch.device("cpu")

def save(waypoints, filename):
    with open(filename, 'wb') as f:
        pickle.dump(waypoints, f, protocol=2)


class Robot(abc.ABC):
    def __init__(self, env, query_param, n_samples=50, prec=10, device='cpu'):
        # prec is expert precision
        self.device = torch.device(device)
        self.query_param = query_param
        self.prec = prec
        self.env = env
        self.tasks = [self.sample_env() for _ in range(n_samples)]
        for w in self.tasks:
            w.set_pos(self.env.sim.data.get_joint_qpos("object0:joint"))
            w.reset()
        self.task_dist = torch.ones(n_samples).to(self.device)
        self.policy = SAC.load('fetch_model.zip')

    def sample_env(self):
        env = gym.make(self.env.unwrapped.spec.id)
        env.reset()
        return env

    def get_policy_pi(self, env):
        state = env.env.get_obs()
        x = spaces.flatten(self.env.observation_space, state)
        x = torch.tensor(x).to(device)
        action, _ = self.policy.predict(x)
        return torch.tensor(action, device=self.device)

    def get_policy_q(self, env, action):
        state = env.env.get_obs()
        x = spaces.flatten(self.env.observation_space, state)
        x = torch.tensor(x, device=device)[None, :].expand(action.size(0), -1)
        action = torch.tensor(action, device=device)
        q1, q2 = self.policy.critic(x, action)
        return torch.minimum(q1, q2)

    @abc.abstractmethod
    def _decide_query(self, actions):
        # return whether to query and the best query if so
        pass

    def act(self):
        task_dist = self.task_dist / self.task_dist.sum()
        with torch.no_grad():
            actions = torch.stack([self.get_policy_pi(w) for w in self.tasks])
            best_action = torch.sum(actions * task_dist[:, None], dim=0).detach().numpy()

        should_query, query = self._decide_query(actions)

        if should_query:
            random.shuffle(query)
            return True, query
        else:
            _, reward, done, _ = self.env.step(best_action)
            for env in self.tasks:
                env.step(best_action)
            return False, (reward, done)

    def learn(self, question, answer):
        q_action1 = torch.stack([self.get_policy_q(w, question[answer][None, :]).flatten() for w in self.tasks])
        q_action2 = torch.stack([self.get_policy_q(w, question[1 - answer][None, :]).flatten() for w in self.tasks])
        q_choice = q_action1 - q_action2
        p_q_response = torch.sigmoid(q_choice * self.prec).flatten()
        self.task_dist *= p_q_response



class EvoiRobot(Robot):
    def _decide_query(self, actions, n_samples=200):
        actions1 = torch.stack([torch.tensor(self.env.action_space.sample(), device=self.device) for _ in range(n_samples)])
        actions2 = torch.stack([torch.tensor(self.env.action_space.sample(), device=self.device) for _ in range(n_samples)])
        task_dist = self.task_dist / self.task_dist.sum()
        q_actions1 = torch.stack([self.get_policy_q(w, actions1).flatten() for w in self.tasks])
        q_actions2 = torch.stack([self.get_policy_q(w, actions2).flatten() for w in self.tasks])
        q_choice = q_actions1 - q_actions2
        p_q_response = torch.sigmoid(q_choice * self.prec)
        p_response = torch.sum(task_dist[:, None] * p_q_response, dim=0)

        potential_task_dist1 = p_q_response * task_dist[:, None]
        potential_task_dist1 = potential_task_dist1 / potential_task_dist1.sum(dim=0, keepdim=True)
        potential_task_dist2 = (1 - p_q_response) * task_dist[:, None]
        potential_task_dist2 = potential_task_dist2 / potential_task_dist2.sum(dim=0, keepdim=True)

        potential_action1 = torch.sum(actions[:, None, :] * potential_task_dist1[:, :, None], dim=0)
        potential_action2 = torch.sum(actions[:, None, :] * potential_task_dist2[:, :, None], dim=0)

        potential_q1 = torch.stack([self.get_policy_q(w, potential_action1).flatten() for w in self.tasks])
        potential_q2 = torch.stack([self.get_policy_q(w, potential_action2).flatten() for w in self.tasks])
        potential_value1 = torch.sum(potential_q1 * potential_task_dist1, dim=0)
        potential_value2 = torch.sum(potential_q2 * potential_task_dist2, dim=0)

        average_value = p_response * potential_value1 + (1 - p_response) * potential_value2
        best_action = torch.sum(task_dist[:, None] * actions, dim=0)
        current_q = torch.stack([self.get_policy_q(w, best_action[None, :]).flatten() for w in self.tasks])
        current_value = torch.sum(task_dist[:, None] * current_q, dim=0)
        query_values = average_value - current_value

        best_query_idx = torch.argmax(query_values)
        best_query = [actions1[best_query_idx], actions2[best_query_idx]]
        should_query = query_values[best_query_idx] > self.query_param
        return should_query, best_query

class RandomRobot(Robot):
    def _decide_query(self, actions, n_samples=200):
        should_query = random.random() < self.query_param
        best_query = [self.env.action_space.sample(), self.env.action_space.sample()]
        best_query = [torch.tensor(x, device=self.device) for x in best_query]
        return should_query, best_query



class UncertainRobot(Robot):
    def _decide_query(self, actions, n_samples=200):
        actions1 = torch.stack([torch.tensor(self.env.action_space.sample(), device=self.device) for _ in range(n_samples)])
        actions2 = torch.stack([torch.tensor(self.env.action_space.sample(), device=self.device) for _ in range(n_samples)])
        task_dist = self.task_dist / self.task_dist.sum()
        q_actions1 = torch.stack([self.get_policy_q(w, actions1).flatten() for w in self.tasks])
        q_actions2 = torch.stack([self.get_policy_q(w, actions2).flatten() for w in self.tasks])
        q_choice = q_actions1 - q_actions2
        p_q_response = torch.sigmoid(q_choice * self.prec)
        p_response = torch.sum(task_dist[:, None] * p_q_response, dim=0)

        potential_task_dist1 = p_q_response * task_dist[:, None]
        potential_task_dist1 = potential_task_dist1 / potential_task_dist1.sum(dim=0, keepdim=True)
        potential_task_dist2 = (1 - p_q_response) * task_dist[:, None]
        potential_task_dist2 = potential_task_dist2 / potential_task_dist2.sum(dim=0, keepdim=True)

        potential_action1 = torch.sum(actions[:, None, :] * potential_task_dist1[:, :, None], dim=0)
        potential_action2 = torch.sum(actions[:, None, :] * potential_task_dist2[:, :, None], dim=0)

        potential_q1 = torch.stack([self.get_policy_q(w, potential_action1).flatten() for w in self.tasks])
        mean_q1 = potential_q1 - torch.mean(potential_q1, dim=0)
        potential_q2 = torch.stack([self.get_policy_q(w, potential_action2).flatten() for w in self.tasks])
        mean_q2 = potential_q2 - torch.mean(potential_q2, dim=0)
        potential_var1 = torch.sum(mean_q1 ** 2 * potential_task_dist1, dim=0)
        potential_var2 = torch.sum(mean_q2 ** 2 *  potential_task_dist2, dim=0)

        average_var = p_response * potential_var1 + (1 - p_response) * potential_var2
        best_action = torch.sum(task_dist[:, None] * actions, dim=0)
        current_q = torch.stack([self.get_policy_q(w, best_action[None, :]).flatten() for w in self.tasks])
        mean_q = current_q - torch.mean(current_q, dim=0)
        current_var = torch.sum(task_dist[:, None] * mean_q ** 2, dim=0)
        query_values = current_var - average_var

        best_query_idx = torch.argmax(query_values)
        best_query = [actions1[best_query_idx], actions2[best_query_idx]]
        should_query = query_values[best_query_idx] > self.query_param
        return should_query, best_query



def policy_preference(a, b, agent, env):
    val_a, val_b = agent.get_policy_q(env, a[None, :]).item(), agent.get_policy_q(env, b[None, :]).item()
    return np.argmax([val_a, val_b])

def evaluate(param, env, n_episodes=10):
    """Deep Q-Learning.
    """
    scores = []
    nums_queries = []
    p_task = []
    rollout = args.save_rollout
    for i_episode in range(1, n_episodes+1):
        print('Episode', i_episode)
        env.reset()
        agent = methods[args.method](env, query_param=param)
        expert_policy = partial(policy_preference, agent=agent, env=env)
        rendering = n_render is not None and i_episode % n_render == 0
        if rollout:
            new_s = [env.sim.data.get_joint_qpos('robot0:' + name) for name in joint_names]
            waypoints = [new_s]
        if rendering:
            env.render()
        score = 0
        total_queries = 0
        for t in range(MAX_T):
            should_query, info = agent.act()
            if should_query:
                print('getting pref ', info)
                response = expert_policy(*info)
                agent.learn(info, response)
                total_queries += 1
            else:
                reward, done = info
                if rollout:
                    new_s = [env.sim.data.get_joint_qpos('robot0:' + name) for name in joint_names]
                    waypoints.append(new_s)
                if rendering:
                    env.render()
                score += reward
                if done:
                    break
        p_task.append(np.max((agent.task_dist / agent.task_dist.sum()).detach().numpy()))
        scores.append(score)
        nums_queries.append(total_queries)
        if rollout:
            save(waypoints, f'{args.output}-ep{i_episode}-param{param}-rollout.pkl')
        torch.save((nums_queries, scores, p_task), args.output)



def run_evaluation(seed, param):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make('OldFixedFetchPush-v0')
    return evaluate(param=param, env=env, n_episodes=args.nval) 


parser = argparse.ArgumentParser()
parser.add_argument("--load", default=False, action='store_true')
parser.add_argument("--save_rollout", default=False, action='store_true')
parser.add_argument("--env", type=str, default='door')
parser.add_argument("--npol", type=int, default=75)
parser.add_argument("--nrender", type=int, default=None)
parser.add_argument("--nval", type=int, default=50)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--acc", type=float, default=0.1)
parser.add_argument("--param", type=float, default=1e-3)
parser.add_argument("--output", type=str, default='results')
parser.add_argument("--method", type=str, default='evoi')
args = parser.parse_args()
n_render = args.nrender

methods = dict(evoi=EvoiRobot, uncertainty=UncertainRobot, random=RandomRobot)


if __name__ == "__main__":
    frontiers = []
    seed = args.seed
    run_evaluation(seed, args.param)

