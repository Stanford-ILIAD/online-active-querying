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


device = torch.device("cpu")


class Robot(abc.ABC):
    def __init__(self, env, query_param, n_samples=50, prec=10, device='cpu'):
        # prec is expert precision
        self.device = torch.device(device)
        self.query_param = query_param
        self.prec = prec
        self.env = env
        self.tasks = [self.sample_w() for _ in range(n_samples)]
        self.task_dist = torch.ones(n_samples).to(self.device)
        self.policy = SAC.load('fetch_model.zip')

    def sample_w(self):
        object_xpos = self.env.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.env.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.env.initial_gripper_xpos[:2] + self.env.np_random.uniform(
                -self.env.obj_range, self.env.obj_range, size=2
            )
        object_qpos = self.env.sim.data.get_joint_qpos("object0:joint")
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self.env.sim.data.set_joint_qpos("object0:joint", object_qpos)

        goal = self.env.initial_gripper_xpos[:3] + self.env.np_random.uniform(
            -self.env.target_range, self.env.target_range, size=3
        )
        goal += self.env.target_offset
        goal[2] = self.env.height_offset
        if self.env.target_in_the_air and self.np_random.uniform() < 0.5:
            goal[2] += self.np_random.uniform(0, 0.45)
        return goal, object_qpos[:3]

    def get_policy_pi(self, w):
        g, ag = w
        def pi(s):
            state = {'observation': s, 'desired_goal': g, 'achieved_goal': ag}
            x = spaces.flatten(self.env.observation_space, state)
            x = torch.tensor(x).to(device)
            action, _ = self.policy.predict(x)
            return torch.tensor(action, device=self.device)
        return pi

    def get_policy_q(self, w):
        g, ag = w
        def Q(s, a):
            state = {'observation': s, 'desired_goal': g, 'achieved_goal': ag}
            x = spaces.flatten(self.env.observation_space, state)
            x = torch.tensor(x, device=device)[None, :].expand(a.size(0), -1)
            a = torch.tensor(a, device=device)
            q1, q2 = self.policy.critic(x, a)
            return torch.minimum(q1, q2)
        return Q

    @abc.abstractmethod
    def _decide_query(self, state, actions):
        # return whether to query and the best query if so
        pass

    def act(self, state):
        task_dist = self.task_dist / self.task_dist.sum()
        with torch.no_grad():
            actions = torch.stack([self.get_policy_pi(w)(state) for w in self.tasks])
            best_action = torch.sum(actions * task_dist[:, None], dim=0)

        should_query, query = self._decide_query(state, actions)

        if should_query:
            random.shuffle(query)
            return True, query
        else:
            return False, best_action

    def learn(self, state, question, answer):
        q_action1 = torch.stack([self.get_policy_q(w)(state, question[answer][None, :]).flatten() for w in self.tasks])
        q_action2 = torch.stack([self.get_policy_q(w)(state, question[1 - answer][None, :]).flatten() for w in self.tasks])
        q_choice = q_action1 - q_action2
        p_q_response = torch.sigmoid(q_choice * self.prec).flatten()
        self.task_dist *= p_q_response



class EvoiRobot(Robot):
    def _decide_query(self, state, actions, n_samples=200):
        actions1 = torch.stack([torch.tensor(self.env.action_space.sample(), device=self.device) for _ in range(n_samples)])
        actions2 = torch.stack([torch.tensor(self.env.action_space.sample(), device=self.device) for _ in range(n_samples)])
        task_dist = self.task_dist / self.task_dist.sum()
        q_actions1 = torch.stack([self.get_policy_q(w)(state, actions1).flatten() for w in self.tasks])
        q_actions2 = torch.stack([self.get_policy_q(w)(state, actions2).flatten() for w in self.tasks])
        q_choice = q_actions1 - q_actions2
        p_q_response = torch.sigmoid(q_choice * self.prec)
        p_response = torch.sum(task_dist[:, None] * p_q_response, dim=0)

        potential_task_dist1 = p_q_response * task_dist[:, None]
        potential_task_dist1 = potential_task_dist1 / potential_task_dist1.sum(dim=0, keepdim=True)
        potential_task_dist2 = (1 - p_q_response) * task_dist[:, None]
        potential_task_dist2 = potential_task_dist2 / potential_task_dist2.sum(dim=0, keepdim=True)

        potential_action1 = torch.sum(actions[:, None, :] * potential_task_dist1[:, :, None], dim=0)
        potential_action2 = torch.sum(actions[:, None, :] * potential_task_dist2[:, :, None], dim=0)

        potential_q1 = torch.stack([self.get_policy_q(w)(state, potential_action1).flatten() for w in self.tasks])
        potential_q2 = torch.stack([self.get_policy_q(w)(state, potential_action2).flatten() for w in self.tasks])
        potential_value1 = torch.sum(potential_q1 * potential_task_dist1, dim=0)
        potential_value2 = torch.sum(potential_q2 * potential_task_dist2, dim=0)

        average_value = p_response * potential_value1 + (1 - p_response) * potential_value2
        best_action = torch.sum(task_dist[:, None] * actions, dim=0)
        current_q = torch.stack([self.get_policy_q(w)(state, best_action[None, :]).flatten() for w in self.tasks])
        current_value = torch.sum(task_dist[:, None] * current_q, dim=0)
        query_values = average_value - current_value

        best_query_idx = torch.argmax(query_values)
        best_query = [actions1[best_query_idx], actions2[best_query_idx]]
        should_query = query_values[best_query_idx] > self.query_param
        return should_query, best_query

class RandomRobot(Robot):
    def _decide_query(self, state, actions, n_samples=200):
        should_query = random.random() < self.query_param
        best_query = [self.env.action_space.sample(), self.env.action_space.sample()]
        return should_query, best_query



class UncertainRobot(Robot):
    def _decide_query(self, state, actions, n_samples=200):
        actions1 = torch.stack([torch.tensor(self.env.action_space.sample(), device=self.device) for _ in range(n_samples)])
        actions2 = torch.stack([torch.tensor(self.env.action_space.sample(), device=self.device) for _ in range(n_samples)])
        task_dist = self.task_dist / self.task_dist.sum()
        q_actions1 = torch.stack([self.get_policy_q(w)(state, actions1).flatten() for w in self.tasks])
        q_actions2 = torch.stack([self.get_policy_q(w)(state, actions2).flatten() for w in self.tasks])
        q_choice = q_actions1 - q_actions2
        p_q_response = torch.sigmoid(q_choice * self.prec)
        p_response = torch.sum(task_dist[:, None] * p_q_response, dim=0)

        potential_task_dist1 = p_q_response * task_dist[:, None]
        potential_task_dist1 = potential_task_dist1 / potential_task_dist1.sum(dim=0, keepdim=True)
        potential_task_dist2 = (1 - p_q_response) * task_dist[:, None]
        potential_task_dist2 = potential_task_dist2 / potential_task_dist2.sum(dim=0, keepdim=True)

        potential_action1 = torch.sum(actions[:, None, :] * potential_task_dist1[:, :, None], dim=0)
        potential_action2 = torch.sum(actions[:, None, :] * potential_task_dist2[:, :, None], dim=0)

        potential_q1 = torch.stack([self.get_policy_q(w)(state, potential_action1).flatten() for w in self.tasks])
        mean_q1 = potential_q1 - torch.mean(potential_q1, dim=0)
        potential_q2 = torch.stack([self.get_policy_q(w)(state, potential_action2).flatten() for w in self.tasks])
        mean_q2 = potential_q2 - torch.mean(potential_q2, dim=0)
        potential_var1 = torch.sum(mean_q1 ** 2 * potential_task_dist1, dim=0)
        potential_var2 = torch.sum(mean_q2 ** 2 *  potential_task_dist2, dim=0)

        average_var = p_response * potential_var1 + (1 - p_response) * potential_var2
        best_action = torch.sum(task_dist[:, None] * actions, dim=0)
        current_q = torch.stack([self.get_policy_q(w)(state, best_action[None, :]).flatten() for w in self.tasks])
        mean_q = current_q - torch.mean(current_q, dim=0)
        current_var = torch.sum(task_dist[:, None] * mean_q ** 2, dim=0)
        query_values = current_var - average_var

        best_query_idx = torch.argmax(query_values)
        best_query = [actions1[best_query_idx], actions2[best_query_idx]]
        should_query = query_values[best_query_idx] > self.query_param
        return should_query, best_query



def policy_preference(state, a, b, Q):
    val_a, val_b = Q(state, a[None, :]).item(), Q(state, b[None, :]).item()
    return np.argmax([val_a, val_b])

def evaluate(param, env, n_episodes=10, w=None):
    """Deep Q-Learning.
    """
    scores = []
    nums_queries = []
    p_task = []
    for i_episode in range(1, n_episodes+1):
        agent = EvoiRobot(env, query_param=param)
        expert_policy = partial(policy_preference, Q=agent.get_policy_q(w))
        rendering = n_render is not None and i_episode % n_render == 0
        state = env.reset()['observation']
        if rendering:
            env.render()
        score = 0
        total_queries = 0
        for t in range(500):
            should_query, action = agent.act(state)
            if should_query:
                response = expert_policy(state, *action)
                agent.learn(state, action, response)
                total_queries += 1
            else:
                state_all, reward, done, _ = env.step(action.numpy())
                state = state_all['observation']
                if rendering:
                    env.render()
                score += reward
                if done:
                    break
        p_task.append(np.max((agent.task_dist / agent.task_dist.sum()).detach().numpy()))
        scores.append(score)
        nums_queries.append(total_queries)
    print(nums_queries, scores, p_task)
    return np.mean(nums_queries), np.mean(scores), np.mean(p_task)



def run_evaluation(seed, param):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make('FetchPush-v1')
    w = (env.goal, env.sim.data.get_joint_qpos("object0:joint")[:3])
    return evaluate(param=param, env=env, n_episodes=args.nval, w=w) 


parser = argparse.ArgumentParser()
parser.add_argument("--load", default=False, action='store_true')
parser.add_argument("--env", type=str, default='door')
parser.add_argument("--npol", type=int, default=75)
parser.add_argument("--nrender", type=int, default=None)
parser.add_argument("--nval", type=int, default=50)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--acc", type=float, default=0.1)
parser.add_argument("--niter", type=int, default=150)
parser.add_argument("--param", type=float, default=1e-3)
parser.add_argument("--output", type=str, default='results')
parser.add_argument("--method", type=str, default='evoi')
args = parser.parse_args()
n_render = args.nrender




if __name__ == "__main__":
    frontiers = []
    for itr in range(args.niter):
        print('Iteration', itr + 1)
        seed = itr + args.seed
        queries, scores, probs = run_evaluation(seed, args.param)
        frontiers.append((queries, scores, probs))
        torch.save(frontiers, args.output)

