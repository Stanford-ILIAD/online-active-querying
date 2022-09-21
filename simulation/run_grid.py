import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import tqdm
import os
import custom_highway_env
import pickle5 as pickle
from stable_baselines3 import DQN
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
import abc
from gridworld import *
import value_iteration
import tqdm

device = torch.device("cpu")
MAX_T = 50


class Robot(abc.ABC):
    def __init__(self, query_param, policies, prec=10, device=device):
        # prec is expert precision
        self.device = torch.device(device)
        self.query_param = query_param
        self.prec = prec
        self.policies = policies
        self.n_policies = len(self.policies)
        self.task_dist = torch.ones(self.n_policies).to(self.device)

    def get_policy_q(self, policy):
        def Q(x):
            return torch.tensor(policy[x], device=self.device)
        return Q

    @abc.abstractmethod
    def _decide_query(self, q_local):
        # return whether to query and the best query if so
        pass

    def act(self, obs):
        with torch.no_grad():
            q_local = torch.stack([self.get_policy_q(policy)(obs) for policy in self.policies])
            action_values = (q_local * self.task_dist[:, None]).sum(dim=0)

        self.q_local = q_local
        self.action_values = action_values.cpu().data.numpy()
        best_action = np.argmax(self.action_values)
        should_query, best_query = self._decide_query(q_local)

        if should_query:
            random.shuffle(best_query)
            return True, best_query
        else:
            return False, best_action

    def learn(self, obs, question, answer):
        # we asked the user question (an array of size 2, each entry is an action), and they chose answer=0 or answer=1 at state obs
        q_choice = self.q_local[:, question[answer]] - self.q_local[:, question[1 - answer]]
        self.task_dist *= torch.sigmoid(q_choice * self.prec)


class UncertainRobot(Robot):
    def _decide_query(self, q_local):
        best_action = np.argsort(self.action_values)[-1]
        uncertainty = (
            (q_local - q_local.mean(axis=0, keepdims=True)) ** 2 
            * self.task_dist[:, None]
        ).sum(dim=0)[best_action]
        best_query = np.argsort(self.action_values)[-1:-3:-1]
        should_query = uncertainty > self.query_param
        return should_query, best_query

class RandomRobot(Robot):
    def _decide_query(self, q_local):
        best_query = np.argsort(self.action_values)[-1:-3:-1]
        should_query = random.random() < self.query_param
        return should_query, best_query

class OriginalEvoiRobot(Robot):
    def _decide_query(self, q_local):
        task_dist = self.task_dist / self.task_dist.sum()
        q_choice = q_local[:, None, :] - q_local[:, :, None]
        p_q_response = torch.sigmoid(q_choice * self.prec)
        p_response = torch.sum(task_dist[:, None, None] * p_q_response, dim=0)
        potential_task_dist1 = p_q_response * task_dist[:, None, None]
        potential_task_dist1 = potential_task_dist1 / potential_task_dist1.sum(dim=0, keepdim=True)
        potential_task_dist2 = (1 - p_q_response) * task_dist[:, None, None]
        potential_task_dist2 = potential_task_dist2 / potential_task_dist2.sum(dim=0, keepdim=True)
        potential_q1 = torch.sum(potential_task_dist1[:, :, :, None] * q_local[:, None, None, :], dim=0)
        potential_q2 = torch.sum(potential_task_dist2[:, :, :, None] * q_local[:, None, None, :], dim=0)
        potential_value1 = potential_q1.max(dim=-1).values
        potential_value2 = potential_q2.max(dim=-1).values
        average_value = p_response * potential_value1 + (1 - p_response) * potential_value2
        current_value = torch.sum(task_dist[:, None] * q_local, dim=0).max()
        query_values = average_value - current_value
        best_query_tmp = query_values.argmax(dim=0)
        best_query1 = query_values.gather(0, best_query_tmp[None, :]).argmax().item()
        best_query2 = best_query_tmp[best_query1].item()
        should_query = query_values[best_query1, best_query2] > self.query_param
        return should_query, [best_query1, best_query2]


class EvoiRobot(Robot):
    def _decide_query(self, q_local):
        task_dist = self.task_dist / self.task_dist.sum()
        p_q_response = torch.sigmoid((q_local[:, None, :] - q_local[:, :, None]) * self.prec)
        p_response = torch.sum(task_dist[:, None, None] * p_q_response, dim=0)
        potential_task_dist1 = p_q_response * task_dist[:, None, None]
        potential_task_dist1 = potential_task_dist1 / potential_task_dist1.sum(dim=0, keepdim=True)
        potential_task_dist2 = (1 - p_q_response) * task_dist[:, None, None]
        potential_task_dist2 = potential_task_dist2 / potential_task_dist2.sum(dim=0, keepdim=True)
        query_values = (p_response * torch.sum(potential_task_dist1[:, :, :, None] * q_local[:, None, None, :], dim=0).max(dim=-1).values + (1 - p_response) * torch.sum(potential_task_dist2[:, :, :, None] * q_local[:, None, None, :], dim=0).max(dim=-1).values) - torch.sum(task_dist[:, None] * q_local, dim=0).max()
        best_query_tmp = query_values.argmax(dim=0)
        best_query1 = query_values.gather(0, best_query_tmp[None, :]).argmax().item()
        best_query2 = best_query_tmp[best_query1].item()
        return query_values[best_query1, best_query2] > self.query_param, [best_query1, best_query2]

      
class ConstrainedEvoiRobot(Robot):
    def _decide_query(self, q_local):
        best_action = np.argsort(self.action_values)[-1]
        q_choice = q_local - q_local[:, best_action, None]
        q_diff = self.task_dist[:, None] * q_choice / self.task_dist.sum()
        p_change = torch.sigmoid(q_choice * self.prec)
        query_value = torch.sum(p_change * q_diff, dim=0)


def evaluate(param, env_class, robot, policies):
    scores = []
    nums_queries = []
    p_task = []
    for variant in range(env_class.variants):
        env = env_class(variant=variant)
        agent = robot(query_param=param, policies=policies, prec=args.prec)
        expert_policy=partial(policy_preference, Q=agent.get_policy_q(policies[variant]))
        state = env.reset()

        rendering = n_render is not None and variant % n_render == 0
        if rendering:
            env.render()
        score = 0
        total_queries = 0
        for t in range(MAX_T):
            should_query, action = agent.act(state)
            if should_query:
                response = expert_policy(state, *action)
                agent.learn(state, action, response)
                total_queries += 1
            else:
                state, reward, done, info = env.step(action)
                if rendering:
                    env.render()
                score += reward
                if done:
                    break
        p_task.append(np.max((agent.task_dist / agent.task_dist.sum()).detach().numpy()))
        scores.append(score)
        nums_queries.append(total_queries)
    return np.mean(nums_queries), np.mean(scores), np.mean(p_task)


def policy_preference(state, a, b, Q):
    values = Q(state).tolist()
    val_a, val_b = values[a], values[b]
    return np.argmax([val_a, val_b])




parser = argparse.ArgumentParser()
parser.add_argument("--nrender", type=int, default=None)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--parammin", type=float, default=1e-6)
parser.add_argument("--parammax", type=float, default=0.1)
parser.add_argument("--paramstep", type=float, default=0.2)
parser.add_argument("--output", type=str, default='results')
parser.add_argument("--env", type=str, default='empty')
parser.add_argument("--prec", type=float, default=50)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--method", type=str)
args = parser.parse_args()

env_list = dict(empty=EmptyGrid, rooms=RoomsGrid, maze=MazeGrid)
n_render = args.nrender
methods = dict(evoi=EvoiRobot, uncertainty=UncertainRobot, random=RandomRobot)


def run_task(method, param):
    robot = methods[method]
    return evaluate(param=param, env_class=env_class, robot=robot, policies=policies)[:2]

if __name__ == "__main__":
    params = np.exp(np.arange(np.log(args.parammin), np.log(args.parammax), np.log(1 + args.paramstep)))
    print(params)
    env_class = env_list[args.env]
    policies = [value_iteration.learn_q(env_class(variant=i), gamma=args.gamma) for i in tqdm.trange(env_class.variants)]

    results = {}
    results = list(starmap(run_task, tqdm.tqdm([(args.method, x) for x in params])))
    torch.save(results, args.output)


