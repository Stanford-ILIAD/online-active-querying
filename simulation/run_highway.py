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

device = torch.device("cpu")
MAX_T = 500


def load_policy(f):
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            return DQN.load(f)

class Robot(abc.ABC):
    def __init__(self, query_param, n_policies=313, prec=10, device=device):
        # prec is expert precision
        self.device = torch.device(device)
        self.query_param = query_param
        self.prec = prec
        self.n_policies = n_policies
        self.descriptions = {0: 'LEFT', 1: 'IDLE', 2: 'RIGHT', 3: 'FASTER', 4: 'SLOWER'}
        self.reset()
        self.policies = []
        for loc in sorted(os.listdir('policies')):
            if loc.startswith('.'):
                continue
            policy = self.load_policy(f'policies/{loc}/dqn_network.zip')
            self.policies.append(policy)
            if len(self.policies) == n_policies:
                break
        else:
            raise RuntimeError(f'Insufficient policies found ({len(self.policies)})')

    def reset(self):
        self.task_dist = torch.ones(self.n_policies).to(self.device)

    def load_policy(self, f):
        devnull = open(os.devnull, 'w') 
        with contextlib.redirect_stderr(devnull):
            return DQN.load(f)

    def get_policy_q(self, policy):
        def Q(x):
            x = torch.tensor(x).to(self.device)
            x = policy.q_net(x.view(*x.shape[:-2], -1).unsqueeze(0))[0]
            return x
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
        print('Best action:', best_action)
        print('Query values:\n', query_values)
        print('Best query:', [best_query1, best_query2])
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


def evaluate(n_episodes, param):
    """Deep Q-Learning.
    """
    scores = []
    nums_queries = []
    p_task = []
    for i_episode in range(1, n_episodes+1):
        print('Episode', i_episode)
        task_idx = random.randrange(len(eval_policies))
        env = eval_envs[task_idx]
        policy_params = eval_policies[task_idx]
        expert_policy=partial(policy_preference, Q=get_policy_q(load_policy(policy_params)))
        state = env.reset()
        agent = methods[args.method](query_param=param)
        rendering = n_render is not None and i_episode % n_render == 0
        if rendering:
            env.render()
        score = 0
        total_queries = 0
        for t in range(MAX_T):
            should_query, action = agent.act(state)
            if should_query:
                print('getting pref ', action)
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
        print('len', t)
        p_task.append(np.max((agent.task_dist / agent.task_dist.sum()).detach().numpy()))
        scores.append(score)
        nums_queries.append(total_queries)
        print((nums_queries, scores, p_task))
        torch.save((nums_queries, scores, p_task), args.output)
    return nums_queries, scores, p_task


def policy_preference(state, a, b, Q):
    values = Q(state).tolist()
    val_a, val_b = values[a], values[b]
    return np.argmax([val_a, val_b])


def run_evaluation(seed, param):
    random.seed(seed)
    return evaluate(n_episodes=args.nval, param=param) 


def get_policy_q(policy):
    def Q(x):
        x = torch.tensor(x).to(device)
        x = x.view(*x.shape[:-2], -1)
        shrink = False
        if x.dim() < 2:
            x = x.unsqueeze(0)
            shrink = True
        x = policy.q_net(x)
        if shrink:
            x = x[0]
        return x
    return Q


parser = argparse.ArgumentParser()
parser.add_argument("--load", default=False, action='store_true')
parser.add_argument("--npol", type=int, default=75)
parser.add_argument("--nrender", type=int, default=None)
parser.add_argument("--nval", type=int, default=50)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--acc", type=float, default=0.1)
parser.add_argument("--parammin", type=float, default=1e-6)
parser.add_argument("--param", type=float, default=1e-3)
parser.add_argument("--output", type=str, default='results')
parser.add_argument("--method", type=str, default='evoi')
parser.add_argument("--policies", type=str, default='policies')
args = parser.parse_args()

n_render = args.nrender

policy_dirs = os.listdir(args.policies)
methods = dict(evoi=EvoiRobot, uncertainty=UncertainRobot, random=RandomRobot)

train_policies = []
eval_policies = []
eval_envs = []
state_size = 25

for i, loc in tqdm.tqdm(list(enumerate(x for x in policy_dirs if not x.startswith('.')))):
    policy_params = f'{args.policies}/{loc}/dqn_network.zip'
    task_params = f'{args.policies}/{loc}/task_params.pkl'
    env = gym.make("customhighway-v0")
    with open(task_params, 'rb') as f:
        env.config.update(pickle.load(f))
    env.reset()
    if i < args.npol:
        train_policies.append(policy_params)
    else:
        eval_policies.append(policy_params)
        eval_envs.append(env)


if __name__ == "__main__":
    run_evaluation(args.seed, args.param)

