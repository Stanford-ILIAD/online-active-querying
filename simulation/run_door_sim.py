import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import tqdm
import os
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
import re
import make_env


device = torch.device("cpu")




class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, policies, state_size, action_space, seed, query_param, expert_acc, method):
        """Initialize an Agent object.
        """
        self.method = method
        self.state_size = state_size
        self.seed = random.seed(seed)
        self.query_param = query_param
        self.expert_acc = expert_acc
        self.action_space, = action_space.shape

        self.policies = [get_policy_action(x) for x in policies]
        self.Q = [get_policy_q(x) for x in policies]
        self.t_step = 0

    def act(self, state, task_dist, expert_policy=None):
        task_dist = torch.nan_to_num(torch.tensor(task_dist).to(device).detach())
        state = torch.tensor(state).float()

        # Greedy action selection
        sampled_policy_idx = torch.multinomial(task_dist / task_dist.sum(), 1).item()
        best_action = self.policies[sampled_policy_idx](state)
        queried = False
        q_local = torch.stack([x(state, best_action) for x in self.Q]).detach()

        if expert_policy is not None:

            if self.method == 'evoi':
                def query_value(other_action):
                    q_other = torch.stack([x(state, other_action) for x in self.Q])
                    return torch.clamp(
                        task_dist * (q_other - q_local.detach()),
                        min=0.,
                    ).mean()
                best_query = torch.zeros(self.action_space, requires_grad=True)
                optimizer = optim.LBFGS([best_query])
                for i in range(5):
                    optimizer.zero_grad()
                    objective = -query_value(best_query)
                    objective.backward()
                    optimizer.step(lambda: -query_value(best_query))
                best_query = best_query.detach()
                should_query = query_value(best_query) > self.query_param
            elif self.method == 'evoi_random':
                trials = 100
                def query_value(other_action):
                    q_other = torch.stack([x(state[None, :].expand(trials, -1), other_action) for x in self.Q])
                    return torch.clamp(
                            task_dist[:, None] * (q_other - q_local[:, None].detach()),
                        min=0.,
                    ).mean(dim=0)
                queries = torch.stack([self.policies[torch.multinomial(task_dist / task_dist.sum(), 1).item()](state) for _ in range(trials)])
                values = query_value(queries)
                best_query = queries[values.argmax().item()].detach()
                should_query = values.max().item() > self.query_param
            elif self.method == 'random':
                tmp_dist = task_dist.clone()
                tmp_dist[sampled_policy_idx] = 0.
                sampled_policy_idx = torch.multinomial(tmp_dist / tmp_dist.sum(), 1).item()
                best_query = self.policies[sampled_policy_idx](state)
                should_query = random.random() < self.query_param


            if should_query:
                query_response = expert_policy(state, best_action, best_query)
                q_query = torch.stack([x(state, best_query) for x in self.Q]).detach()
                q_choice = q_query - q_local
                scaled_probs = torch.sigmoid(q_choice / self.expert_acc)
                best_action = [best_action, best_query][query_response]
                if query_response == 1:
                    task_dist = scaled_probs * task_dist
                else:
                    task_dist = (1 - scaled_probs) * task_dist
                queried = True

        return best_action.detach().numpy(), queried, task_dist.cpu().numpy()


def evaluate(agent, env, n_episodes=10, expert_policy=None):
    """Deep Q-Learning.
    """
    scores = []
    nums_queries = []
    for i_episode in range(1, n_episodes+1):
        rendering = n_render is not None and i_episode % n_render == 0
        state = env.reset()
        task_dist = np.ones([len(train_policies)])
        if rendering:
            env.render()
        score = 0
        total_queries = 0
        for t in range(500):
            action, queried, task_dist = agent.act(state, task_dist, expert_policy=expert_policy)
            next_state, reward, done, _ = env.step(action)
            if rendering:
                env.render()
            state = next_state
            score += reward
            total_queries += queried
            if done:
                break
        print(t)
        print(total_queries)
        scores.append(score)
        nums_queries.append(total_queries)
    return np.mean(nums_queries), np.mean(scores)


def expert_policy(state, a, b, Q):
    val_a, val_b = Q(state, a).item(), Q(state, b).item()
    return np.argmax([val_a, val_b])


def run_evaluation(seed, param):
    random.seed(seed)
    task_idx = random.randrange(len(eval_policies))
    env = make_env.make_door(*eval_envs[task_idx])
    policy_params = eval_policies[task_idx]
    agent = Agent(policies=train_policies, 
            state_size=state_size, action_space=env.action_space, seed=seed, 
            query_param=param, expert_acc=args.acc, method=args.method)
    return evaluate(agent, env=env, n_episodes=args.nval, expert_policy=partial(expert_policy, Q=get_policy_q(policy_params))) 

def get_policy_action(policy):
    data = torch.load(f'{policy}/params.pkl')
    policy = data['evaluation/policy']

    def pi(x):
        x = torch.tensor(x).to(device)
        return policy(x).sample()
    return pi

def get_policy_q(policy):
    data = torch.load(f'{policy}/params.pkl')
    qf1 = data['trainer/qf1']
    qf2 = data['trainer/qf2']

    def Q(s, a):
        if a.dim() == 1:
            s, a = s.unsqueeze(0), a.unsqueeze(0)
            return torch.minimum(qf1(s, a), qf2(s, a)).sum()
        elif a.dim() == 2:
            return torch.minimum(qf1(s, a), qf2(s, a)).sum(dim=-1)
    return Q


parser = argparse.ArgumentParser()
parser.add_argument("--load", default=False, action='store_true')
parser.add_argument("--npol", type=int, default=75)
parser.add_argument("--nrender", type=int, default=None)
parser.add_argument("--nval", type=int, default=50)
parser.add_argument("--seed", type=int, default=1000)
parser.add_argument("--acc", type=float, default=0.1)
parser.add_argument("--niter", type=int, default=150)
parser.add_argument("--parammin", type=float, default=1e-6)
parser.add_argument("--parammax", type=float, default=1e-1)
parser.add_argument("--paramstep", type=float, default=0.3)
parser.add_argument("--paramsample", type=int, default=None)
parser.add_argument("--output", type=str, default='results')
parser.add_argument("--method", type=str, default='evoi')
parser.add_argument("--policies", type=str, default='policies')
args = parser.parse_args()

n_render = args.nrender

policy_dirs = os.listdir(args.policies)

train_policies = []
eval_policies = []
eval_envs = []
state_size = 25

for i, loc in tqdm.tqdm(list(enumerate(x for x in policy_dirs if x.startswith('door')))):
    policy_params = f'{args.policies}/{loc}'
    if 'params.pkl' not in os.listdir(policy_params):
        continue
    get_policy_q(policy_params)
    env = [float(x) for x in re.findall(r'-?([0-9.])+', loc)]
    if i < args.npol:
        train_policies.append(policy_params)
    else:
        eval_policies.append(policy_params)
        eval_envs.append(env)


if __name__ == "__main__":
    params = np.exp(np.arange(np.log(args.parammin), np.log(args.parammax), np.log(1 + args.paramstep)))
    if args.paramsample is not None:
        params = [
            (param, *np.exp(np.random.uniform(np.log(args.parammin), np.log(args.parammax), size=args.paramsample))) 
            for param in params
        ]
    print('Will compute frontier with params:', params)
    frontiers = []
    for itr in range(args.niter):
        print('Iteration', itr + 1)
        seed = itr + args.seed
        #pool = multiprocessing.Pool()
        queries, scores = zip(*starmap(run_evaluation, [(seed, x) for x in params]))
        frontiers.append((np.array(queries), np.array(scores)))
        torch.save(frontiers, args.output)

