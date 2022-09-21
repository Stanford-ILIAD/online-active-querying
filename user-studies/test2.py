import gym
import highway_env
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import random
import torch
import torch.nn.functional as F
from stable_baselines3 import DQN
import pickle
import abc
import contextlib

NUM_EPISODES = 100

class Robot(abc.ABC):
    def __init__(self, query_param, n_policies=313, prec=10, device='cpu'):
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
            #query = [best_action, best_query]
            self.option0, self.option1 = best_query
            random.shuffle(best_query)
            return True, best_query
        else:
            return False, best_action
    
    def learn(self, obs, question, answer):
        # we asked the user question (an array of size 2, each entry is an action), and they chose answer=0 or answer=1 at state obs
        query_response = question[answer]
        q_choice = self.q_local[:, self.option1] - self.q_local[:, self.option0]
        scaled_probs = torch.sigmoid(q_choice * self.prec)
        best_action = query_response
        if query_response == self.option1:
            self.task_dist = scaled_probs * self.task_dist
        else:
            self.task_dist = (1 - scaled_probs) * self.task_dist


class EvoiRobot(Robot):
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




policies = []
for loc in sorted(os.listdir('policies')):
    if loc.startswith('.'):
        continue
    
    devnull = open(os.devnull, 'w') 
    with contextlib.redirect_stderr(devnull):
        policy = DQN.load(f'policies/{loc}/dqn_network.zip')
    policies.append(policy)
    if len(policies) == 313:
        break
else:
    raise RuntimeError(f'Insufficient policies found ({len(self.policies)})')




def get_policy_q(policy):
    def Q(x):
        x = torch.tensor(x).to(torch.device('cpu'))
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










for kkk in range(1):
    kk = 0.06
    num_questions = []
    num_steps = []
    env = gym.make("customhighway-v0")
    robot = EvoiRobot(kk)

    for i in range(NUM_EPISODES):
        #print(np.mean([num_questions[i]/num_steps[i] for i in range(len(num_questions))]))
        idx = int(np.random.randint(len(policies)))
        robot.reset()
        obs = env.reset()
        num_steps.append(0)
        num_questions.append(0)
        done = False
        while not done:
            q_vals = get_policy_q(policies[idx])(obs)
            ask_question, action = robot.act(obs)
            if ask_question:
                probs = F.softmax(torch.Tensor([robot.prec * q_vals[action[0]], robot.prec * q_vals[action[1]]]), dim=0).numpy()
                probs = probs / probs.sum()
                answer = np.random.choice(2, p=probs)
                robot.learn(obs, action, answer)
                num_questions[-1] += 1
                #print(num_questions[-1])
            else:
                obs, rew, done, _ = env.step(action)
                num_steps[-1] += 1
                for j in range(14):
                    obs, rew, done, _ = env.step(1)
        #print(np.mean([num_questions[i]/num_steps[i] for i in range(len(num_questions))]))
        
    print(kk)
    print(np.mean([num_questions[i]/num_steps[i] for i in range(NUM_EPISODES)]))