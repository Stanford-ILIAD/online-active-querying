import gym
import highway_env
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
import random
import torch
from stable_baselines3 import DQN
import pickle
import abc
import contextlib

FPS = 15
COUNTDOWN_SECONDS = 5
NUM_EPISODES = 5

font0 = ImageFont.truetype('fonts/palatino-linotype.ttf', size=30)
font1 = ImageFont.truetype('fonts/palatino-linotype.ttf', size=15)
font2 = ImageFont.truetype('fonts/palatino-linotype.ttf', size=90)


def addInformationToImage(img, time_left, vehicle_speed):
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), 'Speed: ' + "{:2.1f}".format(float(vehicle_speed)), fill='rgb(180, 180, 180)', font=font1)
    #draw.text((10, 550), 'Time Left: ' + "{:2d}".format(int(time_left)), fill='rgb(180, 180, 180)', font=font1)
    return img

def addHeadingToImage(img, text):
    draw = ImageDraw.Draw(img)
    draw.text((75, 75), text, fill='rgb(255, 255, 255)', font=font0)
    return img

def addTextToImage(img, text):
    draw = ImageDraw.Draw(img)
    draw.text((100, 130), text, fill='rgb(255, 255, 255)', font=font2)
    return img
    
    
    
class Robot(abc.ABC):
    def __init__(self, query_param, n_policies=313, prec=10, device='cuda'):
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
        best_query = query_value.argmax().item()
        should_query = query_value[best_query] > self.query_param
        #print(f'Should query: {should_query.item()}')
        #print('Query values: ' + str({self.descriptions[x]: query_value[x].item() for x in self.descriptions}))
        return should_query, [best_action, best_query]


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


def is_identical_question(arr1, arr2):
    return (arr1[0] == arr2[0] and arr1[1] == arr2[1]) or (arr1[0] == arr2[1] and arr1[1] == arr2[0])


class Experiment:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.env = gym.make("customhighway-v0")
        self.env.seed(1)
        #self.robot = EvoiRobot(0.05)
        self.robot = RandomRobot(0.2)
        #self.robot = UncertainRobot(46)
        
        self.obs_data = [[] for _ in range(NUM_EPISODES)]
        self.act_data = [[] for _ in range(NUM_EPISODES)]
        self.rew_data = [[] for _ in range(NUM_EPISODES)]
        self.que_data = [[] for _ in range(NUM_EPISODES)]
        self.ans_data = [[] for _ in range(NUM_EPISODES)]
        self.dst_data = [[] for _ in range(NUM_EPISODES)]
        self.spe_data = [[] for _ in range(NUM_EPISODES)]
        
        self.alreadyCompleted = False
        
    def reset(self):
        self.counter = int(COUNTDOWN_SECONDS/(1./FPS))
        self.episode_done = True
        self.episode_no = -1
        self.robot.reset()
        self.save()
        return self.step(0)

    @property
    def alreadyStarted(self):
        return self.episode_no >= 0
        
    def step(self, human_action):
        ep_no = self.episode_no
        rew = 0.
        
        if not self.episode_done:  
            if human_action > -0.5: # the user sent some response
                self.ans_data[ep_no].append(human_action)
                self.robot.learn(self.obs_data[ep_no][-1], self.que_data[ep_no][-1], human_action)
                self.dst_data[ep_no].append(self.robot.task_dist.cpu().numpy())
            elif len(self.que_data[ep_no]) > 0 and len(self.que_data[ep_no][-1]) > 0:
                # the python script is lagging behind, so we need to show the observation again -- corner case
                img = Image.fromarray(self.env.render(mode="rgb_array")).rotate(90, expand=True)
                return addInformationToImage(img, self.episode_t/float(FPS), str(self.env.vehicle.speed)), 0, False, []
            if self.episode_t % 15 == 0: # time to evaluate the policy
                ask_question, action = self.robot.act(self.obs_data[ep_no][-1])
                while ask_question and len(self.que_data[ep_no]) > 0 and len(self.que_data[ep_no][-1]) > 0 and is_identical_question(action, self.que_data[ep_no][-1]):
                    # We are asking the same question again, so let's just take the human's response from the previous step.
                    self.que_data[ep_no].append(self.que_data[ep_no][-1])
                    self.ans_data[ep_no].append(self.ans_data[ep_no][-1])
                    self.robot.learn(self.obs_data[ep_no][-1], self.que_data[ep_no][-1], self.ans_data[ep_no][-1])
                    self.dst_data[ep_no].append(self.robot.task_dist.cpu().numpy())
                    ask_question, action = self.robot.act(self.obs_data[ep_no][-1])
                    print('skipped')
            else: # we still need to execute the previous action (each action takes 15 time steps)
                ask_question = False
                action = 1 # idle action is 1
            if not ask_question:
                self.que_data[ep_no].append([])
                self.act_data[ep_no].append(action)
                obs, rew, done, _ = self.env.step(action)
                self.rew_data[ep_no].append(rew)
                self.obs_data[ep_no].append(obs)
                self.spe_data[ep_no].append(self.env.vehicle.speed)
                self.episode_done = done
                if done:
                    self.counter = int(COUNTDOWN_SECONDS/(1./FPS))
                img = Image.fromarray(self.env.render(mode="rgb_array")).rotate(90, expand=True)
                self.episode_t -= 1
                return addInformationToImage(img, self.episode_t/float(FPS), str(self.env.vehicle.speed)), rew, done, []
            else:
                self.que_data[ep_no].append(action)
                img = Image.fromarray(self.env.render(mode="rgb_array")).rotate(90, expand=True)
                option_descriptions = [self.robot.descriptions[action[0]], self.robot.descriptions[action[1]]]
                return addInformationToImage(img, self.episode_t/float(FPS), str(self.env.vehicle.speed)), 0, False, option_descriptions
                
        else:
            if self.counter == int(COUNTDOWN_SECONDS/(1./FPS)) and self.episode_no == NUM_EPISODES - 1:
                # the entire experiment has finished
                self.alreadyCompleted = True
                self.save()
                return Image.open('experimentover.png'), rew, False, []
            else:
                # countdown continues
                if self.counter == int(COUNTDOWN_SECONDS/(1./FPS)):
                    # reset the environment
                    self.episode_no += 1
                    self.episode_t = 600
                    self.env.seed(self.episode_no + 1)
                    self.obs_data[self.episode_no].append(self.env.reset())
                    self.robot.reset()
                self.counter -= 1
                if self.counter == 0:
                    self.episode_done = False
                img = Image.fromarray(self.env.render(mode="rgb_array")).rotate(90, expand=True)
                sec_remaining = self.counter // FPS + 1
                img = addHeadingToImage(img, str('Round ' + str(self.episode_no+1)))
                img = addTextToImage(img, str(int(sec_remaining)))
                return img, rew, False, []

    def save(self):
        with open('results/e3_' + self.experiment_name + '.pkl', 'wb') as f:
            pickle.dump({'obs_data': self.obs_data, 'act_data': self.act_data, 'rew_data': self.rew_data, 'que_data': self.que_data, 'ans_data': self.ans_data, 'dst_data': self.dst_data, 'spe_data': self.spe_data, 'alreadyCompleted': self.alreadyCompleted}, f)
