#!/usr/bin/env python

import rospy
import os
import json
import numpy as np
import random
import time
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir+'/utils')
from environment_stage_1 import Env

import torch
import torch.nn.functional as F
import gc
import torch.nn as nn
import math
from collections import deque
import copy
from std_msgs.msg import Float32

#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))
#---Functions to make network updates---#

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.0 - tau)+ param.data*tau)

def hard_update(target,source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

#---Ornstein-Uhlenbeck Noise for action---#

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.01, decay_period= 600000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_noise(self, t=0): 
        ou_state = self.evolve_state()
        decaying = float(float(t)/ self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_state

#---Gaussian Noise for action---#

class GaussianNoise(object):
    def __init__(self, exploration_noise):
        self.noise_clip = noise_clip
    
    def get_noise(self, expl_noise):
        noise_action_1 = np.clip(np.random.normal(0, expl_noise, size=1), -self.noise_clip, self.noise_clip)
        noise_action_2 = np.clip(np.random.normal(0, expl_noise, size=1), -self.noise_clip, self.noise_clip)
        return noise_action_1, noise_action_2


#---Critic--#

EPS = 0.003
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1./np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v,v)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 125)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

        self.fa1 = nn.Linear(action_dim, 125)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)

        self.fca1 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)

        self.fca2 = nn.Linear(250, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)

    def forward(self, state, action):
        xs = torch.relu(self.fc1(state))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs,xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs

#---Actor---#

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        self.fa1 = nn.Linear(state_dim, 250)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)

        self.fa2 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)

        self.fa3 = nn.Linear(250, action_dim)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)

    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        action = self.fa3(x)
        if state.shape <= torch.Size([self.state_dim]):
            action[0] = torch.sigmoid(action[0])*self.action_limit_v
            action[1] = torch.tanh(action[1])*self.action_limit_w
        else:
            action[:,0] = torch.sigmoid(action[:,0])*self.action_limit_v
            action[:,1] = torch.tanh(action[:,1])*self.action_limit_w
        return action

#---Memory Buffer---#

class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])

        return s_array, a_array, r_array, new_s_array, done_array

    def len(self):
        return self.len

    def add(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.len += 1 
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)

#---TD3 Trainer---#

BATCH_SIZE = 256
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001  # TAU value for soft updates

class Trainer:
    
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w, ram):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        self.ram = ram

        self.train_cnt = 0

        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)

        self.critic1 = Critic(self.state_dim, self.action_dim)
        self.critic2 = Critic(self.state_dim, self.action_dim)
        self.target_critic1 = Critic(self.state_dim, self.action_dim)
        self.target_critic2 = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer1 = torch.optim.Adam(self.critic1.parameters(), LEARNING_RATE)
        self.critic_optimizer2 = torch.optim.Adam(self.critic2.parameters(), LEARNING_RATE)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic1, self.critic1)
        hard_update(self.target_critic2, self.critic2)
        
    def get_exploitation_action(self,state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        return action.data.numpy()
        
    def get_exploration_action(self, state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        new_action = action.data.numpy()
        return new_action
    
    def optimizer(self, cur_step):

        self.train_cnt += 1

        s_sample, a_sample, r_sample, new_s_sample, done_sample = ram.sample(BATCH_SIZE)
        
        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)
        done_sample = torch.from_numpy(done_sample)

        '''
        with torch.no_grad():
            noise = (torch.randn_like(a_sample) * 0.2).clamp(-0.5, 0.5)  # Adding clipped noise to target actions
            next_action = (self.target_actor(new_s_sample) + noise).clamp(-1.0, 1.0)
            target_Q1 = self.target_critic1(new_s_sample, next_action)
            target_Q2 = self.target_critic2(new_s_sample, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r_sample + (1 - done_sample) * GAMMA * target_Q
        '''

        new_a_sample = self.target_actor.forward(new_s_sample).detach()

        if use_noise == 'OU noise':
            tar_N = copy.deepcopy(noise.get_noise(t=cur_step))
            tar_N[0] = tar_N[0]*self.action_limit_v/2
            tar_N[1] = tar_N[1]*self.action_limit_w
            
        else: # Gaussian noise
            tar_N = [0, 0]
            tar_N[0], tar_N[1] = noise.get_noise(policy_noise)
        
        new_a_sample[0] = np.clip(new_a_sample[0] + tar_N[0], 0., self.action_limit_v)
        new_a_sample[1] = np.clip(new_a_sample[1] + tar_N[1], -self.action_limit_w, self.action_limit_w)

        target_Q1 = self.target_critic1.forward(new_s_sample, new_a_sample)
        target_Q2 = self.target_critic2.forward(new_s_sample, new_a_sample)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1 - done) * GAMMA * target_Q)

        current_Q1 = self.critic1(s_sample, a_sample)
        current_Q2 = self.critic2(s_sample, a_sample)

        critic_loss1 = F.mse_loss(current_Q1, target_Q)
        critic_loss2 = F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer1.zero_grad()
        critic_loss1.backward(retain_graph=True)
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        if self.train_cnt % 2 == 0:  # Delayed policy updates
            actor_grad = self.critic1(s_sample, self.actor.forward(s_sample))
            loss_actor = -actor_grad.mean() # -1*torch.sum(actor_grad)
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()

            soft_update(self.target_actor, self.actor, TAU)
            soft_update(self.target_critic1, self.critic1, TAU)
            soft_update(self.target_critic2, self.critic2, TAU)

    def save_models(self, episode):

        torch.save(self.target_actor.state_dict(), dirPath + '/Models/td3/hwasu_stage_3/new_result/' + str(episode) + '_actor.pt')

        torch.save(self.target_critic1.state_dict(), dirPath + '/Models/td3/hwasu_stage_3/new_result/' + str(episode) + '_critic1.pt')

        torch.save(self.target_critic2.state_dict(), dirPath + '/Models/td3/hwasu_stage_3/new_result/' + str(episode) + '_critic2.pt')
   
        print('***** stage 3 -> Episode: '+str(episode)+' Model save... *****')
        
        #print('****Models saved***')
        
    def load_models(self, episode):

        self.actor.load_state_dict(torch.load(dirPath + '/Models/td3/hwasu_stage_3/new_result/' + str(episode) + '_actor.pt'))

        self.critic1.load_state_dict(torch.load(dirPath + '/Models/td3/hwasu_stage_3/new_result/' + str(episode)+ '_critic1.pt'))

        self.critic2.load_state_dict(torch.load(dirPath + '/Models/td3/hwasu_stage_3/new_result/' + str(episode)+ '_critic2.pt'))

        print('***** stage 3 -> Episode: '+str(episode)+' Model load... *****')

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        hard_update(self.target_critic2, self.critic2)

        print('*** '+str(episode)+' Models load ***')


#---Save Reward list to Text file--#
def save_result(result_list, filename):
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode) as f:
        for item in result_list:
            f.write(str(item) + '\n')
    print("The result list has been successfully saved.")


#---Run agent---#

is_training = True

MAX_EPISODES = 5000
MAX_STEPS = 1000
MAX_BUFFER = 200000
reward_list = []
train_time_list = []

expl_noise = 1      # Initial exploration noise starting value in range [expl_min ... 1]
expl_min = 0.1
expl_decay_steps = 100000  # Number of steps over which the initial exploration noise will decay over
noise_clip = 0.5
            # Exploration noise after the decay in range [0...expl_noise]
policy_noise = 0.2

reward_path = dirPath + '/Models/td3/hwasu_stage_3/new_result/train_stage_3_reward_result.txt'
time_path = dirPath + '/Models/td3/hwasu_stage_3/new_result/train_stage_3_time_result.txt'

STATE_DIMENSION = 14
ACTION_DIMENSION = 2
ACTION_V_MAX = 0.22  # m/s
ACTION_W_MAX = 2.0  # rad/s

if is_training:
    var_v = ACTION_V_MAX * 0.5
    var_w = ACTION_W_MAX * 2 * 0.5
else:
    var_v = ACTION_V_MAX * 0.10
    var_w = ACTION_W_MAX * 0.10

print('State Dimensions: ' + str(STATE_DIMENSION))
print('Action Dimensions: ' + str(ACTION_DIMENSION))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')
ram = MemoryBuffer(MAX_BUFFER)
trainer = Trainer(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, ram)

use_noise = 'OU noise' # 'Gaussian noise'
if use_noise == 'OU noise':
    noise = OUNoise(ACTION_DIMENSION, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)
else:
    noise = GaussianNoise(noise_clip)

load_model_episode = 0
#trainer.load_models(load_model_episode)

if __name__ == '__main__':
    rospy.init_node('td3_stage_3')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env(action_dim=ACTION_DIMENSION, train=is_training)
    before_training = 4
    past_action = np.zeros(ACTION_DIMENSION)
    total_start_time = time.time()
    reached_goal_flag_cnt = 0

    for ep in range(load_model_episode + 1, MAX_EPISODES + 1):
        episode_start_time = time.time()
        done = False
        state = env.reset()
        reward_episode = 0.0

        if is_training and not ep % 10 == 0 and ram.len >= before_training * MAX_STEPS:
            print('---------------------------------')
            print('Episode: ' + str(ep) + ' training')
            print('---------------------------------')
        else:
            if ram.len >= before_training * MAX_STEPS:
                print('---------------------------------')
                print('Episode: ' + str(ep) + ' evaluating')
                print('---------------------------------')
            else:
                print('---------------------------------')
                print('Episode: ' + str(ep) + ' adding to memory')
                print('---------------------------------')

        for step in range(1, MAX_STEPS+1):
            state = np.float32(state)

            if is_training and not ep % 10 == 0:
                action = trainer.get_exploration_action(state)
                if use_noise == 'OU noise':
                    N = copy.deepcopy(noise.get_noise(t=step))
                    N[0] = N[0]*ACTION_V_MAX/2
                    N[1] = N[1]*ACTION_W_MAX
                    action[0] = np.clip(action[0] + N[0], 0., ACTION_V_MAX)
                    action[1] = np.clip(action[1] + N[1], -ACTION_W_MAX, ACTION_W_MAX)
                        
                        #if check_step < exploitation_step:
                else: # Gaussian noise
                    if expl_noise > expl_min:
                        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)
                    noise_action_1, noise_action_2 = noise.get_noise(expl_noise)
                    action[0] = np.clip(action[0] + noise_action_1, 0., ACTION_V_MAX)
                    action[1] = np.clip(action[1] + noise_action_2, -ACTION_W_MAX, ACTION_W_MAX)
            else:
                action = trainer.get_exploration_action(state)

            if not is_training:
                action = trainer.get_exploitation_action(state)

            next_state, reward, done, reached_goal_flag = env.step(action, past_action)
            if reached_goal_flag == True:
                reached_goal_flag_cnt += 1

            reward_episode += reward
            past_action = copy.deepcopy(action)
            next_state = np.float32(next_state)
            if not ep % 10 == 0 or not ram.len >= before_training * MAX_STEPS:
                if reward == 100.0:
                    print('***\n-------- Maximum Reward ----------\n****')
                    for _ in range(3):
                        ram.add(state, action, reward, next_state, done)
                else:
                    ram.add(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)

            if ram.len >= before_training * MAX_STEPS and is_training and not ep % 10 == 0:
                trainer.optimizer(step)

            if done or step == MAX_STEPS or reached_goal_flag_cnt == 10:
                print('reward per ep: ' + str(reward_episode))
                print('*\nbreak step: ' + str(step) + '\n*')
                #print('sigma: ' + str(noise.sigma))
                if not ep % 10 == 0:
                    pass
                else:
                    result = reward_episode
                    pub_result.publish(result)

                episode_end_time = time.time()
                episode_1_train_time = round(episode_end_time - episode_start_time, 2)

                save_result([reward_episode], reward_path)
                save_result([episode_1_train_time], time_path)

                reached_goal_flag_cnt = 0
                break
        
        if ep % 10 == 0:
            trainer.save_models(ep)

    total_end_time = time.time()
    total_train_time = round(total_end_time - total_start_time, 2)

    print("total time: %f [s]" % total_train_time)

print('Completed Training')

