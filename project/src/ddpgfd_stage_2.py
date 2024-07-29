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

from per import PrioritizedReplayBuffer, NStepBackup
import per_utils as utils
import pickle, joblib


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
        # print('noise' + str(ou_state))
        decaying = float(float(t)/ self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_state

#---Critic--#

EPS = 0.003
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1./np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v,v)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.state_dim = state_dim # = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, 125)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.fa1 = nn.Linear(action_dim, 125)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fca1 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)
        # self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        
        self.fca2 = nn.Linear(250, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)
        # self.fca2.weight.data.uniform_(-EPS, EPS)
        
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
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        
        self.fa1 = nn.Linear(state_dim, 250)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fa2 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)
        # self.fa2.weight.data = fanin_init(self.fa2.weight.data.size())
        
        self.fa3 = nn.Linear(250, action_dim)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)
        # self.fa3.weight.data.uniform_(-EPS,EPS)
        
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


class Trainer:
    
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w, ram):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        #print('w',self.action_limit_w)
        self.ram = ram
        #self.iter = 0 
        
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)
        
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)
        self.pub_qvalue = rospy.Publisher('qvalue', Float32, queue_size=5)
        self.qvalue = Float32()
        
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.backup = NStepBackup(gamma=0.99, n_step=5)
        self.memory = PrioritizedReplayBuffer(size=MAX_BUFFER, seed=1000, alpha=0.5,
                                              beta_init=1.0, beta_inc_n=100)
        
    def get_exploitation_action(self,state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        #print('actionploi', action)
        return action.data.numpy()
        
    def get_exploration_action(self, state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        #noise = self.noise.sample()
        #print('noisea', noise)
        #noise[0] = noise[0]*self.action_limit_v
        #noise[1] = noise[1]*self.action_limit_w
        #print('noise', noise)
        new_action = action.data.numpy() #+ noise
        #print('action_no', new_action)
        return new_action
    
    def optimizer(self):
        (s_sample, a_sample, r_sample, new_s_sample, gamma_sample, flag_sample), weights, batch_idxes = ram.sample(BATCH_SIZE)
        
        #s_sample = torch.from_numpy(s_sample)
        #a_sample = torch.from_numpy(a_sample)
        #r_sample = torch.from_numpy(r_sample)
        #new_s_sample = torch.from_numpy(new_s_sample)
        #gamma_sample = torch.from_numpy(gamma_sample)
        weights = torch.from_numpy(weights.reshape(-1, 1)).float()
        #done_sample = torch.from_numpy(done_sample)
        
        with torch.no_grad():
            action_tgt = self.target_actor.forward(s_sample).detach()
            y_tgt = r_sample + gamma_sample * torch.squeeze(self.target_critic.forward(s_sample, action_tgt).detach())

        # Critic loss
        self.critic_optimizer.zero_grad()
        Q_b = self.critic.forward(s_sample, a_sample)
        loss_critic = F.smooth_l1_loss(torch.squeeze(Q_b), y_tgt)
        
        # Record Demo count
        #d_flags = torch.from_numpy(flag_sample)
        d_flags = flag_sample
        #demo_select = d_flags == DATA_DEMO
        #N_act = demo_select.sum().item()
        #demo_cnt.append(N_act)
        loss_critic.backward()
        self.critic_optimizer.step()

        # Actor loss
        self.actor_optimizer.zero_grad()
        action_b = self.actor.forward(s_sample)
        Q_act = self.critic.forward(s_sample, action_b)
        loss_actor = -torch.mean(Q_act)
        loss_actor.backward()
        self.actor_optimizer.step()

        """

        #-------------- optimize critic
        
        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
        y_expected = r_sample + (1 - done_sample)*GAMMA*next_value
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))

        td_errors = y_expected - y_predicted

        #----------------------------
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)

        
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        #------------ optimize actor
        pred_a_sample = self.actor.forward(s_sample)
        loss_actor = -1*torch.sum(self.critic.forward(s_sample, pred_a_sample))
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        """
        priority = ((Q_b.detach() - y_tgt.view(-1,1)).pow(2) + Q_act.detach().pow(
                        2)).cpu().numpy().ravel() + const_min_priority
        priority[flag_sample == 0] += const_demo_priority
        ram.update_priorities(batch_idxes, priority)

        #soft_update(self.target_actor, self.actor, TAU)
        #soft_update(self.target_critic, self.critic, TAU)


    def pretrain(self):

        update_step = 60000      #4000   #10000
        pretrain_step = 300000   #20000  #50000
        cnt = 0
        for step in np.arange(update_step, pretrain_step + 1, update_step): # total 5 step

            #if ram.ready():
            if pre_ram.len > 0:
                for _ in range(update_step): # total 2000 step
                    cnt += 1
                    print(cnt)

                    s_sample, a_sample, r_sample, new_s_sample, done_sample = pre_ram.sample(BATCH_SIZE)

                    s_sample = torch.from_numpy(s_sample)
                    a_sample = torch.from_numpy(a_sample)
                    r_sample = torch.from_numpy(r_sample)
                    new_s_sample = torch.from_numpy(new_s_sample)
                    done_sample = torch.from_numpy(done_sample)
        
                    #-------------- optimize critic
        
                    a_target = self.target_actor.forward(new_s_sample).detach()
                    next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
                    # y_exp = r _ gamma*Q'(s', P'(s'))
                    y_expected = r_sample + (1 - done_sample)*GAMMA*next_value
                    # y_pred = Q(s,a)
                    y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))
                    #----------------------------
                    loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
 
                    self.critic_optimizer.zero_grad()
                    loss_critic.backward()
                    self.critic_optimizer.step()
        
                    #------------ optimize actor
                    pred_a_sample = self.actor.forward(s_sample)
                    loss_actor = -1*torch.sum(self.critic.forward(s_sample, pred_a_sample))
        
                    self.actor_optimizer.zero_grad()
                    loss_actor.backward()
                    self.actor_optimizer.step()
        
                    soft_update(self.target_actor, self.actor, TAU)
                    soft_update(self.target_critic, self.critic, TAU)
                    """
                    (batch_s, batch_a, batch_r, batch_s2, batch_gamma, batch_flags), weights, idxes = ram.sample(BATCH_SIZE)
                    

                    weights = torch.from_numpy(weights.reshape(-1, 1)).float()

                    with torch.no_grad():
                        action_tgt = self.target_actor.forward(batch_s).detach()
                        y_tgt = batch_r + batch_gamma * torch.squeeze(self.target_critic.forward(batch_s, action_tgt).detach())

                    #self.agent.zero_grad()
                    # Critic loss
                    self.critic_optimizer.zero_grad()
                    Q_b = self.critic.forward(batch_s, batch_a)
                    loss_critic = F.smooth_l1_loss(torch.squeeze(Q_b), y_tgt)
                    #Q_b = self.critic(torch.cat((batch_s, batch_a), dim=1))
                    #loss_critic = (self.q_criterion(Q_b, y_tgt) * weights).mean()
                    
                    # Record Demo count
                    #d_flags = torch.from_numpy(batch_flags)
                    d_flags = batch_flags
                    #demo_select = d_flags == DATA_DEMO
                    #N_act = demo_select.sum().item()
                    #demo_cnt.append(N_act)
                    loss_critic.backward()
                    self.critic_optimizer.step()

                    # Actor loss
                    self.actor_optimizer.zero_grad()
                    action_b = self.actor.forward(batch_s)
                    Q_act = self.critic.forward(batch_s, action_b)
                    loss_actor = -torch.mean(Q_act)
                    loss_actor.backward()
                    self.actor_optimizer.step()
                    
                    #print(Q_b.shape)
                    #print(Q_act.shape)
                    #print(y_tgt.shape)

                    priority = ((Q_b.detach() - y_tgt.view(-1,1)).pow(2) + Q_act.detach().pow(
                        2)).cpu().numpy().ravel() + const_min_priority
                    #print(priority.shape)
                    priority[batch_flags == 0] += const_demo_priority

                    ram.update_priorities(idxes, priority)
                    
                    #soft_update(self.target_actor, self.actor, TAU)
                    #soft_update(self.target_critic, self.critic, TAU)
                    """
        self.save_models(0) # pretrain 300,000 steps

    def save_models(self, episode):

        torch.save(self.target_actor.state_dict(), dirPath + '/Models/ddpgfd/hwasu_stage_2/new_result/' + str(episode) + '_actor.pt')

        torch.save(self.target_critic.state_dict(), dirPath + '/Models/ddpgfd/hwasu_stage_2/new_result/' + str(episode) + '_critic.pt')
            
        print('***** stage 2 -> Episode: '+str(episode)+' Model save... *****')
        
        #print('****Models saved***')
        
    def load_models(self, episode):

        self.actor.load_state_dict(torch.load(dirPath + '/Models/ddpgfd/hwasu_stage_2/new_result/' + str(episode) + '_actor.pt'))

        self.critic.load_state_dict(torch.load(dirPath + '/Models/ddpgfd/hwasu_stage_2/new_result/' + str(episode)+ '_critic.pt'))

        #self.actor.load_state_dict(torch.load(dirPath + '/Models/ddpg/hwasu_stage_1/new_result/' + str(episode) + '_actor.pt'))

        #self.critic.load_state_dict(torch.load(dirPath + '/Models/ddpg/hwasu_stage_1/new_result/' + str(episode)+ '_critic.pt'))

        print('***** stage 2 -> Episode: '+str(episode)+' Model load... *****')

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('*** '+str(episode)+' Models load ***')

    def episode_reset(self):
        self.backup.reset()

    def add_n_step_experience(self, data_flag=1, done=False):  # Pop (s,a,r,s2) pairs from N-step backup to PER
        while backup.available(done):
            #success, exp = backup.pop_exp(done)
            experience = backup.pop_exp(done)
            success = experience[-1]
            if success:
                #print(experience)
                ram.add(experience)
        #if done:
        #    self.logger.debug('Done: All experience added')


#---Mish Activation Function---#
def mish(x):
    '''
        Mish: A Self Regularized Non-Monotonic Neural Activation Function
        https://arxiv.org/abs/1908.08681v1
        implemented for PyTorch / FastAI by lessw2020
        https://github.com/lessw2020/mish
        param:
            x: output of a layer of a neural network
        return: mish activation function
    '''
    return x*(torch.tanh(F.softplus(x)))

#---Save Reward list to Text file--#
def save_result(result_list, filename):
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode) as f:
        for item in result_list:
            f.write(str(item) + '\n')
    print("The result list has been successfully saved.")

# Load demonstrations and add them to the ReplayBuffer
def load_demonstrations(file_path):
    with open(file_path, 'rb') as f:
        demonstrations = joblib.load(f)
        #demonstrations = pickle.load(f)
    return demonstrations

# Add demonstration data to ReplayBuffer
def add_demonstrations(demo_data):
    for exp in demo_data:
        s, a, r, s2, done = exp
        s = np.float32(s)
        a = np.float32(a)
        r = np.float32(r)
        s2 = np.float32(s2)
        done = done

        pre_ram.add(s, a, r, s2, done)

        if not done or n_step == 0:
            ram.add((s, a, r, s2, GAMMA, 0))  # Add one-step to memory, last step added in pop with done=True

        # Add new step to N-step and Pop N-step data to memory
        if n_step > 0:
            backup.add_exp(
                (s, a, r, s2))  # Push to N-step backup
            trainer.add_n_step_experience(0, done)

    ram.set_protect_size(len(ram))


#---Where the train is made---#

BATCH_SIZE = 256
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001
epsilon = 0.00001

#---Run agent---#

is_training = True

MAX_EPISODES = 1000 #5000
MAX_STEPS = 1000
MAX_BUFFER = 1000000 #int(1e6) #200000
reward_list = []
train_time_list = []

reward_path = dirPath + '/Models/ddpgfd/hwasu_stage_2/new_result/train_stage_2_reward_result.txt'
time_path = dirPath + '/Models/ddpgfd/hwasu_stage_2/new_result/train_stage_2_time_result.txt'

STATE_DIMENSION = 14
ACTION_DIMENSION = 2
ACTION_V_MAX = 0.22 # m/s
ACTION_W_MAX = 2.0 #2. # rad/s	
#world = 'world_u'

if is_training:
    var_v = ACTION_V_MAX*.5
    var_w = ACTION_W_MAX*2*.5
else:
    var_v = ACTION_V_MAX*0.10
    var_w = ACTION_W_MAX*0.10

print('State Dimensions: ' + str(STATE_DIMENSION))
print('Action Dimensions: ' + str(ACTION_DIMENSION))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')

### DDPGfD add ###
const_min_priority = 0.001
const_demo_priority = 0.99
n_step = 10
backup = NStepBackup(GAMMA, n_step=n_step)
pre_ram = MemoryBuffer(MAX_BUFFER)
ram = PrioritizedReplayBuffer(MAX_BUFFER, alpha=0.3, beta_init=1.0, beta_inc_n=100, seed=1000)
demonstrations = load_demonstrations('/home/unicon1/catkin_ws/src/project/src/demonstrations_data_stage_2_10000goals.pkl')
backup.reset() # buffer clear
trainer = Trainer(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, ram)
add_demonstrations(demonstrations)
trainer.pretrain()
noise = OUNoise(ACTION_DIMENSION, max_sigma=0.1, min_sigma=0.1, decay_period=8000000)

load_model_episode = 0                         # 0 is pretrain model load
#trainer.load_models(episode=load_model_episode)

if __name__ == '__main__':
    rospy.init_node('ddpgfd_stage_2')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)

    result = Float32()
    env = Env(action_dim=ACTION_DIMENSION, train=is_training)
    before_training = 4

    past_action = np.zeros(ACTION_DIMENSION)
    
    total_start_time = time.time()

    reached_goal_flag_cnt = 0

    #if pre_ram.len > BATCH_SIZE:
    #    # pretrain 1000 steps
    #    for _ in range(1000):
    #        trainer.pretrain()

    for ep in range(load_model_episode+1, MAX_EPISODES+1):

        episode_start_time = time.time()
        step_start_time = 0
        done = False
        state = env.reset()
        reward_episode = 0.
        reached_goal_10_times_flag = False

        if is_training and not ep%10 == 0 and len(ram) >= before_training*MAX_STEPS:
            print('---------------------------------')
            print('Episode: ' + str(ep) + ' training')
            print('---------------------------------')
        else:
            if len(ram) >= before_training*MAX_STEPS:
                print('---------------------------------')
                print('Episode: ' + str(ep) + ' evaluating')
                print('---------------------------------')
            else:
                print('---------------------------------')
                print('Episode: ' + str(ep) + ' adding to memory')
                print('---------------------------------')        

        ram.update_beta()
        print("PER beta: "+str(ram.beta))
        print("OU noise sigma: "+str(noise.sigma))

        for step in range(1, MAX_STEPS+1):
            
            state = np.float32(state)

            # if is_training and not ep%10 == 0 and len(ram) >= before_training*MAX_STEPS:
            if is_training and not ep%10 == 0:
                action = trainer.get_exploration_action(state)
                # action[0] = np.clip(
                #     np.random.normal(action[0], var_v), 0., ACTION_V_MAX)
                # action[0] = np.clip(np.clip(
                #     action[0] + np.random.uniform(-var_v, var_v), action[0] - var_v, action[0] + var_v), 0., ACTION_V_MAX)
                # action[1] = np.clip(
                #     np.random.normal(action[1], var_w), -ACTION_W_MAX, ACTION_W_MAX)
                N = copy.deepcopy(noise.get_noise(t=step))
                
                #print("noise sigma: "+str(noise.sigma))
                N[0] = N[0]*ACTION_V_MAX / 2
                N[1] = N[1]*ACTION_W_MAX
                action[0] = np.clip(action[0] + N[0], 0., ACTION_V_MAX)
                action[1] = np.clip(action[1] + N[1], -ACTION_W_MAX, ACTION_W_MAX)
            else:
                action = trainer.get_exploration_action(state)

            if not is_training:
                action = trainer.get_exploitation_action(state)
            next_state, reward, done, reached_goal_flag = env.step(action, past_action)
            
            if reached_goal_flag == True:
                #path_object_list = list()
                #robot_waypoint.reset_path()
                reached_goal_flag_cnt += 1

            if reached_goal_flag_cnt == 10:
                reached_goal_10_times_flag = True
                done = True
            
            reward_episode += reward
            # print('action', action,'r',reward)
            past_action = copy.deepcopy(action)
            next_state = np.float32(next_state)
            reward = np.float32(reward)
            GAMMA = np.float32(GAMMA)

            '''
            if not ep%10 == 0 or not len(ram) >= before_training*MAX_STEPS:
                if reward == 100.:
                    print('***\n-------- Maximum Reward ----------\n****')
                    for _ in range(3):
                        ram.add(state, action, reward, next_state, done)

                else:
                    ram.add(state, action, reward, next_state, done)
            '''
            if not ep%10 == 0 or not done or not reached_goal_flag:
                #print((state, action, np.float32(reward), next_state,
                #                               np.float32(GAMMA), 1))
                ram.add((state, action,reward, next_state, GAMMA, 1))
            if n_step > 0:
                backup.add_exp(
                    (state, action, reward, next_state))  # Push to N-step backup
                trainer.add_n_step_experience(1, done)  # Pop one

            state = copy.deepcopy(next_state)
            #print(len(ram))

            if len(ram) >= (before_training*MAX_STEPS) and is_training and not ep%10 == 0:
                # var_v = max([var_v*0.99999, 0.005*ACTION_V_MAX])
                # var_w = max([var_w*0.99999, 0.01*ACTION_W_MAX])
                trainer.optimizer()                

            if (done) or (step == MAX_STEPS) or (reached_goal_10_times_flag):
                #hard_update(trainer.target_actor, trainer.actor)
                #hard_update(trainer.target_critic, trainer.critic)
                print('reward per ep: ' + str(reward_episode))
                print('*\nbreak step: ' + str(step) + '\n*')
                # print('explore_v: ' + str(var_v) + ' and explore_w: ' + str(var_w))
                print('sigma: ' + str(noise.sigma))
                # rewards_all_episodes.append(rewards_current_episode)
                if not ep%10 == 0:
                    pass
                else:
                    # if len(ram) >= before_training*MAX_STEPS:
                    result = reward_episode
                    pub_result.publish(result)

                episode_end_time = time.time()
                episode_1_train_time = round(episode_end_time-episode_start_time, 2)    
                
                save_result([reward_episode], reward_path)
                save_result([episode_1_train_time], time_path)

                reached_goal_flag_cnt = 0
                break
        
        if ep%10 == 0: # Modify 20 -> 50
            trainer.save_models(ep)

    total_end_time = time.time()
    total_train_time = round(total_end_time-total_start_time, 2)
    print("total time: %f [s]" % total_train_time)    


print('Completed Training')
