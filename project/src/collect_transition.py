#!/usr/bin/env python

import rospy
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import sys
import copy
sys.path.append('/home/unicon1/catkin_ws/src/project/src/utils/')
from environment_stage_1 import Env

dirPath = os.path.dirname(os.path.realpath(__file__))

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


class DemonstrationCollector:
    def __init__(self, actor_model_path, save_path):
        self.state_dim = 14
        self.action_dim = 2
        self.action_v_max = 0.22
        self.action_w_max = 2.
        self.env = Env(action_dim=self.action_dim, train=True)
        self.actor_model_path = actor_model_path
        self.save_path = save_path
        self.demonstrations = []
        self.reached_goal_flag_cnt = 0
        self.episode_1_experience = []
        # Load the model
        self.actor = None  # This will be initialized when loading the model
        self.load_model()


    def load_model(self):
        self.actor = Actor(self.state_dim, self.action_dim, self.action_v_max, self.action_w_max)
        self.actor.load_state_dict(torch.load(self.actor_model_path))
       

    def collect_demonstrations(self):

        past_action = np.zeros(self.action_dim)

        total_goal_time = 0

        #for ep in range(1, self.num_episodes+1):
        while True:

            if total_goal_time >= 10000:
                print("End collect demo data")
                break

            done = False

            state = self.env.reset()

            #goal_time = 0

            while True:
                
                if total_goal_time >= 10000:
                    self.save_demonstrations()
                    break

                state = np.float32(state)
                state = torch.from_numpy(state)
                action = self.actor.forward(state).detach()
                action = action.data.numpy()
                next_state, reward, done, reached_goal = self.env.step(action, past_action)  # Assumption: the environment's step function returns these values
                
                if reached_goal:
                    #goal_time += 1
                    total_goal_time += 1
                    print("Total reached goal: "+str(total_goal_time))
                    self.reached_goal_flag_cnt += 1
                    #done = True
                    for idx in range(len(self.episode_1_experience)):
                        self.demonstrations.append(self.episode_1_experience[idx])
                    self.episode_1_experience = []

                self.episode_1_experience.append((state, action, reward, next_state, done))
                #self.demonstrations.append((state, action, reward, next_state, done))

                if (done): # or (goal_time == 10):        
                    self.episode_1_experience = []       
                    break

                past_action = copy.deepcopy(action)
                state = next_state

        print("Number of times the goal position is reached: " + str(self.reached_goal_flag_cnt))

    def save_demonstrations(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.demonstrations, f)

        print("Demonstrations saved to "+self.save_path)

# Usage
if __name__ == '__main__':
    rospy.init_node('ddpgfd_data_collector')
    actor_model_path = dirPath + '/Models/ddpg/hwasu_stage_3/new_result/5000_actor.pt'  # Path to the pre-trained model
    collector = DemonstrationCollector(actor_model_path, save_path=dirPath+'/demonstrations_data_stage_3_1000goals.pkl')
    collector.collect_demonstrations()
    #collector.save_demonstrations()

