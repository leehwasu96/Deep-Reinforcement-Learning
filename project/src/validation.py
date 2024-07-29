#!/usr/bin/env python

import rospy
import os
import random
import time
import numpy as np
import copy

#from ddpg_stage_1 import Critic, Actor, Trainer, MemoryBuffer
#from ddpg_stage_2 import Critic, Actor, Trainer, MemoryBuffer
#from ddpg_stage_3 import Critic, Actor, Trainer, MemoryBuffer
#from ddpg_stage_4 import Critic, Actor, Trainer, MemoryBuffer
#from ddpgfd_stage_1 import Critic, Actor, Trainer, MemoryBuffer
import ddpg_stage_4 as ddpg_stage
#import ddpgfd_stage_2 as ddpgfd_stage

#from environment_stage_1 import Env
#dirPath = os.path.dirname(os.path.realpath(__file__))

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir+'/utils')
from environment_stage_1 import Env


def save_result(result_list, filename):
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode) as f:
        for item in result_list:
            f.write(str(item) + '\n')
    print("The result list has been successfully saved.")


class Validation:

    def __init__(self):
        self.start_validation(algorithm='ddpg')
        #self.start_validation(algorithm='ddpgfd')

    def set_param(self):
      
        self.test_stage = 4

        #self.test_episodes = 10
        self.test_steps = 1000
        self.action_v_max = 0.2 # [m/s]
        self.action_w_max = 1.0 # [rad/s]
 
        self.state_dim = 34 #14
        self.action_dim = 2
        self.max_buffer_size = 200000
        self.env = Env(action_dim=self.action_dim, train=False)
        self.past_action = np.zeros(self.action_dim)

        #self.total_rewards = list()

        self.previous_step = 0

        self.reached_goal_count = 0
        self.done_and_reached_goal_count = 0
        self.total_reached_time = 0
        self.total_reward = 0
        self.total_step = 0     
        self.result = []        

        self.limit_goal_and_done_cnt = 1000

    def start_validation(self, algorithm):

        ddpg_episode = 1000
        ddpgfd_episode = 1000

        print('===== Start Validation =====')

        if algorithm == 'ddpg':
            self.set_param()
            self.ram = ddpg_stage.MemoryBuffer(self.max_buffer_size)
            trainer = ddpg_stage.Trainer(self.state_dim, self.action_dim, self.action_v_max, self.action_w_max, self.ram)
            print('DDPG model load...')
            trainer.load_models(episode = ddpg_episode)
            self.validation_result_path = dirPath + '/ddpg_stage_'+str(self.test_stage)+'_validation_result_in_'+str(ddpg_episode)+'episodes.txt'

        elif algorithm == 'ddpgfd':
            self.set_param()
            self.ram = ddpgfd_stage.MemoryBuffer(self.max_buffer_size)
            trainer = ddpgfd_stage.Trainer(self.state_dim, self.action_dim, self.action_v_max, self.action_w_max, self.ram)
            #self.ram = ddpg_stage.MemoryBuffer(self.max_buffer_size)
            #trainer = ddpg_stage.Trainer(self.state_dim, self.action_dim, self.action_v_max, self.action_w_max, self.ram)
            print('DDPGfD model load...')
            trainer.load_models(episode = ddpgfd_episode)
            self.validation_result_path = dirPath + '/ddpgfd_stage_'+str(self.test_stage)+'_validation_result_in_'+str(ddpgfd_episode)+'_episode_and_300000pretrain.txt'


        #for episode in range(1, self.test_episodes+1):
        while self.done_and_reached_goal_count < self.limit_goal_and_done_cnt:

            start = time.time()

            done = False
            state = self.env.reset()

            rewards_current_episode = 0
            self.current_episode_step = 0

            for current_step in range(1, self.test_steps+1):

                self.current_episode_step += 1
                #print("current_episode_step: "+str(self.current_episode_step))
                self.position = self.env.position

                state = np.float32(state)
                action = trainer.get_exploitation_action(state)
                action[0] = np.clip(action[0], 0., self.action_v_max)
                action[1] = np.clip(action[1], -self.action_w_max, self.action_w_max)

                next_state, reward, done, reached_goal_flag = self.env.step(action, self.past_action)
                
                rewards_current_episode += reward

                if reached_goal_flag:
                    self.reached_goal_count += 1
                    self.done_and_reached_goal_count += 1
                    print('Reached ' + str(self.reached_goal_count) + ' th goal !!')

                    self.total_step += self.current_episode_step
                    print('current step: ' + str(self.current_episode_step))

                    end = time.time()
                    reached_time = round(end - start, 2)
                    print('reached time: ' + str(reached_time) + ' [s]')
                    print('reward: ' + str(rewards_current_episode))
                    self.total_reached_time += reached_time
                    self.total_reward += rewards_current_episode
                    print('current total reward: ' + str(self.total_reward))
                    start = time.time()
                    rewards_current_episode = 0
                    self.current_episode_step = 0


                if self.current_episode_step > 300:
                    done = True


                if done:
                    self.done_and_reached_goal_count += 1
                    self.current_episode_step = 0
                    print('collision: ' + str(self.done_and_reached_goal_count-self.reached_goal_count))

                
                self.past_action = copy.deepcopy(action)
                next_state = np.float32(next_state)
                state = copy.deepcopy(next_state)
              
                if (done) or (current_step == self.test_steps) or (self.done_and_reached_goal_count==self.limit_goal_and_done_cnt):
                    print('reward per ep: ' + str(rewards_current_episode))
                    print('*\nbreak step: ' + str(current_step) + '\n*')
                    break

            #self.total_rewards.append(rewards_current_episode) 

        print('===========================================')
        print('===== Completed Valiation =====')
        print('===== Collision : ' + str(self.done_and_reached_goal_count - self.reached_goal_count) + ' =====')
        print('===== Total reached goal times : ' + str(self.reached_goal_count) + ' =====')
        print('===== Everage time : ' + str(self.total_reached_time/self.reached_goal_count) + ' [s] =====')
        print('===== Everage step : ' + str(self.total_step/self.reached_goal_count) + ' =====')
        print('===== Everage reward : ' + str(self.total_reward/self.reached_goal_count) + ' =====')
        print('===========================================')
        self.result.append('Collision: '+str(self.done_and_reached_goal_count - self.reached_goal_count))
        self.result.append('Total reached goal times: '+str(self.reached_goal_count))
        self.result.append('Everage time: '+str(self.total_reached_time/self.reached_goal_count))
        self.result.append('Everage step: '+str(self.total_step/self.reached_goal_count))
        self.result.append('Everage reward: '+str(self.total_reward/self.reached_goal_count))
        save_result(self.result, self.validation_result_path)

def main():
    Validation()

if __name__ == '__main__':
    rospy.init_node('ddpg_validation')
    main()
