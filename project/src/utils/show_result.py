#!/usr/bin/env python

import matplotlib.pyplot as plt

def plot_from_txt(stage):

    comparison = False

    if comparison:
        
        ddpg_reward_path = '/home/unicon1/catkin_ws/src/project/src/Models/ddpg/hwasu_stage_'+str(stage)+'/new_result/train_stage_'+str(stage)+'_reward_result.txt'
        ddpg_time_path = '/home/unicon1/catkin_ws/src/project/src/Models/ddpg/hwasu_stage_'+str(stage)+'/new_result/train_stage_'+str(stage)+'_time_result.txt'
    
        ddpgfd_reward_path = '/home/unicon1/catkin_ws/src/project/src/Models/ddpgfd/hwasu_stage_'+str(stage)+'/new_result2/train_stage_'+str(stage)+'_reward_result.txt'
        ddpgfd_time_path = '/home/unicon1/catkin_ws/src/project/src/Models/ddpgfd/hwasu_stage_'+str(stage)+'/new_result2/train_stage_'+str(stage)+'_time_result.txt'
    
        ddpg_reward = []
        ddpgfd_reward = []
        ddpg_time = []
        ddpgfd_time = []

        # Read reward data from file
        with open(ddpg_reward_path, 'r') as file:
            for line in file:
                ddpg_reward.append(float(line))
        with open(ddpgfd_reward_path, 'r') as file:
            for line in file:
                ddpgfd_reward.append(float(line))

        # Read training time data from file
        with open(ddpg_time_path, 'r') as file:
            for line in file:
                ddpg_time.append(float(line))
        # Read training time data from file
        with open(ddpgfd_time_path, 'r') as file:
            for line in file:
                ddpgfd_time.append(float(line))


        length = len(ddpgfd_reward) #1000
        ddpg_reward = ddpg_reward[0:length]
        ddpgfd_reward = ddpgfd_reward[0:length]
        ddpg_time = ddpg_time[0:length]
        ddpgfd_time = ddpgfd_time[0:length]


        # Plotting the data
        plt.figure(figsize=(12, 6))  # Set the figure size (optional)

        # First subplot for the reward dataset
        plt.subplot(1, 2, 1)  # (rows, columns, subplot number)
        plt.plot(ddpg_reward, color='r', label='DDPG')
        plt.plot(ddpgfd_reward, color='b', label='DDPGfD')

        #plt.title('End-to-End Controller based DDPG')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()

        # Second subplot for the time dataset
        plt.subplot(1, 2, 2)  # (rows, columns, subplot number)
        plt.plot(ddpg_time, color='r', label='DDPG')
        plt.plot(ddpgfd_time, color='b', label='DDPGfD')
        #plt.title('End-to-End Controller based DDPG')
        plt.xlabel('Episode')
        plt.ylabel('Training time [s]')
        plt.legend()

        plt.suptitle('Results of DDPG and DDPGfD based End-to-End Controllers')

        # Adjust layout to prevent overlapping
        #plt.tight_layout()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        #plt.grid(True)
        plt.show()

    else:
        '''
        reward_path = '/home/unicon1/catkin_ws/src/project/src/Models/ddpg/hwasu_stage_'+str(stage)+'/new_result2/train_stage_'+str(stage)+'_reward_result.txt'
        time_path = '/home/unicon1/catkin_ws/src/project/src/Models/ddpg/hwasu_stage_'+str(stage)+'/new_result2/train_stage_'+str(stage)+'_time_result.txt'
        '''

        '''
        reward_path = '/home/unicon1/catkin_ws/src/project/src/Models/ddpgfd/hwasu_stage_'+str(stage)+'/new_result3/train_stage_'+str(stage)+'_reward_result.txt'
        time_path = '/home/unicon1/catkin_ws/src/project/src/Models/ddpgfd/hwasu_stage_'+str(stage)+'/new_result3/train_stage_'+str(stage)+'_time_result.txt'
        '''

        #'''
        reward_path = '/home/unicon1/catkin_ws/src/project/src/Models2/ddpg/hwasu_stage_'+str(stage)+'/new_result/train_stage_'+str(stage)+'_reward_result.txt'
        time_path = '/home/unicon1/catkin_ws/src/project/src/Models2/ddpg/hwasu_stage_'+str(stage)+'/new_result/train_stage_'+str(stage)+'_time_result.txt'
        #'''
        
        reward = []
        time = []

        # Read reward data from file
        with open(reward_path, 'r') as file:
            for line in file:
                reward.append(float(line))

        # Read training time data from file
        with open(time_path, 'r') as file:
            for line in file:
                time.append(float(line))

        # Plotting the data
        plt.figure(figsize=(12, 6))  # Set the figure size (optional)

        # First subplot for the reward dataset
        plt.subplot(1, 2, 1)  # (rows, columns, subplot number)
        plt.plot(reward, color='r')
        #plt.title('End-to-End Controller based DDPG')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        # Second subplot for the time dataset
        plt.subplot(1, 2, 2)  # (rows, columns, subplot number)
        plt.plot(time, color='b')
        #plt.title('End-to-End Controller based DDPG')
        plt.xlabel('Episode')
        plt.ylabel('Training time [s]')

        plt.suptitle('Results of DDPGfD based End-to-End Controllers')

        # Adjust layout to prevent overlapping
        #plt.tight_layout()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        #plt.grid(True)
        plt.show()


if __name__ == '__main__':
    stage_number = 4
    plot_from_txt(stage_number)
