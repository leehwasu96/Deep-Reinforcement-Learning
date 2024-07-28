#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import time
from gazebo_msgs.msg import ModelState, ModelStates

class Combination():
    def __init__(self):
        
        self.boundary = 2           # map size 2x2 [m^2]
        self.stop_distance = 0.3   # [m]
        self.move_distance = 0.01   # [m]
        
        self.move_distance_1 = self.move_distance   # [m]
        self.move_distance_2 = -self.move_distance   # [m]
        self.move_distance_3 = self.move_distance   # [m]
        self.move_distance_4 = -self.move_distance   # [m]

        self.pub_model = rospy.Publisher('gazebo/set_model_state', ModelState, queue_size=1)
        self.moving()

    def moving(self):
        while not rospy.is_shutdown():
            model = rospy.wait_for_message('gazebo/model_states', ModelStates)
            for i in range(len(model.name)):
                ###### Dynamic Obstacle 1 ######
                if model.name[i] == 'obstacle_1':
                    obstacle_1 = ModelState()
                    obstacle_1.model_name = model.name[i]
                    obstacle_1.pose = model.pose[i]

                    dist_1 = self.boundary - obstacle_1.pose.position.y
                    
                    if (dist_1 > (2*self.boundary - self.stop_distance)):
                        self.move_distance_1 = self.move_distance
                    elif (dist_1 < self.stop_distance):
                        self.move_distance_1 = -self.move_distance
                    obstacle_1.pose.position.y += self.move_distance_1
                
                    self.pub_model.publish(obstacle_1)

                ###### Dynamic Obstacle 2 ######
                if model.name[i] == 'obstacle_2':
                    obstacle_2 = ModelState()
                    obstacle_2.model_name = model.name[i]
                    obstacle_2.pose = model.pose[i]

                    dist_2 = self.boundary - obstacle_2.pose.position.y
                    
                    if (dist_2 > (2*self.boundary - self.stop_distance)):
                        self.move_distance_2 = self.move_distance
                    elif (dist_2 < self.stop_distance):
                        self.move_distance_2 = -self.move_distance
                    obstacle_2.pose.position.y += self.move_distance_2

                    self.pub_model.publish(obstacle_2)
                    
                ###### Dynamic Obstacle 3 ######
                if model.name[i] == 'obstacle_3':
                    obstacle_3 = ModelState()
                    obstacle_3.model_name = model.name[i]
                    obstacle_3.pose = model.pose[i]

                    dist_3 = self.boundary - obstacle_3.pose.position.y
                    
                    if (dist_3 > (2*self.boundary - self.stop_distance)):
                        self.move_distance_3 = self.move_distance
                    elif (dist_3 < self.stop_distance):
                        self.move_distance_3 = -self.move_distance
                    obstacle_3.pose.position.y += self.move_distance_3
                
                    self.pub_model.publish(obstacle_3)

                ###### Dynamic Obstacle 4 ######
                if model.name[i] == 'obstacle_4':
                    obstacle_4 = ModelState()
                    obstacle_4.model_name = model.name[i]
                    obstacle_4.pose = model.pose[i]

                    dist_4 = self.boundary - obstacle_4.pose.position.y
                    
                    if (dist_4 > (2*self.boundary - self.stop_distance)):
                        self.move_distance_4 = self.move_distance
                    elif (dist_4 < self.stop_distance):
                        self.move_distance_4 = -self.move_distance
                    obstacle_4.pose.position.y += self.move_distance_4
                
                    self.pub_model.publish(obstacle_4)

            time.sleep(0.05)

def main():
    rospy.init_node('combination_obstacle_1')
    try:
        combination = Combination()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
