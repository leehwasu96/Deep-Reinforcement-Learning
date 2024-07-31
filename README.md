# Deep-Reinforcement-Learning
  - Author : Hwasu Lee<br><br>
  - Affiliation : UNICON LAB (Incheon National University, South Korea)<br><br>
  - Position : M.A. student<br><br>
  - E-mail : leehwasu96@naver.com (or leehwasu9696@inu.ac.kr)<br><br>

# Project Description
  - This repository designs and validates an End-to-End controller that learns control inputs for moving a Turtlebot3 model <br><br>
    to a desired location based on Deep Reinforcement Learning in the ROS Gazebo simulation environment. <br><br>

#  Repository Description
  - The 'project' directory contains packages related to DRL(Deep Reinforcement Learning). <br><br>
  - The 'turtlebot3_gazebo' directory is a folder that stores custom world information. <br><br>

**Note: This practice was conducted in an Ubuntu 18.04 LTS and ROS(Robot Operating System) 1 Melodic environment.** <br><br>

**To set up the project, follow these steps:** <br><br>

## 1. Installing packages related to ROS Gazebo simulation.
```
cd ~/catkin_ws/src/
```
And
```
git clone https://github.com/ROBOTIS-GIT/turtlebot3
git clone https://github.com/ROBOTIS-GIT/turtlebot3_msgs
git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations
git clone https://github.com/leehwasu96/Deep-Reinforcement-Learning.git
```

<br><br>

## 2. Building and sourcing the catkin workspace.
```
cd ~/catkin_ws && catkin_make
source devel/setup.bash
```

<br><br>

## 3. Configuring LiDAR scan data.
- In: turtlebot3/turtlebot3_description/urdf/turtlebot3_burger.gazebo.xacro.
```
roscd turtlebot3_description/urdf
```
If you encounter the error "roscd: No such package/stack 'turtlebot3_description/urdf'", execute the following command and then source it:
```
echo 'export TURTLEBOT3_MODEL=burger' >> ~/.bashrc
echo 'export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/home/{your local PC name}/catkin_ws/src/turtlebot3:$ROS_PACKAGE_PATH' >> ~/.bashrc
echo 'export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/home/{your local PC name}/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo:$ROS_PACKAGE_PATH' >> ~/.bashrc
echo 'export ROS_PACKAGE_PATH=/home/{your local PC name}/catkin_ws/src/Deep-Reinforcement-Learning/project:$ROS_PACKAGE_PATH' >> ~/.bashrc
```
Example image of a bashrc file
![1](https://github.com/user-attachments/assets/78b55cd0-44ed-49e1-b88c-e9a77a63513d)

<br>

And
```
source ~/.bashrc
```
And
```
gedit turtlebot3_burger.gazebo.xacro
```
And
```
<xacro:arg name="laser_visual" default="false"/>
```
Visualization of LDS. If you want to see LDS, set it to true as follows:
```
<xacro:arg name="laser_visual" default="true"/>
```
And
```
<scan>
  <horizontal>
    <samples>360</samples>
    <resolution>1</resolution>
    <min_angle>0.0</min_angle>
    <max_angle>6.28319</max_angle>
  </horizontal>
</scan>
```
If you want to modify the LiDAR data to detect only the front 180 degrees and recognize only 10 scan data points, update the code as follows:
```
<scan>
  <horizontal>
    <samples>10</samples>             <!-- Number of LiDAR scan data samples modified to 10 -->
    <resolution>1</resolution>
    <min_angle>-1.5708</min_angle>    <!-- -π/2 in radians -->
    <max_angle>1.5708</max_angle>     <!-- π/2 in radians -->
  </horizontal>
</scan>
```

<br><br>

## 4. Setting up Turtlebot3 world file.
```
cd ~ && cd catkin_ws/src/Deep-Reinforcement-Learning/turtlebot3_gazebo/worlds
```
And
```
cp * ~/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/worlds
```
And
```
roscd turtlebot3_gazebo/worlds
```

<br><br>

## 5. Configuring Turtlebot3 launch file.
```
cd ~ && cd catkin_ws/src/Deep-Reinforcement-Learning/turtlebot3_gazebo/launch
```
And
```
cp * ~/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch
```

<br><br>

## 6. Configuring Goal model file.
```
cd /home/{your local PC name}/catkin_ws/src/Deep-Reinforcement-Learning/turtlebot3_gazebo/models/turtlebot3_square/goal_box
```
And
```
cp model2.* ~/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box
```

<br><br>

## 7. Run Code

First terminal:
```
roslaunch turtlebot3_gazebo hs_turtlebot3_stage_{number_of_stage}.launch
```
Example image of Gazebo simulation environment <br>
<img src="https://github.com/user-attachments/assets/fc859e17-ec38-421d-b8c6-90feda93309f" alt="Example image of Gazebo simulation environment" width="500">

<br>

In another terminal:
```
roslaunch project hs_ddpg_stage_{number_of_stage}.launch
```

<br><br>

## 8. Validation result

**Validation Video 1**

[Validation Video 1](https://github.com/user-attachments/assets/9fd9f933-0f39-4ca6-bddc-f282881700d1)

**Validation Video 2**

[Validation Video 2](https://github.com/user-attachments/assets/4c8d99fd-807c-4954-bf1e-90f77ce08f3e)

**Validation Video 3**

[Validation Video 3](https://github.com/user-attachments/assets/19b1cf15-781e-48e6-bee9-a4a3b5f348c9)

**Validation Video 4**

[Validation Video 4](https://github.com/user-attachments/assets/04e4a073-3893-41ac-82a7-4fa6bd44c5a9)



