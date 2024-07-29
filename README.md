# Deep-Reinforcement-Learning
This repository designs and validates an End-to-End controller that learns control inputs for moving a Turtlebot3 model to a desired location based on Deep Reinforcement Learning in the ROS Gazebo simulation environment.

<br><br>

# How to use

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

<br>

```
source ~/.bashrc
```
<br>

And

<br>

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
if 


And
```
cd ~ && cd catkin_ws/src/Deep-Reinforcement-Learning/turtlebot3_gazebo/launch
```
And
```
cp * ~/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch
```


## 6. Run Code

First terminal:
```
roslaunch turtlebot3_gazebo hs_turtlebot3_stage_{number_of_stage}.launch
```
In another terminal:
```
roslaunch project hs_ddpg_stage_{number_of_stage}.launch
```
