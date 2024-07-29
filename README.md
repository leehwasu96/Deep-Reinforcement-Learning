# Deep-Reinforcement-Learning
This repository designs and validates an End-to-End controller that learns control inputs for moving a Turtlebot3 model to a desired location based on Deep Reinforcement Learning in the ROS Gazebo simulation environment.

# How to use

<br>

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
gedit turtlebot3_burger.gazebo.xacro
```
And
```
<xacro:arg name="laser_visual" default="false"/>   # Visualization of LDS. If you want to see LDS, set to `true`
```
And
```
<scan>
  <horizontal>
    <samples>360</samples>            # Number of LiDAR scan data sample. Modify it to 10
    <resolution>1</resolution>
    <min_angle>0.0</min_angle>
    <max_angle>6.28319</max_angle>
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
echo 'export ROS_PACKAGE_PATH=/home/unicon1/catkin_ws/src/Deep-Reinforcement-Learning/project:$ROS_PACKAGE_PATH' >> ~/.bashrc
```
And
```
source ~/.bashrc
```
And
```
cd ~ && cd catkin_ws/src/Deep-Reinforcement-Learning/turtlebot3_gazebo/launch
```
And
```
cp * ~/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch
```
