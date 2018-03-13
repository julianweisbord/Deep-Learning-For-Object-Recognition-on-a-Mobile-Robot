# CS 461 Group 67 Repository by Julian Weisbord, Miles McCall, and Michael Rodriguez.

For our senior design capstone project, we will build an image classifier on top of an autonomous robot. By leveraging ROS (Robot Operating System) and the existing mobile robot platform, we will provide a Convolutional Neural Network (CNN) model that utilizes online learning so that the robot can continuously learn to recognize objects in its environment.

## Installation
### 1. git clone this repo
### 2. pip install -r dependencies.txt
### 3. Set PYTHONPATH in your .bashrc
	export PYTHONPATH=${PYTHONPATH}:~/Deep-Learning-For-Object-Recognition-on-a-Mobile-Robot/src
### More Dependencies
For now, this repository only supports Python 2

Install ROS Indigo for Ubuntu 14.04LTS

Install Catkin with: 

	sudo apt-get install ros-indigo-catkin

Create catkin_ws by running the following commands:

	mkdir -p ~/catkin_ws/src
	cd ~/catkin_ws/
	catkin_make

Add /src/image_capture/lifelong_object_learning package to the robots catkin_ws/src directory

Add following two lines to your .bash_rc

	export ROS_MASTER_URI=http://bandit.engr.oregonstate.edu:11311
	export ROS_IP=00.000.000.000  //Add your computers IP adress

Install necessary ROS fetch packages on local computer

	sudo apt-cache search fetch
	sudo apt-get install ros-indigo-fetch-description