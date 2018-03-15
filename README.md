# CS 461 Group 67 Repository by Julian Weisbord, Miles McCall, and Michael Rodriguez.

For our senior design capstone project, we will build an image classifier on top of an autonomous robot. By leveraging ROS (Robot Operating System) and the existing mobile robot platform, we will provide a Convolutional Neural Network (CNN) model that utilizes online learning so that the robot can continuously learn to recognize objects in its environment.

## Installation
### 1. git clone this repo
### 2. pip install -r dependencies.txt
### 3. Set PYTHONPATH in your .bashrc
	export PYTHONPATH=${PYTHONPATH}:~/Deep-Learning-For-Object-Recognition-on-a-Mobile-Robot/src
### More Dependencies
For now, this repository only supports Python 2

- If you are working on the fetch robot in the OSU Robotics department, make sure you have access and can connect to the hidden robotics wifi network.

Install ROS Indigo for Ubuntu 14.04LTS

Install Catkin with: 

	sudo apt-get install ros-indigo-catkin

Create catkin_ws by running the following commands:

	mkdir -p ~/catkin_ws/src
	cd ~/catkin_ws/
	catkin_make

Add /src/image_capture/lifelong_object_learning package to the robots catkin_ws/src directory

Add the following two lines to your .bash_rc

	export ROS_MASTER_URI=http://bandit.engr.oregonstate.edu:11311
	export ROS_IP=00.000.000.000  //Add your computers IP adress

Install the necessary ROS fetch packages to your local computer

	sudo apt-cache search fetch
	sudo apt-get install ros-indigo-fetch-description

## To Run Data Capture

### Make new map

ON ROBOT SIDE

1. ssh into  robot and source workspace
	
	ssh username@bandit.engr.oregonstate.edu
	source catkin_ws/devel/setup.bash

2. Start creating map of room by running the command below and then driving the robot around room to scan
	
	rosrun gmapping slam_gmapping scan:=base_scan _odom_frame:=odom

ON LOCAL COMPUTER SIDE

3. Once the entire room is scanned, save map with: 
	
	rosrun map_server map_saver -f <map_name>

4. Save another copy to make the keepout file

	rosrun map_server map_saver -f <map_name_keepout>

5. Edit the map_name_keepout.pgm with an image editor to add black lines to indicate which sections of the map the robot should be staying out of. You can look at ours in results/setup/graf_HRI_keepout.pgm to get an idea of what it can look like.

7. Add .yaml and .pgm of both maps (4 files total) to the robot under lifelong_object_learning/mapping
	
	scp map_name.ext username@bandit.engr.oregonstate.edu:/home/user/catkin_ws/src/lifelong_object_learning/mapping/

SWITCH BACK TO ROBOT SIDE WINDOW

8. Edit the launch file to tell it to read from these map files 

	vim catkin_ws/src/lifelong_object_learning/launch/startup.launch

### Run capture_data.py

ON ROBOT SIDE

1. Open new terminal window to ssh into robot and source workspace
	
	ssh username@bandit.engr.oregonstate.edu
	source catkin_ws/devel/setup.bash

2. Launch the startup launch file 

	roslaunch lifelong_object_learning startup.launch

ON LOCAL COMPUTER SIDE

3. Open new terminal window and run the command below to make sure you are talking to Fetch on robotics network

	rostopic echo base_scan

4. Copy the fetch.rviz file from results/setup/fetch.rviz to your current working directory and then run the following command:

	rosrun rviz rviz -d fetch.rviz

ON ROBOT SIDE

5. Repeat Step 1

6. Update class name and instance. Also update starting image index if necessary.  
	
	vim catkin_ws/src/lifelong_object_learning/src/data_capture/capture_data.py

7. Register Points in point cloud by clicking "register point" in top right of RVIZ GUI and put 4 points around the object.

8. Add marker to visualization by clicking add in bottom left of RVIZ GUI and look for /object_point_marker. Once added, you should see a blue dot appear where you clicked in RVIZ.

9. Run capture_data.py to start data capturing process
	
	rosrun lifelong_learning capture_data.py    