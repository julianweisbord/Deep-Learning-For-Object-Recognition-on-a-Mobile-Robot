'''
Created on February 9th, 2018
author: Michael Rodriguez
sources: http://docs.fetchrobotics.com/
description: Module to monitor keyboard activity for ROS
'''
# External Imports
import rospy
import sys, select, termios, tty
# Local Imports
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# Definitions and Constants
msg = """
Reading from the keyboard  and Publishing to key_monitor!
---------------------------
k: kill
CTRL-C to quit
"""
moveBindings = {
		'k':'k'
}

def getKey():
	'''
		getKey is called to check standard input for user input
	    # Arguments
			None
	    # Returns
			Value of key
	    # Raises
			None
	'''
	tty.setraw(sys.stdin.fileno())
	select.select([sys.stdin], [], [], 0)
	key = sys.stdin.read(1)
	termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
	print key
	return key

if __name__=="__main__":
	settings = termios.tcgetattr(sys.stdin)
	pub = rospy.Publisher('key_monitor', String, queue_size=10)
	rospy.init_node('keyboard_monitor')
	print msg

	# Main loop that listens for key strokes
	while(1):
		key = getKey()
		if key in moveBindings.keys():
			key_press = key
			pub.publish(key_press)
		else:
			if (key == '\x03'):
				break
