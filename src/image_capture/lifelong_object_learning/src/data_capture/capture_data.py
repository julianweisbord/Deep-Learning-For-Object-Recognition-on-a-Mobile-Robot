#!/usr/bin/env python
import rospy
from tf import TransformListener, transformations
from std_msgs.msg import String, Int32
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from move_base_msgs.msg import MoveBaseGoal
from move_base_msgs.msg import MoveBaseAction
from control_msgs.msg import PointHeadAction, PointHeadGoal, SingleJointPositionAction, SingleJointPositionGoal
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Quaternion
import actionlib
from actionlib_msgs.msg import GoalStatus
from moveit_msgs.msg import MoveItErrorCodes
from moveit_python import MoveGroupInterface, PlanningSceneInterface
from image_geometry import PinholeCameraModel
import cv2
import math
import sets
import random
import os
import argparse

class Node:
    def __init__(self, image_topic, camera_info_topic,
                 camera_frame, published_point_num_topic, published_point_base_topic,
                 torso_movement_topic, head_movement_topic, num_published_points,
                 max_spine_height, min_spine_height, spine_offset):

        self.camera_frame = camera_frame
        self.published_point_base_topic = published_point_base_topic
        self.bridge = CvBridge()
        self.tf = TransformListener()

        rospy.Subscriber(image_topic, Image, self.image_callback)
        rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)

        self.robot_pose = None
        self.img = None
        self.pc = None
        self.camera_info = None
        self.num_published_points = num_published_points
        self.published_points = [[0, 0, 0] for i in range(num_published_points)]
        for i in range(num_published_points):
            rospy.Subscriber(self.published_point_base_topic + str(i),
                             PointStamped, self.point_published_callback, i)
        self.points_registered = sets.Set()

        # base movement
        self.base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.base_client.wait_for_server()

        # torso movement
        self.max_spine_height = max_spine_height
        self.min_spine_height = min_spine_height
        self.spine_offset = spine_offset

        # head movement
        self.point_head_client = actionlib.SimpleActionClient(head_movement_topic, PointHeadAction)
        self.point_head_client.wait_for_server()

        # keyboard listener
        self.keypress_sub = rospy.Subscriber('/key_monitor', String, self.key_callback)

        rospy.loginfo("move group")
        self.move_group = MoveGroupInterface("arm_with_torso", "base_link")
        rospy.loginfo("move group end")

    def robot_pose_callback(self, data):
        self.robot_pose = data

    def image_callback(self, data):
        try:
            self.img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def move_base_to(self, x, y, theta):
        goal = MoveBaseGoal()
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y

        quat = transformations.quaternion_from_euler(0, 0, theta)
        orientation = Quaternion()
        orientation.x = quat[0]
        orientation.y = quat[1]
        orientation.z = quat[2]
        orientation.w = quat[3]
        goal.target_pose.pose.orientation = orientation

        # send goal
        self.base_client.send_goal(goal)

    def move_torso(self, pose):
        joint_names = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint',
                       'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint',
                       'wrist_flex_joint', 'wrist_roll_joint']

        poses = [pose, 1.3192424769714355, 1.4000714648620605,
                 -0.20049656002880095, 1.5290160491638183, -0.0004613047506046297,
                 1.660243449769287, -0.00012475593825578626]

        self.move_group.moveToJointPosition(joint_names, poses, wait=False)   # plan
        self.move_group.get_move_action().wait_for_result()

        return self.move_group.get_move_action().get_result()

    def look_at(self, frame_id, x, y, z):
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame_id
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z

        goal.pointing_frame = "pointing_frame"
        goal.pointing_axis.x = 1
        goal.pointing_axis.y = 0
        goal.pointing_axis.z = 0

        # send goal
        self.point_head_client.send_goal(goal)

    def camera_info_callback(self, data):
        self.camera_info = data

    def key_callback(self, keypress):
        if keypress.data == "k":
            self.base_client.cancel_goal()

    def point_published_callback(self, data, point_id):
        self.points_registered.add(point_id)
        self.published_points[point_id][0] = data.point.x
        self.published_points[point_id][1] = data.point.y
        self.published_points[point_id][2] = data.point.z

    def sample_position(self, x_center, y_center, sample_max_radius, sample_min_radius):
        min_x = x_center - sample_max_radius
        max_x = x_center - sample_min_radius*math.sin(.1745)

        sampled_theta = random.random()*(2*math.pi)
        sampled_r = random.random()*(sample_max_radius - sample_min_radius) + sample_min_radius
        sampled_x = sampled_r*math.cos(sampled_theta) + x_center
        sampled_y = sampled_r*math.sin(sampled_theta) + y_center
        sampled_z = random.random()*(self.max_spine_height - self.min_spine_height) + self.min_spine_height

        x_diff = sampled_x - x_center
        y_diff = sampled_y - y_center

        theta = math.atan2(y_diff, x_diff) + math.pi
        if theta > (2*math.pi):
            theta = theta - (2*math.pi)
        if theta < 0:
            theta = theta + (2*math.pi)

        position = [sampled_x, sampled_y, theta, sampled_z]

        return position

    def get_img(self):
        return self.img

    def get_pc(self):
        return self.pc

    def rviz_point_tl_tf(self):
        return self.rviz_point_tl_tf

    def rviz_point_br_tf(self):
        return self.rviz_point_br_tf


def euclidian_dist(point_1, point_2):
    return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2 + (point_1[2] - point_2[2])**2)

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Fetch Data Capture')
    parser.add_argument('-c', '--class',
        dest='class_name',
        action='store',
        help='object class name: mug, stapler, book, etc...',
        required=True)
    parser.add_argument('-n', '--num', '--number',
        dest='class_number',
        action='store',
        help='object number in data set: mug # 6, 7, 8, etc...',
        required=True)

    print len(sys.argv)
    if(len(sys.argv)):
        print "Need to provide command line arguments, use \"-help\" to see examples."
        exit();

    args = parser.parse_args()


    # Initialize variables
    class_name = args.class_name
    instance_name = class_name + "_" + args.class_number

    rospy.init_node('data_collector', anonymous=True)

    num_published_points = 4
    sample_min_radius = .6
    sample_max_radius = 1
    sample_height = .5
    height_offset = 1.0
    num_positions_to_sample = 80
    max_spine_height = .386
    min_spine_height = 0.00313
    spine_offset = 0.0

    # set starting_image_index back to 0 when doing a new object
    starting_image_index = 0
    desired_num_images = 80

    # Relevant paths
    results_path = "/home/mccallm/catkin_ws/src/lifelong_object_learning/data_captured/"
    base_filepath = results_path + class_name + "/" + instance_name
    image_filepath = results_path + class_name + "/" + instance_name + "/images/"
    circle_image_filepath = results_path + class_name + "/" + instance_name + "/circle_images/"
    image_data_filepath = results_path + class_name + "/" + instance_name + "/metadata/"

    # Path dir creation
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(base_filepath):
        os.makedirs(base_filepath)
    if not os.path.exists(image_filepath):
        os.makedirs(image_filepath)
    if not os.path.exists(circle_image_filepath):
        os.makedirs(circle_image_filepath)
    if not os.path.exists(image_data_filepath):
        os.makedirs(image_data_filepath)

    # ROS paths
    image_topic = "/head_camera/rgb/image_rect_color"
    camera_info_topic = "/head_camera/rgb/camera_info"
    map_frame = "/map"
    camera_frame = "/head_camera_rgb_optical_frame"
    ar_tag_frame = "/april_tag_0"
    published_point_num_topic = "/object_point_num"
    published_point_base_topic = "/object_point"
    torso_movement_topic = "/torso_controller/follow_joint_trajectory"
    head_movement_topic = "/head_controller/point_head"

    node = Node(image_topic, camera_info_topic, camera_frame,
                published_point_num_topic, published_point_base_topic, torso_movement_topic,
                head_movement_topic, num_published_points, max_spine_height,
                min_spine_height, spine_offset)

    count_pub = rospy.Publisher('data_capture_index', String, queue_size=10)

    camera_model = PinholeCameraModel()
    while node.camera_info is None:     # wait for camera info
        continue
    camera_model.fromCameraInfo(node.camera_info)

    while (len(node.points_registered) != node.num_published_points):
        rospy.loginfo(str(len(node.points_registered)))
        continue

    # find center of object
    x_center = 0.0
    y_center = 0.0
    z_center = 0.0

    for i in range(node.num_published_points):
        x_center += node.published_points[i][0]
        y_center += node.published_points[i][1]
        z_center += node.published_points[i][2]

    x_center = x_center/node.num_published_points
    y_center = y_center/node.num_published_points
    z_center = z_center/node.num_published_points

    rospy.loginfo("x center: " + str(x_center))
    rospy.loginfo("y center: " + str(y_center))
    rospy.loginfo("z center: " + str(x_center))

    # send first goal
    goalID = starting_image_index
    num_images_captured = starting_image_index
    position = node.sample_position(x_center, y_center, sample_max_radius, sample_min_radius)

    goal_x = position[0]
    goal_y = position[1]
    goal_theta = position[2]

    rospy.loginfo("Goal is " + str(goal_x) + " " + str(goal_y) + " " + str(goal_theta))
    rospy.loginfo("Sending goal")
    node.move_base_to(goal_x, goal_y, goal_theta)

    g_status = GoalStatus()

    image_file_index = starting_image_index
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        if (node.base_client.get_state() == g_status.PREEMPTED) or ((node.base_client.get_state() == g_status.ABORTED) or (node.base_client.get_state() == g_status.REJECTED)):
            position = node.sample_position(x_center, y_center, sample_max_radius, sample_min_radius)
            goal_x = position[0]
            goal_y = position[1]
            goal_theta = position[2]

            count_pub.publish("New goal ID is " + str(goalID))
            rospy.loginfo("New goal ID is " + str(goalID))
            rospy.loginfo("Goal is " + str(goal_x) + " " + str(goal_y) + " " + str(goal_theta))
            rospy.loginfo("Sending goal")
            node.move_base_to(goal_x, goal_y, goal_theta)

        # when base reaches position, adjust spine and point camera at object
        if (node.base_client.get_state() == g_status.SUCCEEDED):
            # adjust spine height
            spine_height = position[3]

            result = node.move_torso(spine_height)
            rospy.loginfo("Adjusting spine")
            if result:
                if result.error_code.val == MoveItErrorCodes.SUCCESS:
                    rospy.loginfo("Spine adjustment succeeded")

                    # make robot look at object
                    rospy.sleep(1)
                    rospy.loginfo("Turning head")
                    node.look_at("/map", x_center, y_center, z_center)
                    result = node.point_head_client.wait_for_result()
                    rospy.loginfo(result)

                    if result == True:
                        rospy.loginfo("Head turn succeeded")
                        rospy.sleep(.1)

                        # capture and save image
                        img_cur = node.get_img()
                        rospy.sleep(.1)
                        if (img_cur is not None) and (len(node.points_registered) == node.num_published_points):
                            rospy.loginfo("Capturing image")

                            # find pixel coordinates of points of interest
                            # tl, tr, bl, br
                            ref_points = node.published_points

                            height, width, channels = img_cur.shape

                            ref_points_camera_frame = []
                            points_to_write = []
                            for ref_point in ref_points:
                                ps = PointStamped()
                                ps.header.frame_id = map_frame
                                ps.header.stamp = node.tf.getLatestCommonTime(
                                    camera_frame, ps.header.frame_id)
                                # ps.header.stamp = rospy.Time.now()
                                ps.point.x = ref_point[0]
                                ps.point.y = ref_point[1]
                                ps.point.z = ref_point[2]

                                ps_new = node.tf.transformPoint(camera_frame, ps)
                                # ref_points_camera_frame.append([ps_new.point.x, ps_new.point.y, ps_new.point.z])

                                (u, v) = camera_model.project3dToPixel(
                                    (ps_new.point.x, ps_new.point.y, ps_new.point.z))
                                points_to_write.append([int(round(u)), int(round(v))])

                            image_file = image_filepath + instance_name + \
                                "_" + str(image_file_index) + '.png'
                            circle_image_file = circle_image_filepath + \
                                instance_name + "_" + str(image_file_index) + '.png'
                            text_file = image_data_filepath + instance_name + \
                                "_" + str(image_file_index) + '.txt'

                            f = open(text_file, 'w')
                            f.write(image_file + "\n")
                            f.write(str(height) + "\t" + str(width) + "\n")

                            for point in points_to_write:
                                f.write(str(point[0]) + "\t")
                                f.write(str(point[1]) + "\n")

                            f.write(str(goal_x) + "\n")
                            f.write(str(goal_y) + "\n")
                            f.write(str(position[3]) + "\n")        # spine height
                            f.close()

                            circle_img = img_cur.copy()
                            # visualize
                            for point in points_to_write:
                                cv2.circle(circle_img, (point[0], point[1]), 2, (0, 0, 255), 3)

                            cv2.imwrite(circle_image_file, circle_img)

                            cv2.imwrite(image_file, img_cur)
                            image_file_index += 1
                            rospy.loginfo("Metadata and Image saved")
                            rospy.loginfo("Num images captured: " + str(image_file_index))

                            # update goal id
                            goalID += 1
                            num_images_captured += 1

            # Send next position
            position = node.sample_position(
                x_center, y_center, sample_max_radius, sample_min_radius)
            goal_x = position[0]
            goal_y = position[1]
            goal_theta = position[2]

            # exit if we have tried all positions
            if num_images_captured == desired_num_images:
                rospy.loginfo("Total images captured: " + str(image_file_index))
                return

            # move to next position
            count_pub.publish("New goal ID is " + str(goalID))
            rospy.loginfo("New goal ID is " + str(goalID))
            rospy.loginfo("Goal is " + str(goal_x) + " " + str(goal_y) + " " + str(goal_theta))
            rospy.loginfo("Sending goal...")
            node.move_base_to(goal_x, goal_y, goal_theta)

        rate.sleep()

if __name__ == "__main__":
    main()
