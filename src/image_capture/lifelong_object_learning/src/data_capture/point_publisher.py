#!/usr/bin/env python
import rospy
# from tf import TransformListener
from std_msgs.msg import Int32
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion
import tf


class PointPublisher:
    def __init__(self, ar_tag_frame, point_topic):
        rospy.Subscriber(point_topic, PointStamped, self.point_callback)
        self.tf = tf.TransformListener()
        self.ar_tag_frame = ar_tag_frame
        self.last_seq = -1
        self.num_points = 0
        self.points = []
        self.pubs = []
        self.marker_pubs = []


    def point_callback(self, data):
        # if data.header.seq != self.last_seq:
        #     self.last_seq  = data.header.seq

        # find point coords in ar tag frame
        # t = self.tf.getLatestCommonTime(data.header.frame_id, self.map_frame)
        # data.header.stamp = t
        # point = self.tf.transformPoint(self.map_frame, data)
        point = data
        self.points.append(point)
        point_pub = rospy.Publisher('/object_point' + str(data.header.seq), PointStamped, queue_size=10)
        marker_pub = rospy.Publisher('/object_point_marker' + str(data.header.seq), Marker, queue_size = 10)
        self.pubs.append(point_pub)
        self.marker_pubs.append(marker_pub)
        self.num_points += 1

        print "Point " + str(data.header.seq) +" clicked"


def main():

    # ar_tag_frame = rospy.get_param('ar_tag_frame')
    # point_topic = rospy.get_param('point_topic')

    ar_tag_frame = '/tag_0'
    point_topic = '/clicked_point'

    rospy.init_node('point_publisher', anonymous=True)
    point_publisher = PointPublisher(ar_tag_frame, point_topic)
    num_pub = rospy.Publisher('/object_point_num', Int32, queue_size=10)

    rate = rospy.Rate(10) # 10hz
    while not (rospy.is_shutdown()):
        num_pub.publish(point_publisher.num_points)
        for i in range(point_publisher.num_points):
            point = point_publisher.points[i]
            point_pub = point_publisher.pubs[i]
            point_pub.publish(point)

            marker = Marker()
            marker.header.seq = i
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = point.header.frame_id
            marker.ns = "/clicked_points/"
            marker.id = i
            marker.type = 2
            marker.action = 0
            marker.pose.position = point.point
            quat = tf.transformations.quaternion_from_euler(0, 0, 0)
            orientation = Quaternion()
            orientation.x = quat[0]
            orientation.y = quat[1]
            orientation.z = quat[2]
            orientation.w = quat[3]
            marker.pose.orientation = orientation
            marker.scale.x = .01
            marker.scale.y = .01
            marker.scale.z = .01
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration.from_sec(1)
            marker.frame_locked = True

            marker_pub = point_publisher.marker_pubs[i]
            marker_pub.publish(marker)

        rate.sleep()

if __name__ == "__main__":
    main()
