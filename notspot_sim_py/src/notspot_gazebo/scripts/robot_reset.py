#!/usr/bin/env python3

import rospy
import rospkg
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String
import subprocess
import random

class RobotSpawner:
    def __init__(self):
        rospy.init_node('robot_reset')
        rospy.Subscriber('/robot_reset_command', String, self.reset_callback)
        rospy.Subscriber('/robot_reset_position', Pose, self.pose_callback)

        self.rospack = rospkg.RosPack()
        self.urdf_path = self.rospack.get_path('notspot_description') + '/urdf/notspot.urdf'
        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.robot_name = 'notspot_gazebo'
        self.robot_exists = True
        
        # Process the URDF file using the xacro command-line tool
        self.urdf_xml = self.expand_xacro(self.urdf_path)

        self.pose = Pose(position=Point(0.0, 0.0, 0.15), orientation=Quaternion(0.0, 0.0, 0.0, 1.0))

        rospy.spin()


    def set_robot_pose(self):
        try:
            set_state_msg = ModelState()
            set_state_msg.model_name = self.robot_name
            set_state_msg.pose = self.pose
            self.set_state_service(set_state_msg)
            # rospy.loginfo("Robot pose set successfully")
        except rospy.ServiceException as e:
            rospy.logerr("Set Model State service call failed: %s", e)

    def reset_callback(self, data):
        if data.data == "reset":
            self.set_robot_pose()
        else:
            rospy.logwarn("Invalid command received: %s", data.data)

    def pose_callback(self, data):
            # Generate a random Z rotation (in radians)
            random_z_rotation = random.uniform(0, 2 * 3.14159)

            # Convert the random Z rotation to a quaternion
            quaternion = quaternion_from_euler(0, 0, random_z_rotation)

            # Set the orientation (quaternion) of the pose
            self.pose.orientation = Quaternion(*quaternion)

    def expand_xacro(self, file_path):
        try:
            result = subprocess.run(['xacro', '--inorder', file_path], capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            rospy.logerr("Failed to expand Xacro: %s", e)
            return None

if __name__ == '__main__':
    robot_spawner = RobotSpawner()