#!/usr/bin/env python3

import rospy
import rospkg
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String, Int32
import subprocess
import random
from numpy import pi

class RobotSpawner:
    def __init__(self):
        rospy.init_node('robot_reset')
        rospy.Subscriber('/robot_reset_command', String, self.reset_callback)
        rospy.Subscriber('/robot_reset_position', Pose, self.pose_callback)
        # rospy.Subscriber('/RL/environment', Int32, self.environment_callback)

        self.rospack = rospkg.RosPack()
        self.urdf_path = self.rospack.get_path('notspot_description') + '/urdf/notspot.urdf'
        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.robot_name = 'notspot_gazebo'
        self.robot_exists = True
        
        # Process the URDF file using the xacro command-line tool
        self.urdf_xml = self.expand_xacro(self.urdf_path)

        self.pose = Pose(position=Point(0.0, 0.0, 0.15), orientation=Quaternion(0.0, 0.0, 0.0, 1.0))
        self.hydrant_pose = Pose(position=Point(7.5, 7.5, 0), orientation=Quaternion(0.0, 0.0, 0.0, 1.0))

        rospy.spin()

    # def environment_callback(self, msg):
    #     random_z_rotation = random.uniform(0,2*pi)
    #     # axis = random.choice([0,1,2,3])
    #     # if axis == 0:
    #     #     x_start = random.uniform(0, -4.5)
    #     #     y_start = 0
    #     # elif axis == 1:
    #     #     x_start = random.uniform(-3.25, -4)
    #     #     y_start = random.uniform(0,3.5)
    #     # elif axis == 2:
    #     #     x_start = random.uniform(-3.25, -4.5)
    #     #     y_start = random.uniform(2.5, 3.5)
    #     # elif axis == 3:
    #     #     x_start = random.uniform(-8.5,-9.5)
    #     #     y_start = random.uniform(1, 3.5)
    #     x_start = 0
    #     y_start = 0

    #     if msg.data == 1:
    #         y_start -= 10

    #     quaternion = quaternion_from_euler(0, 0, random_z_rotation)
    #     self.pose = Pose(position=Point(x_start, y_start, 0.15), orientation=Quaternion(*quaternion))
        
        # if msg.data == 0:
        #     quaternion = quaternion_from_euler(0, 0, random_z_rotation)
        #     self.pose = Pose(position=Point(x_start, 0.0, 0.15), orientation=Quaternion(*quaternion))
        # else:
        #     quaternion = quaternion_from_euler(0, 0, random_z_rotation)
        #     self.pose = Pose(position=Point(x_start, -10.0, 0.15), orientation=Quaternion(*quaternion))
    def set_robot_pose(self):
        # try:
        set_state_msg = ModelState()
        set_state_msg.model_name = self.robot_name
        set_state_msg.pose = self.pose
        self.set_state_service(set_state_msg)

        set_state_msg = ModelState()
        set_state_msg.model_name = 'fire_hydrant'
        self.get_hydrant_pose()
        set_state_msg.pose = self.hydrant_pose
        self.set_state_service(set_state_msg)

            
            # rospy.loginfo("Robot pose set successfully")
        # except rospy.ServiceException as e:
        #     rospy.logerr("Set Model State service call failed: %s", e)

    def get_hydrant_pose(self):
        while True:
            self.hydrant_pose.position.x = random.uniform(-9.5, 9.5)
            self.hydrant_pose.position.y = random.uniform(-9.5, 9.5)
            if not (abs(self.hydrant_pose.position.x) < 2.5 and abs(self.hydrant_pose.position.y) < 2.5):
                break


    def reset_callback(self, data):
        if data.data == "reset":
            self.set_robot_pose()
        else:
            rospy.logwarn("Invalid command received: %s", data.data)

    def pose_callback(self, data):
            # Generate a random Z rotation (in radians)
            # random_z_rotation = random.uniform(0, 2 * 3.14159)
            random_z_rotation = 3.14159
            

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