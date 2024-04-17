#!/usr/bin/env python3

import rospy
import rospkg
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
import subprocess

class RobotSpawner:
    def __init__(self):
        rospy.init_node('robot_spawner')
        rospy.Subscriber('/robot_spawn_command', String, self.spawn_callback)
        rospy.Subscriber('/robot_spawn_position', Pose, self.pose_callback)

        self.spawn_service = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        self.delete_service = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.rospack = rospkg.RosPack()
        self.urdf_path = self.rospack.get_path('notspot_description') + '/urdf/notspot.urdf'
        self.robot_name = 'notspot_gazebo'
        self.robot_exists = True
        
        # Process the URDF file using the xacro command-line tool
        self.urdf_xml = self.expand_xacro(self.urdf_path)

        self.pose = Pose(position=Point(0.0, 0.0, 0.15), orientation=Quaternion(0.0, 0.0, 0.0, 1.0))

        rospy.spin()

    def spawn_robot(self):
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            if not self.robot_exists:
                self.robot_exists = self.spawn_service(self.robot_name, self.urdf_xml, 'notspot/', self.pose, 'world')
                if self.robot_exists:
                    rospy.loginfo("Robot spawned successfully")
                else:
                    rospy.logerr("Robot spawn failed")
            else:
                rospy.logwarn("Robot already exists")
        except rospy.ServiceException as e:
            rospy.logerr("Spawn service call failed: %s", e)

    def remove_robot(self):
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            self.delete_service(self.robot_name)
            self.robot_exists = False
            
        except rospy.ServiceException as e:
            rospy.logerr("Delete service call failed: %s", e)

    def set_robot_pose(self):
        try:
            set_state_msg = ModelState()
            set_state_msg.model_name = self.robot_name
            set_state_msg.pose = self.pose
            self.set_state_service(set_state_msg)
            rospy.loginfo("Robot pose set successfully")
        except rospy.ServiceException as e:
            rospy.logerr("Set Model State service call failed: %s", e)

    def spawn_callback(self, data):
        if data.data == "spawn":
            self.spawn_robot()
        elif data.data == "remove":
            self.remove_robot()
        elif data.data == "reset":
            self.set_robot_pose()
        else:
            rospy.logwarn("Invalid command received: %s", data.data)

    def pose_callback(self, data):
        self.pose.position = data.position
        self.pose.orientation = data.orientation

    def expand_xacro(self, file_path):
        try:
            result = subprocess.run(['xacro', '--inorder', file_path], capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            rospy.logerr("Failed to expand Xacro: %s", e)
            return None

if __name__ == '__main__':
    robot_spawner = RobotSpawner()