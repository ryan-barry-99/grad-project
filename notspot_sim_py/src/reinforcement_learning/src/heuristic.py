#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from reinforcement_learning.msg import Heuristic

class GenerateHeuristic:
    def __init__(self):
        rospy.init_node('heuristic_node', anonymous=True)
        params = rospy.get_param('~')
        self.models = params['models']
        
        rospy.Subscriber('/gazebo/model_poses/robot/notspot', PoseStamped, self.robot_pose_callback)
        self.robot_pose_stamped = PoseStamped()
        self.publishers = {}

        self.goals = [model for model in self.models if model['namespace'] == "goal"]
        for goal in self.goals:
            tag = goal['tag']
            rospy.Subscriber(f'/gazebo/model_poses/goal/{tag}', PoseStamped, lambda msg, tag=tag: self.goal_pose_callback(msg, tag))
            self.publishers[tag] = rospy.Publisher(f'/heuristic/goal/{tag}', Heuristic, queue_size=10)  # Use 'tag' as the key
        rospy.spin()

    def robot_pose_callback(self, msg):
        self.robot_pose_stamped = msg

    def goal_pose_callback(self, msg, tag):
        h = Heuristic()
        h.x = self.robot_pose_stamped.pose.position.x - msg.pose.position.x
        h.y = self.robot_pose_stamped.pose.position.y - msg.pose.position.y
        self.publishers[tag].publish(h)  # Use 'tag' to access the correct publisher

if __name__ == '__main__':
    heuristic_generator = GenerateHeuristic()
