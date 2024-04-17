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
        self.heuristics = {'closest_goal': Heuristic()}

        self.goals = [model for model in self.models if model['namespace'] == "goal"]
        for goal in self.goals:
            tag = goal['tag']
            rospy.Subscriber(f'/gazebo/model_poses/goal/{tag}', PoseStamped, lambda msg, tag=tag: self.goal_pose_callback(msg, tag))
            self.publishers[tag] = rospy.Publisher(f'/heuristic/goal/{tag}', Heuristic, queue_size=10)  # Use 'tag' as the key
        self.publishers['closest_goal'] = rospy.Publisher('heuristic/goal/closest', Heuristic, queue_size=10)
        self.run()

    def robot_pose_callback(self, msg: PoseStamped):
        self.robot_pose_stamped = msg

    def goal_pose_callback(self, msg: PoseStamped, tag):
        self.heuristics[tag] = Heuristic()
        self.heuristics[tag].x_distance = self.robot_pose_stamped.pose.position.x - msg.pose.position.x
        self.heuristics[tag].y_distance = self.robot_pose_stamped.pose.position.y - msg.pose.position.y
        self.heuristics[tag].manhattan_distance = abs(self.heuristics[tag].x_distance) + abs(self.heuristics[tag].y_distance)
        self.publishers[tag].publish(self.heuristics[tag])  # Use 'tag' to access the correct publisher

    def run(self):
        while not rospy.is_shutdown():
            min_heuristic = float("inf")
            heuristics_copy = self.heuristics.copy()  # Create a copy of the dictionary
            for key, value in heuristics_copy.items():
                if 'closest_goal' != key:
                    if value.manhattan_distance < min_heuristic:
                        self.heuristics['closest_goal'] = value
                        min_heuristic = value.manhattan_distance
            self.publishers['closest_goal'].publish(self.heuristics['closest_goal'])


if __name__ == '__main__':
    heuristic_generator = GenerateHeuristic()
