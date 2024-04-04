#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from reinforcement_learning.msg import Heuristic
from std_msgs.msg import Float32, Bool

class Reward:
    def __init__(self):
        rospy.init_node('reward_node', anonymous=True)
        
        rospy.Subscriber('/gazebo/model_poses/robot/notspot', PoseStamped, self.robot_pose_callback)
        rospy.Subscriber('/heuristic/goal/closest', Heuristic, self.heuristic_callback)
        rospy.Subscriber('/episode/new', Bool, self.new_episode_callback)
        self.robot_pose_stamped = PoseStamped()
        self.publishers = {
            'action_reward': rospy.Publisher('reward/action', Float32, queue_size=10),
            'total_reward': rospy.Publisher('reward/total', Float32, queue_size=10)
        }
        self.closest_heuristic = Heuristic()
        self.total_reward = 0
        self.episode_num = 0
        self.run()

    def robot_pose_callback(self, msg: PoseStamped):
        self.robot_pose_stamped = msg

    def heuristic_callback(self, msg: Heuristic):
        self.closest_heuristic = msg

    def calc_reward(self):
        pass

    def new_episode_callback(self, msg: Bool):
        with open(f'rewards/episode_{self.episode_num}.txt', 'w')as f:
            f.write(str(self.total_reward))
        self.episode_num += 1
        self.total_reward = 0

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    reward = Reward()