#!/usr/bin/env python3

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from reinforcement_learning.msg import Heuristic
from std_msgs.msg import Float32, Bool
import os

class Reward:
    def __init__(self):
        rospy.init_node('reward_node', anonymous=True)
        
        rospy.Subscriber('/gazebo/model_poses/robot/notspot', PoseStamped, self.robot_pose_callback)
        rospy.Subscriber('/RL/heuristic/goal/closest', Heuristic, self.heuristic_callback)
        rospy.Subscriber('/RL/episode/new', Bool, self.new_episode_callback)
        
        self.new_episode_pub = rospy.Publisher('/RL/episode/new', Bool, queue_size=10)
        self.robot_pose_stamped = PoseStamped()
        self.publishers = {
            'action_reward': rospy.Publisher('/RL/reward/action', Float32, queue_size=10),
            'total_reward': rospy.Publisher('/RL/reward/total', Float32, queue_size=10)
        }
        self.rospack = rospkg.RosPack()
        self.rewards_folder = self.rospack.get_path('reinforcement_learning') + '/runs'

        self.closest_heuristic = Heuristic()
        self.init_rewards_dir = False
        self.initialize_rewards()

        self.run()

    def robot_pose_callback(self, msg: PoseStamped):
        self.robot_pose_stamped = msg

    def heuristic_callback(self, msg: Heuristic):
        self.closest_heuristic = msg

    def new_episode_callback(self, msg: Bool):
        self.new_episode = msg.data
        if self.new_episode:
            with open(f'{self.rewards_folder}/episode_{self.episode_num}.txt', 'w') as f:
                f.write(str(self.total_reward))
            self.episode_num += 1
            self.total_reward = 0
            self.new_episode_pub.publish(False)

    def initialize_rewards(self):
        while not self.init_rewards_dir:
            if rospy.has_param('/RL/runs/rewards_folder'):
                self.rewards_folder = rospy.get_param('/RL/runs/rewards_folder')
                self.episode_num = len(os.listdir(self.rewards_folder))
                self.total_reward = 0
                self.new_episode = False
                self.init_rewards_dir = True

    def calc_reward(self):
        pass

            

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    reward = Reward()