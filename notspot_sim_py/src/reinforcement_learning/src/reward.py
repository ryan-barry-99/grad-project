#!/usr/bin/env python3

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from reinforcement_learning.msg import Heuristic
from std_msgs.msg import Float32, Bool, String
import os

class Reward:
    def __init__(self):
        rospy.init_node('reward_node', anonymous=True)
                
        self.new_episode_pub = rospy.Publisher('/RL/episode/new', Bool, queue_size=10)
        self.save_model_pub = rospy.Publisher('/RL/model/save', String, queue_size=10)
        self.robot_pose_stamped = PoseStamped()
        self.publishers = {
            'action_reward': rospy.Publisher('/RL/reward/action', Float32, queue_size=10),
            'total_reward': rospy.Publisher('/RL/reward/total', Float32, queue_size=10)
        }
        self.rospack = rospkg.RosPack()

        self.closest_heuristic = Heuristic()
        self.init_rewards_dir = False
        self.initialize_rewards()


        rospy.Subscriber('/gazebo/model_poses/robot/notspot', PoseStamped, self.robot_pose_callback)
        rospy.Subscriber('/RL/heuristic/goal/closest', Heuristic, self.heuristic_callback)
        rospy.Subscriber('/RL/episode/new', Bool, self.new_episode_callback)
        rospy.Subscriber('/RL/states/reward', String, self.calc_reward)

        rospy.spin()

    def robot_pose_callback(self, msg: PoseStamped):
        self.robot_pose_stamped = msg

    def heuristic_callback(self, msg: Heuristic):
        self.closest_heuristic = msg

    def new_episode_callback(self, msg: Bool):
        self.new_episode = msg.data
        if self.new_episode:
            with open(f'{self.rewards_folder}/episode_{self.episode_num}.txt', 'w') as f:
                f.write(str(self.total_reward.data))
            if self.episode_num % 10 == 0:
                self.save_model_pub.publish(f"episode_{self.episode_num}.pt")
            self.episode_num += 1
            self.total_reward = Float32()
            self.new_episode_pub.publish(False)

    def initialize_rewards(self):
        while not self.init_rewards_dir:
            if rospy.has_param('/RL/runs/rewards_folder'):
                self.rewards_folder = rospy.get_param('/RL/runs/rewards_folder')
                self.episode_num = len(os.listdir(self.rewards_folder))
                self.total_reward = Float32()
                self.new_episode = False
                self.init_rewards_dir = True

    def hits_wall_callback(self, msg):
        reward = -1
        self.publishers['action_reward'].publish(reward)

    def reach_goal_callback(self, msg):
        reward = 1
        self.publishers['action_reward'].publish(reward)

    def not_moving_callback(self, msg):
        reward = -1
        self.publishers['action_reward'].publish(reward)
            


    def calc_reward(self, msg: String):
        self.rewards = {
            "hits_wall": -1,
            "reach_goal": 1,
            "not_moving": -1,
        }
        if msg.data in self.rewards.keys():
            reward = self.rewards[msg.data]
            self.publishers['action_reward'].publish(reward)
            self.total_reward.data += reward
            self.publishers['total_reward'].publish(self.total_reward)
        if msg.data == "reach_goal" or msg.data == "stuck":
            self.new_episode_pub.publish(True)
            

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    reward = Reward()