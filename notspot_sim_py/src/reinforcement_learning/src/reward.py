#!/usr/bin/env python3

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from reinforcement_learning.msg import Heuristic
from std_msgs.msg import Float32, Bool, String, Int32
import os
import numpy as np

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
        self.epi_num_pub = rospy.Publisher("/RL/episode_num", Int32, queue_size=10)
        self.rospack = rospkg.RosPack()

        self.closest_heuristic = Heuristic()
        self.init_rewards_dir = False
        self.upright = False
        self.dist = np.inf
        self.initialize_rewards()


        rospy.Subscriber('/gazebo/model_poses/robot/notspot', PoseStamped, self.robot_pose_callback)
        rospy.Subscriber('/RL/heuristic/goal/closest', Heuristic, self.heuristic_callback)
        rospy.Subscriber('/RL/episode/new', Bool, self.new_episode_callback)
        rospy.Subscriber('/RL/states/reward', String, self.calc_reward)
        rospy.Subscriber('/RL/reward/dist', Float32, self.dist_callback)
        rospy.Subscriber('/RL/step', Bool, self.step_callback)


        self.run()

    def robot_pose_callback(self, msg: PoseStamped):
        self.robot_pose_stamped = msg

    def heuristic_callback(self, msg: Heuristic):
        self.closest_heuristic = msg

    def new_episode_callback(self, msg: Bool):
        self.new_episode = msg.data
        if self.new_episode:
            if self.episode_num % 50 == 0:
                self.save_model_pub.publish(f"episode_{self.episode_num}.pt")
            self.episode_num += 1
            self.total_reward = Float32()

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
            # "not_moving": 0,
            # "upright": 0,
            "fell": -1,
            "moving_forward": 0.1,
            # "moving_backward": 0
        }
        if msg.data in self.rewards.keys():
            reward = self.rewards[msg.data]
            self.publishers['action_reward'].publish(reward)
            self.total_reward.data += reward
        if msg.data == "upright":
            self.upright = True

        if "moving_forward" in msg.data:
            split_string = msg.data.split('/')

            # Get the part to the right of the slash
            right_of_slash = split_string[1] if len(split_string) > 1 else None
            dist = float(right_of_slash)
            
            # Convert to float
            if dist is not None:
                try:
                    if abs(dist) < 3:
                        self.publishers['action_reward'].publish(min(1, 100*abs(dist)))
                except ValueError:
                    rospy.loginfo("Cannot convert to float. The string after the slash is not a number.")
                    
    
        if msg.data == "not_moving":
            distance_threshold = 2 # Threshold for minimum distance moved
            max_penalty = -1.0  # Maximum penalty value

            # Calculate the penalty based on the distance moved
            penalty = -distance_threshold / max(distance_threshold - abs(self.dist), 0.001)  # Ensure non-zero denominator
            
            # Clip the penalty to ensure it does not exceed the maximum penalty
            penalty = min(penalty, max_penalty)

            penalty /= 100
            self.publishers['action_reward'].publish(penalty)
            self.total_reward.data += penalty
            
        
        self.publishers['total_reward'].publish(self.total_reward)
        if msg.data == "reach_goal" or msg.data == "stuck" or msg.data == "fell":
            self.new_episode_pub.publish(True)

    def dist_callback(self, msg: Float32):
        self.dist = msg.data
            

    def step_callback(self, msg):
        self.upright = False
    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.epi_num_pub.publish(self.episode_num)
            rate.sleep()

if __name__ == '__main__':
    reward = Reward()