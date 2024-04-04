#!/usr/bin/env python3

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from reinforcement_learning.msg import Heuristic
from std_msgs.msg import Float32, Bool
import os
from datetime import datetime

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
        self.total_reward = 0
        self.init_rewards_dir = False
        self.episode_num = 0
        self.set_rewards_folder()
        self.new_episode = False

        self.run()

    def robot_pose_callback(self, msg: PoseStamped):
        self.robot_pose_stamped = msg

    def heuristic_callback(self, msg: Heuristic):
        self.closest_heuristic = msg

    def calc_reward(self):
        pass

    def new_episode_callback(self, msg: Bool):
        self.new_episode = msg.data
        if self.new_episode:
            with open(f'{self.rewards_folder}/episode_{self.episode_num}.txt', 'w') as f:
                f.write(str(self.total_reward))
            self.episode_num += 1
            self.total_reward = 0
            self.new_episode_pub.publish(False)

    def set_rewards_folder(self):
        if not self.init_rewards_dir:
            if rospy.has_param('/RL/runs/new_run'):
                if rospy.get_param('/RL/runs/new_run'):
                    c = datetime.now()
                    current_time = c.strftime('%Y_%m_%d_%H_%M_%S')
                    self.runs_folder = f"{self.rewards_folder}/{current_time}"
                    os.mkdir(f"{self.runs_folder}")
                    self.rewards_folder = f"{self.runs_folder}/rewards"
                    os.mkdir(self.rewards_folder)
                    self.episode_num = len(os.listdir(self.rewards_folder))
                    self.init_rewards_dir = True
                else:
                    # List all folders in the directory
                    folders = os.listdir(self.rewards_folder)

                    # Filter out non-folder items and sort by name (timestamp)
                    folders = sorted(folder for folder in folders if os.path.isdir(os.path.join(self.rewards_folder, folder)))
                    
                    # Select the most recent folder (the last one after sorting)
                    if folders:
                        self.rewards_folder = self.rospack.get_path('reinforcement_learning') + '/runs/' + folders[-1] + '/rewards/'
                    self.episode_num = len(os.listdir(self.rewards_folder))
                    self.init_rewards_dir = True
                    rospy.loginfo(self.rewards_folder)

            

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.set_rewards_folder()
            rate.sleep()

if __name__ == '__main__':
    reward = Reward()