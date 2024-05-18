#!/usr/bin/env python3

import rospy
import rospkg
import os
from datetime import datetime
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist

class RunsManager:
    def __init__(self):
        rospy.init_node("run_manager")

        self.rospack = rospkg.RosPack()
        self.runs_folder = '/media/ryan/Media/Grad_Project/runs'
        self.runs_folder = '/media/ryan/Media/Grad_Project/runs'

        self.init_rewards_dir = False
        self.init_runs_folder = False
        self.set_runs_folder()

        rospy.Subscriber('/RL/episode/new', Bool, self.new_episode_callback)
        self.reset_pub = rospy.Publisher('/robot_reset_command', String, queue_size=10)
        self.velo_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.spin()


    def set_runs_folder(self):
        while not self.init_rewards_dir:
            if rospy.has_param('/RL/runs/new_run'):
                if rospy.get_param('/RL/runs/new_run'):
                    c = datetime.now()
                    current_time = c.strftime('%Y_%m_%d_%H_%M_%S')
                    self.runs_folder = f"{self.runs_folder}/{current_time}"
                    os.mkdir(f"{self.runs_folder}")
                    self.rewards_folder = f"{self.runs_folder}/rewards"
                    os.mkdir(self.rewards_folder)
                    self.models_folder = f"{self.runs_folder}/models"
                    os.mkdir(self.models_folder)
                    self.loss_folder = f"{self.models_folder}/loss"
                    os.mkdir(self.loss_folder)
                    os.mkdir(self.loss_folder + '/policy')
                    os.mkdir(self.loss_folder + '/value')
                    
                    self.steps_folder = f"{self.runs_folder}/steps"
                    os.mkdir(self.steps_folder)
                    self.distance_folder = f"{self.runs_folder}/distance"
                    os.mkdir(self.distance_folder)
                    self.goal_folder = f"{self.runs_folder}/goal"
                    os.mkdir(self.goal_folder)
                    self.environment_folder = f"{self.runs_folder}/environment"
                    os.mkdir(self.environment_folder)
                    self.init_rewards_dir = True
                else:
                    while not self.init_runs_folder:
                        if rospy.has_param('/RL/runs/run_folder'):
                            self.runs_folder = f"{rospy.get_param('/RL/runs/run_folder')}"
                            self.rewards_folder = self.runs_folder + '/rewards'
                            self.init_runs_folder = True
                    # List all folders in the directory
                    folders = os.listdir(self.rewards_folder)

                    # Filter out non-folder items and sort by name (timestamp)
                    folders = sorted(folder for folder in folders if os.path.isdir(os.path.join(self.rewards_folder, folder)))
                    
                    # Select the most recent folder (the last one after sorting)
                    if folders:
                        self.runs_folder = self.rospack.get_path('reinforcement_learning') + '/runs/' + folders[-1]
                    self.rewards_folder = self.runs_folder + '/rewards/'
                    self.models_folder = self.runs_folder + '/models/'
                    self.loss_folder = self.models_folder + '/loss/'
                    self.steps_folder = self.runs_folder + '/steps/'
                    self.goal_folder = self.runs_folder + '/goal/'
                    self.distance_folder = self.runs_folder + '/distance/'
                    self.environment_folder = self.runs_folder + '/environment/'
                    self.init_rewards_dir = True

                rospy.set_param('/RL/runs/runs_folder', self.runs_folder)
                rospy.set_param('/RL/runs/rewards_folder', self.rewards_folder)
                rospy.set_param('/RL/runs/models_folder', self.models_folder)
                rospy.set_param('/RL/runs/loss_folder', self.loss_folder)
                rospy.set_param('/RL/runs/steps_folder', self.steps_folder)
                rospy.set_param('/RL/runs/goals_folder', self.goal_folder)
                rospy.set_param('/RL/runs/distance_folder', self.distance_folder)
                rospy.set_param('/RL/runs/environment_folder', self.environment_folder)

    def new_episode_callback(self, msg: Bool):
        if msg.data:
            self.velo_pub.publish(Twist())
            self.reset_pub.publish("reset")



if __name__ == '__main__':
    runs_manager = RunsManager()