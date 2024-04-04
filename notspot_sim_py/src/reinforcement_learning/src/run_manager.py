#!/usr/bin/env python3

import rospy
import rospkg
import os
from datetime import datetime

class RunsManager:
    def __init__(self):
        self.rospack = rospkg.RosPack()
        self.rewards_folder = self.rospack.get_path('reinforcement_learning') + '/runs'

        self.init_rewards_dir = False
        self.set_runs_folder()


    def set_runs_folder(self):
        while not self.init_rewards_dir:
            if rospy.has_param('/RL/runs/new_run'):
                if rospy.get_param('/RL/runs/new_run'):
                    c = datetime.now()
                    current_time = c.strftime('%Y_%m_%d_%H_%M_%S')
                    self.runs_folder = f"{self.rewards_folder}/{current_time}"
                    os.mkdir(f"{self.runs_folder}")
                    self.rewards_folder = f"{self.runs_folder}/rewards"
                    os.mkdir(self.rewards_folder)
                    self.init_rewards_dir = True
                else:
                    # List all folders in the directory
                    folders = os.listdir(self.rewards_folder)

                    # Filter out non-folder items and sort by name (timestamp)
                    folders = sorted(folder for folder in folders if os.path.isdir(os.path.join(self.rewards_folder, folder)))
                    
                    # Select the most recent folder (the last one after sorting)
                    if folders:
                        self.runs_folder = self.rospack.get_path('reinforcement_learning') + '/runs/' + folders[-1]
                    self.rewards_folder = self.rewards_folder + '/rewards/'
                    self.init_rewards_dir = True

                rospy.set_param('/RL/runs/runs_folder', self.runs_folder)
                rospy.set_param('/RL/runs/rewards_folder', self.rewards_folder)


if __name__ == '__main__':
    runs_manager = RunsManager()