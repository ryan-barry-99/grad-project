#!/usr/bin/env python3

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped, Point32, Twist
from reinforcement_learning.msg import Heuristic
from std_msgs.msg import Float32, Bool, String
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointCloud
import numpy as np

NUM_POSES_TRACKED = 60
NOT_MOVING_POSES = 2

class StateMachine:
    def __init__(self):
        rospy.init_node("state_machine")
        self.states = ["hits_wall", "reach_goal", "not_moving", "stuck"]
        self.reward_pub = rospy.Publisher('/RL/states/reward', String, queue_size=10)

        rospy.Subscriber('/heuristic/goal/0', Heuristic, self.goal_callback)
        self.atGoal = False
        self.init_goal_dist = None
        self.goal_dist = Heuristic()
        self.min_goal_dist = np.inf

        rospy.Subscriber('/gazebo/model_poses/robot/notspot', PoseStamped, self.robot_pose_callback)
        self.init_pos = None
        self.max_start_dist = 0

        self.timer = rospy.Timer(rospy.Duration(0.75), self.timer_callback, reset=True)
        self.pc_pub = rospy.Publisher('velodyne_points_xyz', PointCloud, queue_size=10)
        self.pc_rs_pub = rospy.Publisher('realsense_points_xyz', PointCloud, queue_size=10)
        self.step_pub = rospy.Publisher('/RL/step', Bool, queue_size=1)
        self.dist_pub = rospy.Publisher('/RL/reward/dist', Float32, queue_size=10)
        self.start_dist_pub = rospy.Publisher('RL/reward/start_dist', Float32, queue_size=10)
        
        rospy.Subscriber("/cmd_vel", Twist, self.velo_callback)
        rospy.Subscriber('/RL/episode/new', Bool, self.new_episode_callback)

        self.poses_tracked = []
        self.newStep = False
        self.checking = False
        self.point_cloud = PointCloud()
        self.velo = Twist()

        self.wall_indices = [
            [0.92, 1.125],
            [0.92, -1.125],
            [-10.33, -1.125],
            [-10.33, 4.125],
            [-2.83, 4.125],
            [-2.83, 1.125],
            [-4.59, 2.48],
            [-4.59, 0.98],
            [-8.34, 0.98],
            [-8.34, 2.48]
        ]

        rospy.spin()

    def timer_callback(self, event):
        self.step_pub.publish(True)
        self.newStep = True
        self.checking = True
        self.check_movement()


    def goal_callback(self, msg: Heuristic):
        self.goal_dist = msg
        if self.init_goal_dist is None:
            self.init_goal_dist = msg.manhattan_distance
        if not self.atGoal and msg.manhattan_distance < 1:
            self.atGoal = True
            self.reward_pub.publish("reach_goal")
        elif msg.manhattan_distance > 1:
            self.atGoal = False

    def robot_pose_callback(self, msg: PoseStamped):
        if self.newStep:
            if self.init_pos is None:
                self.init_pos = msg.pose.position
            self.poses_tracked.append(msg)
            if len(self.poses_tracked) > NUM_POSES_TRACKED:
                self.poses_tracked.pop(0)
            self.newStep = False

    def velo_callback(self, msg: Twist):
        self.velo = msg

    def check_movement(self):
        if self.checking:
            if len(self.poses_tracked) >= NOT_MOVING_POSES:
                old_pos = self.poses_tracked[-NOT_MOVING_POSES].pose.position
                new_pos = self.poses_tracked[-1].pose.position
                new_orient = self.poses_tracked[-1].pose.orientation

                self.check_wall()

                dist = self.calc_dist(old_pos, new_pos)
                # if self.init_goal_dist is not None:
                #     rospy.loginfo(f"{self.old_goal_dist.manhattan_distance < self.goal_dist.manhattan_distance}")
                # rospy.loginfo(f"old: {self.old_goal_dist}\nnew: {self.goal_dist}")
                # rospy.loginfo(f"{type(self.old_goal_dist.manhattan_distance)},  {type(self.goal_dist.manhattan_distance)}")
                if self.init_goal_dist is not None and self.goal_dist.manhattan_distance < self.min_goal_dist:
                    self.min_goal_dist = self.goal_dist.manhattan_distance
                    self.reward_pub.publish("moving_forwards")
                self.old_goal_dist = self.goal_dist

                if dist < 0.75:
                    self.dist_pub.publish(dist)
                    if abs(new_orient.x) < 0.1 and abs(new_orient.y) < 0.1 and len(self.poses_tracked) >= NOT_MOVING_POSES:
                        self.reward_pub.publish("upright")
                    self.reward_pub.publish("not_moving")

                if abs(new_orient.x) > 0.15 or abs(new_orient.y) > 0.15:
                    self.reward_pub.publish("fell")
            
                old_pos = self.poses_tracked[-2].pose.position
                dist = self.calc_dist(old_pos, new_pos)
                start_dist = self.calc_manhattan_dist(self.init_pos, new_pos)
                if  start_dist > self.max_start_dist:
                    self.max_start_dist = start_dist
                    # if self.init_goal_dist is not None and start_dist < self.init_goal_dist:
                    #     self.reward_pub.publish("moving_forwards")
                self.start_dist_pub.publish(self.max_start_dist)

                # if self.velo.linear.x <= 0 or dist <= 0.05:
                #     self.reward_pub.publish("moving_backwards")

            if len(self.poses_tracked) >= NUM_POSES_TRACKED:
                old_pos = self.poses_tracked[0].pose.position

                dist = self.calc_dist(old_pos, new_pos)

                if dist < 0.25:
                    self.reward_pub.publish("stuck")
                    self.poses_tracked = []


            self.checking = False

    def new_episode_callback(self, msg):
        self.max_start_dist = 0
        self.min_goal_dist = np.inf


    def calc_dist(self, pos1, pos2):
        return ((pos2.x - pos1.x)**2 + (pos2.y - pos1.y)**2)**(1/2)
    
    def calc_manhattan_dist(self, pos1, pos2):
        return abs(pos2.x - pos1.x) + abs(pos2.y - pos1.y)
    
    def check_wall(self):
        pos = self.poses_tracked[-1].pose.position
        x, y = pos.x, pos.y
        thres = 0.5
        if y < -1.125 + thres or x < -10.33 + thres or y > 4.125 - thres:
            self.reward_pub.publish("hits_wall")
            return
        if x > -2.83 - thres:
            if x > 0.92 - thres or y > 1.125 - thres:
                self.reward_pub.publish("hits_wall")
                return
        if y < 2.48 + thres and y > 0.98 - thres and x < -4.59 + thres and x > -8.34 - thres:
            self.reward_pub.publish("hits_wall")
            return



if __name__ == "__main__":
    stateMachine = StateMachine()
        