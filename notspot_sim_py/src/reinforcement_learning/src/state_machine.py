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
NOT_MOVING_POSES = 4

class StateMachine:
    def __init__(self):
        rospy.init_node("state_machine")
        self.states = ["hits_wall", "reach_goal", "not_moving", "stuck"]
        self.reward_pub = rospy.Publisher('/RL/states/reward', String, queue_size=10)

        rospy.Subscriber('/heuristic/goal/0', Heuristic, self.goal_callback)
        self.atGoal = False

        rospy.Subscriber('/gazebo/model_poses/robot/notspot', PoseStamped, self.robot_pose_callback)

        self.timer = rospy.Timer(rospy.Duration(0.75), self.timer_callback, reset=True)
        self.pc_pub = rospy.Publisher('velodyne_points_xyz', PointCloud, queue_size=10)
        self.pc_rs_pub = rospy.Publisher('realsense_points_xyz', PointCloud, queue_size=10)
        self.step_pub = rospy.Publisher('/RL/step', Bool, queue_size=1)
        self.dist_pub = rospy.Publisher('/RL/reward/dist', Float32, queue_size=10)
        
        rospy.Subscriber("/cmd_vel", Twist, self.velo_callback)

        self.poses_tracked = []
        self.newStep = False
        self.checking = False
        self.point_cloud = PointCloud()
        self.velo = Twist()

        rospy.spin()

    def timer_callback(self, event):
        self.step_pub.publish(True)
        self.newStep = True
        self.checking = True
        self.check_movement()


    def goal_callback(self, msg: Heuristic):
        if not self.atGoal and msg.manhattan_distance < 3:
            self.atGoal = True
            self.reward_pub.publish("reach_goal")
        elif msg.manhattan_distance > 3:
            self.atGoal = False

    def robot_pose_callback(self, msg: PoseStamped):
        if self.newStep:
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

                dist = ((old_pos.x - new_pos.x)**2 + (old_pos.y - new_pos.y)**2)**(1/2)

                if dist < 0.5:
                    self.reward_pub.publish("not_moving")
                    self.dist_pub.publish(dist)
                    if abs(new_orient.x) < 0.1 and abs(new_orient.y) < 0.1 and len(self.poses_tracked) >= NOT_MOVING_POSES:
                        self.reward_pub.publish("upright")

                if abs(new_orient.x) > 0.15 or abs(new_orient.y) > 0.15:
                    self.reward_pub.publish("fell")
            
                old_pos = self.poses_tracked[-2].pose.position
                dist = ((old_pos.x - new_pos.x)**2 + (old_pos.y - new_pos.y)**2)**(1/2)
                if self.velo.linear.x > 0 and dist > 0.05:
                    self.reward_pub.publish("moving_forwards")
                else:
                    self.reward_pub.publish("moving_backward")

            if len(self.poses_tracked) >= NUM_POSES_TRACKED:
                old_pos = self.poses_tracked[0].pose.position

                dist = ((old_pos.x - new_pos.x)**2 + (old_pos.y - new_pos.y)**2)**(1/2)

                if dist < 0.5:
                    self.reward_pub.publish("stuck")
                    self.poses_tracked = []


            self.checking = False





if __name__ == "__main__":
    stateMachine = StateMachine()
        