#!/usr/bin/env python3

import rospy
import rospkg
from geometry_msgs.msg import PoseStamped
from reinforcement_learning.msg import Heuristic
from std_msgs.msg import Float32, Bool, String

class StateMachine:
    def __init__(self):
        rospy.init_node("state_machine")
        self.states = ["hits_wall", "reach_goal", "not_moving", "stuck"]
        self.reward_pub = rospy.Publisher('/RL/states/reward', String, queue_size=10)

        rospy.Subscriber('/heuristic/goal/0', Heuristic, self.goal_callback)
        self.atGoal = False

        rospy.Subscriber('/gazebo/model_poses/robot/notspot', PoseStamped, self.robot_pose_callback)
        self.poses_tracked = []
        self.newStep = False

        rospy.spin()

    def goal_callback(self, msg: Heuristic):
        if not self.atGoal and msg.manhattan_distance < 3:
            self.atGoal = True
            self.reward_pub.publish("reach_goal")
        elif msg.manhattan_distance > 3:
            self.atGoal = False

    def robot_pose_callback(self, msg: PoseStamped):
        if self.newStep:
            self.poses_tracked.append(msg)
            if len(self.poses_tracked) > 100:
                self.poses_tracked.pop(0)
            self.newStep = False

    def new_step_callback(self, msg):
        old_pos = self.poses_tracked[70].pose.position
        new_pos = self.poses_tracked[-1].pose.position

        dist = ((old_pos.x - new_pos.x)**2 + (old_pos.y - new_pos.y)**2)**(1/2)

        if dist < 1:
            self.reward_pub.publish("not_moving")

        old_pos = self.poses_tracked[0].pose.position

        dist = ((old_pos.x - new_pos.x)**2 + (old_pos.y - new_pos.y)**2)**(1/2)

        if dist < 1:
            self.reward_pub.publish("stuck")

        


if __name__ == "__main__":
    stateMachine = StateMachine()
        