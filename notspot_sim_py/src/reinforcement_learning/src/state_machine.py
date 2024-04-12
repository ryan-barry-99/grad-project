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
        self.terminal_pub = rospy.Publisher('RL/episode/New', Bool, queue_size=10)
        rospy.spin()


if __name__ == "__main__":
    stateMachine = StateMachine()
        