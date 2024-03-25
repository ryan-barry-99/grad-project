#!/usr/bin/env python3
# Author: Ryan Barry

import rospy
from math import fabs
from numpy import array_equal

from sensor_msgs.msg import Joy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class EmulateJoystick(object):
    def __init__(self, rate):
        rospy.init_node("joystick_emulator")
        rospy.Subscriber("gait", String, self.set_gait)
        rospy.Subscriber("cmd_vel", Twist, self.set_velocity)
        rospy.Subscriber("robot_spawn_command", String, self.robot_spawner)
        self.publisher = rospy.Publisher("notspot_joy/joy_ramped", Joy, queue_size=10)
        self._gait = "stand"
        self._velocity = Twist()
        self.rate = rospy.Rate(rate)
        self.joy = Joy()
        self.joy.axes = [0.,0.,0.,0.,0.,0.,0.,0.]
        self.joy.buttons = [0,0,0,0,0,0,0,0,0,0,0]
        self.robot_exists = False
        
        
    def set_gait(self, gait):
        self._gait = gait.data
        self.joy.buttons = [0,0,0,0,0,0,0,0,0,0,0]  # Reset all buttons
        
        # Update buttons based on gait
        if self._gait == "rest":
            self.joy.buttons[0] = 1
        elif self._gait == "trot":
            self.joy.buttons[1] = 1
        elif self._gait == "crawl":
            self.joy.buttons[2] = 1
        elif self._gait == "stand":
            self.joy.buttons[3] = 1
        else:
            rospy.logwarn("Unknown gait: %s" % self._gait)
            
    def set_velocity(self, velocity):
        self._velocity = velocity
        
        self.joy.axes[4] = self._velocity.linear.x
        self.joy.axes[3] = self._velocity.linear.y
        self.joy.axes[0] = self._velocity.angular.z
        
    def robot_spawner(self, command):
        if command.data == "spawn":
            self.robot_exists = True
        elif command.data == "remove":
            self.robot_exists = False
        
            
            
    def run(self):
        while not rospy.is_shutdown():
            if self.robot_exists:
                self.publisher.publish(self.joy)
                self.rate.sleep()
        


if __name__ == "__main__":
    joystick = EmulateJoystick(10)
    joystick.run()
