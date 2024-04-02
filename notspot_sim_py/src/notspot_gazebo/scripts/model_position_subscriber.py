#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

class ModelPositionSubscriber:
    def __init__(self):
        rospy.init_node('model_position_node', anonymous=True)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        self.model_name = rospy.get_param('~model_name')
        self.pub = rospy.Publisher(f'/gazebo/model_positions/{self.model_name}', Pose, queue_size=10)
        self.model_position = Pose()
        self.run()

    def model_states_callback(self, data):
        model_index = data.name.index(self.model_name)  # Replace 'your_model_name' with the name of your model in Gazebo
        self.model_position = data.pose[model_index]

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.pub.publish(self.model_position)
            rate.sleep()

if __name__ == '__main__':
    model_position_subscriber = ModelPositionSubscriber()
