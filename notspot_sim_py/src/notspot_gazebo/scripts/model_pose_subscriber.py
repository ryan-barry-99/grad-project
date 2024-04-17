#!/usr/bin/env python3

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped

class ModelPoseSubscriber:
    def __init__(self):
        rospy.init_node('model_pose_node', anonymous=True)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_states_callback)
        params = rospy.get_param('~')
        self.models = params['models']
        self.publishers = {}
        self.run()

    def model_states_callback(self, data):
        for model in self.models:
            model_name = model['model_name']
            namespace = model['namespace']
            tag = model['tag']
            if model_name in data.name and data.name.index(model_name) < len(data.pose):
                index = data.name.index(model_name)
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = rospy.Time.now()
                pose_stamped.header.frame_id = 'map'  # Change 'map' to your desired frame
                pose_stamped.pose = data.pose[index]
                topic = f'/gazebo/model_poses/{namespace}/{tag}'
                if topic not in self.publishers:
                    self.publishers[topic] = rospy.Publisher(topic, PoseStamped, queue_size=10)
                self.publishers[topic].publish(pose_stamped)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    model_pose_subscriber = ModelPoseSubscriber()
