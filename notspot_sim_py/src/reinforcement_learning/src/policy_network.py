#!/usr/bin/env python3

import rospy

class PolicyNetwork:
    def __init__(self):
        rospy.init_node('policy_network_node', anonymous=True)
        # Load hyperparameters from YAML file
        self.hyperparameters = rospy.get_param('/policy_network/hyperparameters', default={})[0]
        if self.hyperparameters['new_run']:
            rospy.set_param('/RL/runs/new_run', True)
        else:
            rospy.set_param('/RL/runs/new_run', False)

        rospy.spin()


if __name__ == '__main__':
    model = PolicyNetwork() 