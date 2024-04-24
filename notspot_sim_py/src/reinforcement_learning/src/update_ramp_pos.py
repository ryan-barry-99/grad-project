#!/usr/bin/env python3

import rospy
from gazebo_msgs.srv import SetWorldProperties

def load_world(world_name):
    rospy.wait_for_service('/gazebo/set_world_properties')
    try:
        set_world_properties = rospy.ServiceProxy('/gazebo/set_world_properties', SetWorldProperties)
        set_world_properties(world_name, True)  # Pass the world name and reset parameter
        rospy.loginfo("Loaded world: {}".format(world_name))
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: {}".format(e))



# Define the path to the world file you want to load
world_file = "/media/ryan/DataPad/ros_ws/src/notspot_sim_py/src/notspot_gazebo/launch/world/P2.world"

    
    # rospy.spin()
if __name__ == '__main__':
    rospy.init_node('world_loader')
    load_world(world_file)
