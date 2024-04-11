#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
from reinforcement_learning.msg import Heuristic
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from policy_network import PolicyNetwork


def ros_image_to_pytorch_tensor(image_msg):
    bridge = CvBridge()
    try:
        # Convert ROS Image message to a NumPy array
        cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    except CvBridgeError as e:
        print(e)
    
    # Assuming cv_image is a numpy.uint16 array from a depth image
    # Convert the numpy array to float32
    cv_image_float = cv_image.astype(np.float32)
    
    # Normalize or scale the depth values if necessary
    # For example, dividing by the maximum sensor range (in mm or meters) can bring the values to a [0, 1] range
    # cv_image_float /= 1000.0  # Example: if the depth values are in millimeters
    
    # Convert the NumPy array (now float32) to a PyTorch tensor
    tensor_image = torch.from_numpy(cv_image_float)
    
    return tensor_image


class Policy:
    def __init__(self):
        rospy.init_node('policy_node', anonymous=True)
        self.model = PolicyNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load hyperparameters from YAML file
        self.hyperparameters = rospy.get_param('/policy_network/hyperparameters', default={})[0]
        if not self.hyperparameters['load_model']:
            rospy.set_param('/RL/runs/new_run', True)
        else:
            rospy.set_param('/RL/runs/new_run', False)
            self.model.load_state_dict(torch.load(self.hyperparameters['model_path']))

        rospy.Subscriber('/realsense/color/image_raw', Image, self.rgb_image_callback)
        rospy.Subscriber('/realsense/depth/image_raw', Image, self.depth_image_callback)
        rospy.Subscriber('/occupancy_grid/velodyne', OccupancyGrid, self.occupancy_grid_callback)
        rospy.Subscriber('heuristic/goal/closest', Heuristic, self.heuristic_callback)

        self.rgb_image_tensor = Image()
        self.depth_image_tensor = Image()
        self.occupancy_grid = np.empty(100)
        self.heuristic = Heuristic()
        self.image_pub = rospy.Publisher("/occupancy_image", Image, queue_size=10)
        self.bridge = CvBridge()

        self.velo_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.gait_pub = rospy.Publisher("/gait", String, queue_size=20)
        self.gait = "trot"

        # Run the main loop
        self.main_loop()

    def main_loop(self):
        while not rospy.is_shutdown():
            self.gait_pub.publish(self.gait)



    def rgb_image_callback(self, img: Image):
        # Tensor of size ([480, 640, 3])
        self.rgb_image_tensor = ros_image_to_pytorch_tensor(img)
        rospy.loginfo(f"image tensor of size {self.rgb_image_tensor.size()}")
        
    def depth_image_callback(self, img: Image):
        # Tensor of size ([720, 1280])
        self.depth_image_tensor = ros_image_to_pytorch_tensor(img)
        rospy.loginfo(f"depth image tensor of size {self.depth_image_tensor.size()}")

    def occupancy_grid_callback(self, occupancy_grid: OccupancyGrid):
        occupancy_grid_np = np.array(occupancy_grid.data).reshape((100, 100))
        
        # Map values: 100 to 0 (black) and -1 to 255 (white)
        # First, normalize -1 to 1, then invert (1 to 0, 0 to 1), finally scale to 255
        image_np = np.interp(occupancy_grid_np, [-1, 100], [1, 0])
        image_np = (image_np * 255).astype(np.uint8)
        
        # Convert numpy array to ROS Image message
        image_msg = self.bridge.cv2_to_imgmsg(image_np, encoding="mono8")
        
        # Publish the image
        self.image_pub.publish(image_msg)
        self.occupancy_grid = np.array(occupancy_grid.data).reshape((100,100))

        # Tensor of size ([100, 100])
        self.occupancy_grid = torch.from_numpy(self.occupancy_grid).float()
        rospy.loginfo(f"occupancy grid tensor of size {self.occupancy_grid.size()}")


    def heuristic_callback(self, msg: Heuristic):
        self.heuristic = msg

    


if __name__ == '__main__':
    model = Policy() 