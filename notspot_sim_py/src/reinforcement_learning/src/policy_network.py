#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
from cv_bridge import CvBridge
import torch


def ros_image_to_pytorch_tensor(ros_image):
    # Initialize the CvBridge
    bridge = CvBridge()
    
    # Convert the ROS Image message to a CV2 image (numpy array)
    cv_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')
    
    # Optionally, convert the image color (e.g., if the model expects RGB but the image is BGR)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Convert the numpy array to a PyTorch tensor
    tensor_image = torch.from_numpy(cv_image)
    
    # If the image is not grayscale, permute the dimensions from HWC to CHW
    if len(tensor_image.shape) == 3:
        tensor_image = tensor_image.permute(2, 0, 1)
    
    # Add a batch dimension with unsqueeze if necessary
    tensor_image = tensor_image.unsqueeze(0)
    
    # Normalize the tensor to [0, 1] if your model expects that
    tensor_image = tensor_image.float() / 255.0
    
    return tensor_image


class PolicyNetwork:
    def __init__(self):
        rospy.init_node('policy_network_node', anonymous=True)
        # Load hyperparameters from YAML file
        self.hyperparameters = rospy.get_param('/policy_network/hyperparameters', default={})[0]
        if not self.hyperparameters['load_model']:
            rospy.set_param('/RL/runs/new_run', True)
        else:
            rospy.set_param('/RL/runs/new_run', False)

        rospy.Subscriber('/realsense/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/occupancy_grid', OccupancyGrid, self.occupancy_grid_callback)

        self.image_tensor = Image()
        self.occupancy_grid = np.empty(100)

        # Run the main loop
        self.main_loop()

    def main_loop(self):
        rospy.spin()


    def image_callback(self, img: Image):
        self.image_tensor = ros_image_to_pytorch_tensor(img)

    def occupancy_grid_callback(self, occupancy_grid: OccupancyGrid):
        self.occupancy_grid = np.array(occupancy_grid.data).reshape((100,100))
        self.occupancy_grid = torch.from_numpy(self.occupancy_grid).float()


if __name__ == '__main__':
    model = PolicyNetwork() 