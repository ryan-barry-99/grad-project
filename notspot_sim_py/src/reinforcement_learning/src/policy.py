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
from std_msgs.msg import String, Bool
from policy_network import PolicyNetwork
from functools import partial
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu



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

        self.model.to(self.device)

        rospy.Subscriber('/realsense/color/image_raw', Image, self.rgb_image_callback)
        rospy.Subscriber('/realsense/depth/image_raw', Image, self.depth_image_callback)
        rospy.Subscriber('/occupancy_grid/velodyne', OccupancyGrid, self.occupancy_grid_callback)
        rospy.Subscriber('/gazebo/model_poses/robot/notspot', PoseStamped, self.pose_callback)
        rospy.Subscriber('/notspot_imu/base_link_orientation', Imu, self.imu_callback)
        rospy.Subscriber('/AI_Control', Bool, self.control_type_callback)

        num_goals = 1
        for i in range(num_goals):
            rospy.Subscriber(f'heuristic/goal/{i}', Heuristic, partial(self.heuristic_callback, goal=i))


        
        self.new_data = [False, False, False, False]
        self.heuristic_tensor = torch.zeros(num_goals, 3)  # Initialize with zeros
        self.image_pub = rospy.Publisher("/occupancy_image", Image, queue_size=10)
        self.bridge = CvBridge()

        self.velo_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.gait_pub = rospy.Publisher("/gait", String, queue_size=20)
        self.gait = "trot"
        self.ai_controls = True

        # Run the main loop
        self.main_loop()

    def main_loop(self):
        while not rospy.is_shutdown():
            self.gait_pub.publish(self.gait)
            if self.ai_controls:
                self.predict_velocity()

    def predict_velocity(self):
        if all(self.new_data):
            self.new_data = [False, False, False, False]
            self.model.eval() # Set model to evaluation mode

            rgb = self.rgb_tensor.to(self.device)                # 480, 640, 3
            depth = self.depth_tensor.to(self.device)            # 720, 1280
            occupancy_grid = self.occupancy_grid.to(self.device) # 100, 100
            heuristic = self.heuristic_tensor.to(self.device)    # 1, 3
            position = self.position_tensor.to(self.device)      # 1, 2
            orientation = self.orientation_tensor.to(self.device)# 1, 4
            ang_vel = self.ang_vel_tensor.to(self.device)        # 1, 3
            lin_acc = self.lin_acc_tensor.to(self.device)        # 1, 3

            with torch.no_grad():
                pred = self.model(rgb, depth, occupancy_grid, heuristic, position, orientation, ang_vel, lin_acc).to('cpu').squeeze().tolist()
            velo = Twist()
            velo.linear.x = pred[0]
            velo.linear.y = pred[1]
            velo.angular.z = pred[2]
            rospy.loginfo(f"Predicted Velocity: {velo}")
            self.velo_pub.publish(velo)
            return velo

    def rgb_image_callback(self, img: Image):
        # Tensor of size ([480, 640, 3])
        self.rgb_tensor = ros_image_to_pytorch_tensor(img).permute(2,0,1)
        self.new_data[0] = True
        
    def depth_image_callback(self, img: Image):
        # Tensor of size ([720, 1280])
        self.depth_tensor = ros_image_to_pytorch_tensor(img).unsqueeze(0)
        self.new_data[1] = True

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
        grid = np.array(occupancy_grid.data).reshape((100,100))

        # Tensor of size ([100, 100])
        self.occupancy_grid = torch.from_numpy(grid).float().unsqueeze(0)
        self.new_data[2] = True
        # rospy.loginfo(f"occupancy grid tensor of size {self.occupancy_grid.size()}")


    def heuristic_callback(self, msg, goal):
        self.heuristic_tensor[goal] = torch.tensor([msg.x_distance, msg.y_distance, msg.manhattan_distance])
        self.new_data[3] = True

    def pose_callback(self, msg):
        position = msg.pose.position
        self.position_tensor = torch.tensor([[position.x, position.y]])

    def imu_callback(self, msg):
        orient = msg.orientation
        self.orientation_tensor = torch.tensor([[orient.x, orient.y, orient.z, orient.w]])

        ang_vel = msg.angular_velocity
        self.ang_vel_tensor = torch.tensor([[ang_vel.x, ang_vel.y, ang_vel.z]])

        lin_acc = msg.linear_acceleration
        self.lin_acc_tensor = torch.tensor([[lin_acc.x, lin_acc.y, lin_acc.z]])

    def control_type_callback(self, msg):
        self.ai_controls = msg.data

    
    


if __name__ == '__main__':
    model = Policy() 