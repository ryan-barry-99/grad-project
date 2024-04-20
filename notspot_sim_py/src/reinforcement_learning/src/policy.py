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
from std_msgs.msg import String, Bool, Float32
from navigation_networks import PolicyNetwork, ValueNetwork
from functools import partial
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from rl_util import *
from experience_buffer import ExperienceBuffer
import torch.nn.functional as F


class ProximalPolicyOptimization:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('policy_node', anonymous=True)
        
        # Load hyperparameters from YAML file
        self.hyperparameters = rospy.get_param('/policy_network/hyperparameters', default={})
        
        # Check if a model should be loaded
        load_model = self.hyperparameters.get('load_model', False)
        if not load_model:
            rospy.set_param('/RL/runs/new_run', True)
        else:
            rospy.set_param('/RL/runs/new_run', False)
            self.policy.load_state_dict(torch.load(self.hyperparameters['policy_model_path']))
            self.value.load_state_dict(torch.load(self.hyperparameters['value_model_path']))

        # Check for cuda and set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize policy and value networks
        self.policy_network = PolicyNetwork()
        self.policy_optimizer = set_optimizer(
            model=self.policy_network, 
            optimizer=self.hyperparameters["optimizer"],
            lr=self.hyperparameters["lr_policy"]
            )
        self.policy_network.to(self.device)
        
        self.value_network = ValueNetwork()
        self.value_optimizer = set_optimizer(
            model=self.value_network, 
            optimizer=self.hyperparameters["optimizer"],
            lr=self.hyperparameters["lr_value"]
            )
        self.value_network.to(self.device)
        
        # Initialize ROS publishers
        self.image_pub = rospy.Publisher("/occupancy_image", Image, queue_size=10)
        self.velo_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.gait_pub = rospy.Publisher("/gait", String, queue_size=10)
        
        # Initialize other variables
        self.bridge = CvBridge()
        self.gait = "trot"
        self.ai_controls = True
        self.rewards = 0
        self.new_episode = True
        self.models_folder = False
        self.experiences = {}
        self.buffer = ExperienceBuffer(
            batch_size=self.hyperparameters.get("batch_size", 32),
            max_trajectory_length=self.hyperparameters.get("max_trajectory_length", 1000)
        )

         # Initialize ROS subscribers
        num_goals = 1
        for i in range(num_goals):
            rospy.Subscriber(f'heuristic/goal/{i}', Heuristic, partial(self.heuristic_callback, goal=i))
        rospy.Subscriber('/realsense/color/image_raw', Image, self.rgb_image_callback)
        rospy.Subscriber('/realsense/depth/image_raw', Image, self.depth_image_callback)
        rospy.Subscriber('/occupancy_grid/velodyne', OccupancyGrid, self.occupancy_grid_callback)
        rospy.Subscriber('/gazebo/model_poses/robot/notspot', PoseStamped, self.pose_callback)
        rospy.Subscriber('/notspot_imu/base_link_orientation', Imu, self.imu_callback)
        rospy.Subscriber('/AI_Control', Bool, self.control_type_callback)
        rospy.Subscriber('/RL/step', Bool, self.step_callback)
        rospy.Subscriber('/RL/reward/action', Float32, self.update_reward)
        rospy.Subscriber('/RL/episode/new', Bool, self.new_episode_callback)
        rospy.Subscriber('/RL/model/save', String, self.save_model_callback)

        # Initialize the heuristic tensor
        self.heuristic_tensor = torch.zeros(num_goals, 3)  # Initialize with zeros
        
        # Start ROS node
        rospy.spin()


    def step_callback(self, msg: Bool):
        if msg.data:
            try:
                if self.new_episode:
                    self.steps_since_update = 0
                    self.new_episode = False
                    if self.buffer.length >= self.hyperparameters["batch_size"]:
                        self.update_networks()
                        self.buffer.clear()
                        if not self.models_folder:
                            if rospy.has_param('/RL/runs/models_folder'):
                                self.models_folder = rospy.get_param('/RL/runs/models_folder')
                                rospy.loginfo("saving models....")
                                save_model(self.policy_network, f"{self.models_folder}/latest_policy.pt")
                                save_model(self.value_network, f"{self.models_folder}/latest_value.pt")
                        else:
                            rospy.loginfo("saving models....2")
                            save_model(self.policy_network, f"{self.models_folder}/latest_policy.pt")
                            save_model(self.value_network, f"{self.models_folder}/latest_value.pt")
                self.steps_since_update += 1
                if self.steps_since_update >= 25:  
                    self.experiences = {
                            "rgb": self.rgb_tensor.to(self.device),                # 480, 640, 3
                            "depth":self.depth_tensor.to(self.device),             # 720, 1280
                            "occupancy_grid": self.occupancy_grid.to(self.device), # 100, 100
                            "heuristic": self.heuristic_tensor.to(self.device),    # 1, 3
                            "position": self.position_tensor.to(self.device),      # 1, 2
                            "orientation": self.orientation_tensor.to(self.device),# 1, 4
                            "ang_vel": self.ang_vel_tensor.to(self.device),        # 1, 3
                            "lin_acc": self.lin_acc_tensor.to(self.device),        # 1, 3
                        }
                    
                    mean, std = self.policy(self.experiences)
                    velocity_vector, self.log_probs = self.sample_velocity(mean, std)
                    value = self.value(self.experiences, velocity_vector)
                    self.buffer.store(state=self.experiences, action=velocity_vector, reward=self.rewards, log_prob = self.log_probs, value=value)
                    
                    
                    velocity_vector = velocity_vector.to('cpu').squeeze().tolist()
                    self.publish_velocity(velocity_vector)

                    if self.buffer.at_capacity():
                        self.update_networks()
                        self.buffer.clear()
                else:
                    self.publish_velocity([0,0,0])
            except:
                pass

        else:
            self.rewards = 0


            
    def policy(self, experience):
        """
        Use the policy network to predict velocity distributions of a state
        """
        self.policy_network.eval() # Set model to evaluation mode
        with torch.no_grad():
            mean, std = self.policy_network(experience)
        return mean, std
    
    def value(self, experience, action):
        """
        Use the value network to estimate the value of a state, action pair
        """
        self.value_network.eval()
        with torch.no_grad():
            value = self.value_network(experience, action)
        return value
    
    def publish_velocity(self, velocity_vector):
        self.gait_pub.publish(self.gait)
        velo = Twist()
        if not self.new_episode:
            velo.linear.x = velocity_vector[0]
            velo.linear.y = velocity_vector[1]
            velo.angular.z = velocity_vector[2]
        else:
            if len(self.buffer.states) > 25:
                self.new_episode = False
        self.velo_pub.publish(velo)
        

    def sample_velocity(self, mean, std):
        dist = torch.distributions.Normal(mean, std)
        velocity_vector = dist.sample()
        log_probs = dist.log_prob(velocity_vector).sum(axis=-1)  # Sum log probs for total log prob of the velocity vector
        return velocity_vector, log_probs
    
    def advantage(self, rewards: list, values: list):
        """
        Compute advantages for each timestep in the trajectory.
        
        Args:
        - rewards (list): List/array of actual rewards obtained.
        - values (list): List/array of predicted values estimated by the Critic network.
        
        Returns:
        - advantages (numpy array): Array of advantages for each timestep.
        """
        return np.array(rewards) - np.array(values)
    
    def update_networks(self):
        _, _, rewards, old_log_probs, new_log_probs, values = self.buffer.get_batch()
        advantages = self.advantage(rewards=rewards, values=values)
        
        policy_loss = self.compute_policy_loss(old_log_probs=old_log_probs, new_log_probs=new_log_probs, advantages=advantages)
        value_loss = self.compute_value_loss(values=values, rewards=rewards)

        # Backpropagation and optimization for policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Backpropagation and optimization for value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


    def compute_policy_loss(self, old_log_probs, new_log_probs, advantages):
        clip_epsilon = self.hyperparameters["ppo_clip"]

        ratio = torch.exp(new_log_probs / old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # Add entropy regularization term
        entropy = -(new_log_probs.exp() * new_log_probs).sum(dim=1).mean()  # Compute entropy of the action distribution
        policy_loss -= self.hyperparameters["entropy_coef"] * entropy  # Add entropy regularization to the policy loss
        return policy_loss
    
    def compute_value_loss(self, values, rewards):
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(rewards):
            running_reward = reward + self.hyperparameters["gamma"] * running_reward
            discounted_rewards.insert(0, running_reward)

        values = torch.tensor(values)
        discounted_rewards = torch.tensor(discounted_rewards)
        value_loss = F.mse_loss(values, discounted_rewards)
        return value_loss


    def rgb_image_callback(self, img: Image):
        # Tensor of size ([480, 640, 3])
        self.rgb_tensor = ros_image_to_pytorch_tensor(img).permute(2,0,1)
        
    def depth_image_callback(self, img: Image):
        # Tensor of size ([720, 1280])
        self.depth_tensor = ros_image_to_pytorch_tensor(img).unsqueeze(0)

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


    def heuristic_callback(self, msg, goal: Heuristic):
        self.heuristic_tensor[goal] = torch.tensor([msg.x_distance, msg.y_distance, msg.manhattan_distance])

    def pose_callback(self, msg: PoseStamped):
        position = msg.pose.position
        self.position_tensor = torch.tensor([[position.x, position.y]])

    def imu_callback(self, msg: Imu):
        orient = msg.orientation
        self.orientation_tensor = torch.tensor([[orient.x, orient.y, orient.z, orient.w]])

        ang_vel = msg.angular_velocity
        self.ang_vel_tensor = torch.tensor([[ang_vel.x, ang_vel.y, ang_vel.z]])

        lin_acc = msg.linear_acceleration
        self.lin_acc_tensor = torch.tensor([[lin_acc.x, lin_acc.y, lin_acc.z]])

    def control_type_callback(self, msg: Bool):
        self.ai_controls = msg.data

    
    def update_reward(self, msg: Float32):
        self.rewards += msg.data

    def new_episode_callback(self, msg):
        self.velo_pub.publish(Twist())
        self.new_episode = True
        

    def save_model_callback(self, msg: String):
        save_model(self.policy_network, f"{self.models_folder}/policy_{msg.data}")
        save_model(self.value_network, f"{self.models_folder}/value_{msg.data}")


if __name__ == '__main__':
    ppo = ProximalPolicyOptimization() 