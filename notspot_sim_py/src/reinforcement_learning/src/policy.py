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
from std_msgs.msg import String, Bool, Float32, Int32
from navigation_networks import PolicyNetwork
from functools import partial
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import Imu
from rl_util import *
from experience_buffer import ExperienceBuffer
import torch.nn.functional as F
import os
import rospkg


class ProximalPolicyOptimization:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('policy_node', anonymous=True)


        self.policy_network = PolicyNetwork()
        # self.value_network = ValueNetwork()
        
        # Load hyperparameters from YAML file
        self.hyperparameters = rospy.get_param('/policy_network/hyperparameters', default={})

        self.rospack = rospkg.RosPack()
        self.runs_folder = self.rospack.get_path('reinforcement_learning') + '/runs/' + self.hyperparameters['run']
        
        # Check if a model should be loaded
        load_model = self.hyperparameters["load_model"]
        if not load_model:
            rospy.set_param('/RL/runs/new_run', True)
        else:
            rospy.loginfo("loading model")
            rospy.set_param('/RL/runs/run_folder', f"{self.runs_folder}")
            rospy.set_param('/RL/runs/new_run', False)
            policy_model_path = f"{self.runs_folder}/models/{self.hyperparameters['policy_model_path']}"
            # value_model_path = f"{self.runs_folder}/models/{self.hyperparameters['value_model_path']}"
            self.policy_network.load_state_dict(torch.load(policy_model_path))
            # self.value_network.load_state_dict(torch.load(value_model_path))

        got_loss_folder = False
        while not got_loss_folder:
            if rospy.has_param('/RL/runs/loss_folder'):
                self.loss_folder = rospy.get_param('/RL/runs/loss_folder')
                self.num_updates = int(len(os.listdir(self.loss_folder))/2)
                got_loss_folder = True

        # Check for cuda and set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize policy and value networks
        self.policy_optimizer = set_optimizer(
            model=self.policy_network, 
            optimizer=self.hyperparameters["optimizer"],
            lr=self.hyperparameters["lr_policy"],
            momentum=self.hyperparameters["momentum"]
            )
        self.policy_network.to(self.device)
        
        # self.value_optimizer = set_optimizer(
        #     model=self.value_network, 
        #     optimizer=self.hyperparameters["optimizer"],
        #     lr=self.hyperparameters["lr_value"]
        #     )
        # self.value_network.to(self.device)
        
        # Initialize ROS publishers
        self.image_pub = rospy.Publisher("/occupancy_image", Image, queue_size=10)
        self.velo_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.gait_pub = rospy.Publisher("/gait", String, queue_size=10)
        self.reset_pub = rospy.Publisher('/robot_reset_command', String, queue_size=10)
        self.pose_pub = rospy.Publisher('/robot_reset_position', Pose, queue_size=10)
        
        # Initialize other variables
        self.bridge = CvBridge()
        self.gait = "trot"
        self.ai_controls = True
        self.rewards = 0
        self.total_rewards = 0
        self.steps_since_update = 0
        self.lifespan = 0
        self.total_distance_traveled = 0
        self.goal = False
        self.new_episode = True
        self.models_folder = False
        self.init_rewards_dir = False
        self.init_steps_dir = False
        self.init_goal_dir = False
        self.init_distance_dir = False
        self.init_environment_dir = False
        self.old_position = None
        self.go = False
        self.dist = np.inf
        self.closest_goal = np.inf
        self.furthest_goal = 0
        self.goal_dist = np.inf
        self.ending_goal = np.inf
        self.max_start_dist = 0
        self.starting_distance = 0
        self.environment = 0
        self.last_known_environment = 0
        self.initialize_episodes()
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
        rospy.Subscriber('/RL/states/reward', String, self.states_callback)
        rospy.Subscriber('/heuristic/goal/0', Heuristic, self.goal0_callback)
        rospy.Subscriber('/heuristic/goal/1', Heuristic, self.goal1_callback)
        rospy.Subscriber('/RL/reward/dist', Float32, self.dist_callback)
        rospy.Subscriber('RL/reward/start_dist', Float32, self.start_dist_callback)
        rospy.Subscriber('/RL/environment', Int32, self.environment_callback)

        # Initialize the heuristic tensor
        self.heuristic_tensor = torch.zeros(num_goals, 3)  # Initialize with zeros
        
        # Start ROS node
        rospy.spin()

    def environment_callback(self, msg):
        self.environment = msg.data

    def step_callback(self, msg: Bool):
        if msg.data:
            if self.new_episode:
                self.go = False
                with open(f'{self.rewards_folder}/episode_{self.episode_num}.txt', 'w') as f:
                    f.write(f"{self.total_rewards}")
                    self.total_rewards = 0
                with open(f'{self.steps_folder}/episode_{self.episode_num}.txt', 'w') as file:
                    file.write(f"{self.lifespan}")
                with open(f'{self.goal_folder}/episode_{self.episode_num}.txt', 'w') as file:
                    if self.ending_goal < self.closest_goal:
                        self.closest_goal = self.ending_goal
                    if self.ending_goal > self.furthest_goal:
                        self.furthest_goal = self.ending_goal
                    file.write(f"{self.closest_goal}\n{self.furthest_goal}\n{self.ending_goal}")
                with open(f'{self.distance_folder}/episode_{self.episode_num}.txt', 'w') as file:
                    file.write(f"{self.total_distance_traveled}\n{self.starting_distance}")
                    self.total_distance_traveled = 0
                    self.starting_distance = 0
                with open(f'{self.environment_folder}/episode_{self.episode_num}.txt', 'w') as file:
                    file.write(f"{self.last_known_environment}")
                    self.last_known_environment = self.environment
                self.episode_num += 1
                rospy.loginfo(f"Starting episode {self.episode_num}")
                if self.buffer.length >= 1:
                    self.lifespan = 0
                    self.new_episode = False
                    self.update_networks(batch_size=self.buffer.length)
                    self.buffer.clear()
                    if not self.models_folder:
                        if rospy.has_param('/RL/runs/models_folder'):
                            self.models_folder = rospy.get_param('/RL/runs/models_folder')
                            save_model(self.policy_network, f"{self.models_folder}/latest_policy.pt")
                            # save_model(self.value_network, f"{self.models_folder}/latest_value.pt")
                    if self.models_folder is not None:
                        save_model(self.policy_network, f"{self.models_folder}/latest_policy.pt")
                        # save_model(self.value_network, f"{self.models_folder}/latest_value.pt")
                self.new_episode = False       
            self.steps_since_update += 1
            if self.steps_since_update >= 4: 
                self.starting_distance = self.max_start_dist 
                self.go = True
                self.ending_goal = self.goal_dist
                if self.steps_since_update == 4:
                    self.pose_pub.publish(Pose())
                self.lifespan += 1
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
                
                mean, std, value = self.policy(self.experiences)
                velocity_vector, self.log_probs = self.sample_velocity(mean, std)
                velocity_vector = torch.tensor(self.prep_velo(velocity_vector), requires_grad=True).to(self.device).unsqueeze(0)
                value = value.cpu().squeeze().tolist()
                # value = self.value(self.experiences, velocity_vector).tolist()
                self.buffer.store(state=self.experiences, action=velocity_vector, reward=self.rewards, log_prob = self.log_probs, value=value)
                self.rewards=0
                
                velocity_vector = velocity_vector.cpu().squeeze().tolist()
                if self.ai_controls:
                    self.publish_velocity(velocity_vector)

                if self.buffer.at_capacity():
                    self.update_networks()
                    self.buffer.clear()
            else:
                self.go = False
                self.closest_goal = np.inf
                self.furthest_goal = 0
                if self.ai_controls:
                    self.publish_velocity([0,0,0])
                self.reset_pub.publish("reset")

        else:
            self.rewards = 0
        self.goal = False


            
    def policy(self, experience):
        """
        Use the policy network to predict velocity distributions of a state
        """
        self.policy_network.eval() # Set model to evaluation mode
        with torch.no_grad():
            mean, std, value = self.policy_network(experience)
        return mean, std, value
    
    # def value(self, experience, action):
    #     """
    #     Use the value network to estimate the value of a state, action pair
    #     """
    #     self.value_network.eval()
    #     with torch.no_grad():
    #         value = self.value_network(experience, action)
    #     return value
    
    def prep_velo(self, velocity_vector):
        # vx = -abs(velocity_vector[0][0]) + abs(velocity_vector[0][1])
        # vy = -abs(velocity_vector[0][2]) + abs(velocity_vector[0][3])
        # # vy = 0
        # vz = -abs(velocity_vector[0][4]) + abs(velocity_vector[0][5])
        def sign(num):
            return (num > 0).float() - (num < 0).float()
        
        vx = sign(velocity_vector[0][0])*min(abs(velocity_vector[0][0]), 2.5)
        vy = sign(velocity_vector[0][1])*min(abs(velocity_vector[0][1]), 1.0)
        vz = sign(velocity_vector[0][2])*min(abs(velocity_vector[0][2]), 1.0)

        if vx < 0:
            vx = max(vx, -1)
        # if abs(vx) >= abs(vz) or abs(vy) >= abs(vz):
        #     vz = 0
        # if abs(vy) >= abs(vx) or abs(vz) >= abs(vx):
        #     vx = 0
        # if abs(vx) >= abs(vy) or abs(vz) >= abs(vy):
        #     vy = 0
        return [vx, vy, vz]
        
    def publish_velocity(self, velocity_vector):
        self.gait_pub.publish(self.gait)
        velo = Twist()
        if not self.new_episode:
            velo.linear.x = velocity_vector[0]
            velo.linear.y = velocity_vector[1]
            velo.angular.z = velocity_vector[2]
        else:
            if len(self.buffer.states) > 4:
                self.new_episode = False
        self.velo_pub.publish(velo)
        

    def sample_velocity(self, mean, std):
        dist = torch.distributions.Normal(mean, std)
        velocity_vector = dist.sample()
        log_probs = dist.log_prob(velocity_vector).sum(axis=-1)  # Sum log probs for total log prob of the velocity vector
        return velocity_vector, log_probs
    
    def advantage(self, rewards: list, values: list):
        T = len(rewards)
        values = np.array(values)
        advantages = np.zeros(T)
        gamma = self.hyperparameters['gamma']
        lambda_ = self.hyperparameters['lambda']

        # Iterate over time steps in reverse order
        for t in reversed(range(T)):
            if t == T - 1:
                delta_t = rewards[t] - values[t]  # No future rewards beyond this point
            else:
                delta_t = rewards[t] + gamma * values[t+1] - values[t]  # Temporal difference error

            # Update temporal difference error using discounted future rewards
            for i in range(t+1, T):
                delta_t += (gamma * lambda_)**(i - t) * (rewards[i] - values[i-1])

            advantages[t] = delta_t

        return advantages
    
    def update_networks(self, batch_size=None):
        _, _, rewards, old_log_probs, new_log_probs, values = self.buffer.get_batch(batch_size)
        advantages = self.advantage(rewards=rewards, values=values)
        
        # policy_loss = self.compute_policy_loss(old_log_probs=old_log_probs, new_log_probs=new_log_probs, advantages=advantages)
        # value_loss = self.compute_value_loss(values=values, rewards=rewards)
        policy_loss, value_loss = self.compute_loss(old_log_probs, new_log_probs, advantages, values, rewards)
        with open(f'{self.loss_folder}/policy/loss_{self.num_updates}.txt', 'w') as f:
            f.write(f"{policy_loss}")

        with open(f'{self.loss_folder}/value/loss_{self.num_updates}.txt', 'w') as f:
            f.write(f"{value_loss}")
        
        self.num_updates += 1

        # Backpropagation and optimization for policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()


        # policy_loss, value_loss = self.compute_loss(old_log_probs, new_log_probs, advantages, values, rewards)

        # # Backpropagation and optimization for value network
        # self.value_optimizer.zero_grad()
        # value_loss.backward(retain_graph=True)
        # self.value_optimizer.step()


    def compute_loss(self, old_log_probs, new_log_probs, advantages, values, rewards):
        old_log_probs = torch.tensor(old_log_probs)
        new_log_probs = torch.tensor(new_log_probs)
        clip_epsilon = self.hyperparameters["ppo_clip"]
        clip_c1 = self.hyperparameters["clip_c1"]
        clip_c2 = self.hyperparameters["clip_c2"]
        gamma = self.hyperparameters["gamma"]

        # Compute policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = (torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages).requires_grad_(True)
        policy_loss = -torch.min(surr1, surr2).mean()

        # Compute value loss
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(rewards):
            running_reward = reward + gamma * running_reward
            discounted_rewards.insert(0, running_reward)

        values = torch.tensor(values, requires_grad=True).reshape(-1)  # Reshape to match discounted_rewards
        discounted_rewards = torch.tensor(discounted_rewards, requires_grad=True)
        value_loss = F.mse_loss(values, discounted_rewards)

        # Compute entropy regularization term
        entropy = -(new_log_probs.exp() * new_log_probs).sum(dim=-1).mean()  # Compute entropy of the action distribution

        # Combine policy and value losses with entropy regularization
        combined_loss = policy_loss - clip_c1 * value_loss + clip_c2 * entropy

        return combined_loss, value_loss

    def compute_policy_loss(self, old_log_probs, new_log_probs, advantages):
        old_log_probs = torch.tensor(old_log_probs)
        new_log_probs = torch.tensor(new_log_probs)
        clip_epsilon = self.hyperparameters["ppo_clip"]

        ratio = torch.exp(new_log_probs / old_log_probs)
        surr1 = (ratio * advantages).requires_grad_(True)
        surr2 = (torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages).requires_grad_(True)


        policy_loss = -torch.min(surr1, surr2).mean()

        # Add entropy regularization term
        entropy = -(new_log_probs.exp() * new_log_probs).sum(dim=-1).mean()  # Compute entropy of the action distribution
        policy_loss -= self.hyperparameters["entropy_coef"] * entropy  # Add entropy regularization to the policy loss
        return policy_loss
    
    def compute_value_loss(self, values, rewards):
        discounted_rewards = []
        running_reward = 0
        gamma = self.hyperparameters["gamma"]
        for reward in reversed(rewards):
            running_reward = reward + gamma * running_reward
            discounted_rewards.insert(0, running_reward)

        values = torch.tensor(values, requires_grad=True).reshape(-1)  # Reshape to match discounted_rewards
        discounted_rewards = torch.tensor(discounted_rewards, requires_grad=True)

        value_loss = F.mse_loss(values, discounted_rewards)
        return value_loss


    def initialize_episodes(self):
        while not self.init_rewards_dir:
            if rospy.has_param('/RL/runs/rewards_folder'):
                self.rewards_folder = rospy.get_param('/RL/runs/rewards_folder')
                self.episode_num = len(os.listdir(self.rewards_folder))
                self.new_episode = False
                self.init_rewards_dir = True

        while not self.init_steps_dir:
            if rospy.has_param('/RL/runs/steps_folder'):
                self.steps_folder = rospy.get_param('/RL/runs/steps_folder')
                self.init_steps_dir = True

        while not self.init_goal_dir:
            if rospy.has_param('/RL/runs/goals_folder'):
                self.goal_folder = rospy.get_param('/RL/runs/goals_folder')
                self.init_goal_dir = True

        while not self.init_distance_dir:
            if rospy.has_param('/RL/runs/distance_folder'):
                self.distance_folder = rospy.get_param('/RL/runs/distance_folder')
                self.init_distance_dir = True

        while not self.init_environment_dir:
            if rospy.has_param('/RL/runs/environment_folder'):
                self.environment_folder = rospy.get_param('/RL/runs/environment_folder')
                self.init_environment_dir = True
                

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
        if self.old_position is not None:
            # Calculate distance traveled from the old position to the current position
            distance_traveled = self.calculate_distance(self.old_position, position)
            if distance_traveled < 3:
                self.total_distance_traveled += distance_traveled
        else:
            distance_traveled = 0
        
        # Update the old position with the current position
        self.old_position = position
        self.position_tensor = torch.tensor([[distance_traveled, self.closest_goal]])

    def calculate_distance(self, pos1, pos2):
        distance = ((pos2.x - pos1.x) ** 2 + (pos2.y - pos1.y) ** 2) ** 0.5
        return distance
    
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
        if self.go:
            self.rewards += msg.data
            self.total_rewards += msg.data

    
    def states_callback(self, msg: String):
        if msg.data == "reach_goal":
            self.goal = True

    def new_episode_callback(self, msg):
        self.velo_pub.publish(Twist())
        self.new_episode = True
        self.steps_since_update = 0

    def save_model_callback(self, msg: String):
        save_model(self.policy_network, f"{self.models_folder}/policy_{msg.data}")
        # save_model(self.value_network, f"{self.models_folder}/value_{msg.data}")

    def goal0_callback(self, msg: Heuristic):
        if self.environment == 0:
            self.goal_dist = msg.manhattan_distance
            if msg.manhattan_distance < self.closest_goal:
                self.closest_goal = msg.manhattan_distance
            if msg.manhattan_distance > self.furthest_goal and msg.manhattan_distance < 15:
                self.furthest_goal = msg.manhattan_distance

    def goal1_callback(self, msg: Heuristic):
        if self.environment == 1:
            self.goal_dist = msg.manhattan_distance
            if msg.manhattan_distance < self.closest_goal:
                self.closest_goal = msg.manhattan_distance
            if msg.manhattan_distance > self.furthest_goal and msg.manhattan_distance < 15:
                self.furthest_goal = msg.manhattan_distance

    def dist_callback(self, msg: Float32):
        self.dist = msg.data

    def start_dist_callback(self, msg: Float32):
        self.max_start_dist = msg.data


if __name__ == '__main__':
    ppo = ProximalPolicyOptimization() 