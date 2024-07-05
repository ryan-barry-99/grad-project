import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class CNN_Branch(nn.Module):
    def __init__(self, in_channels, dim1, dim2):
        super(CNN_Branch, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(16 * (dim1 // 8) * (dim2 // 8), 64)  # Assuming 3 maxpool layers with kernel_size=2 and stride=2

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = x.view(1, 16 * (self.dim1 // 8) * (self.dim2 // 8))  # Flatten before fully connected layer
        x = self.fc(x)
        return x

    
LOG_SIG_MIN = 0
LOG_SIG_MAX = 0.01

MAX_X_VEL = 3.5
MAX_Y_VEL = 0.5
MAX_Z_VEL = 0.3

class StateExtractionNet(nn.Module):
    def __init__(self):
        super(StateExtractionNet, self).__init__()
        self.rgb_branch = CNN_Branch(in_channels=3, dim1=480, dim2=640)  # RGB image
        self.depth_branch = CNN_Branch(in_channels=1, dim1=720, dim2=1280)  # Depth image
        self.grid_branch = CNN_Branch(in_channels=1, dim1=100, dim2=100)  # Grid image

    def forward(self, state):
        # rgb, depth, grid, heuristic, position, orientation, ang_vel, lin_acc
        rgb_out = self.rgb_branch(state["rgb"])
        depth_out = self.depth_branch(state["depth"])
        grid_out = self.grid_branch(state["occupancy_grid"])

        return rgb_out, depth_out, grid_out
        # return grid_out
    
class DenseNetwork(nn.Module):
    def __init__(self, input_size):
        super(DenseNetwork, self).__init__()
        # Define the layers of the dense network
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)

    def forward(self, x):
        # Concatenate the output tensors from the CNN branches with other tensors
        # x = torch.cat(fused_observation, dim=1)
        # Pass through dense layers
        # rospy.loginfo(x.shape)

        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.tanh(self.fc4(x))
        return x

        
class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.extractor = StateExtractionNet()
        self.fc1 = nn.Linear(202, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fusion_net = DenseNetwork(hidden_size)

        self.dist_sign_layer1 = nn.Linear(32,16)
        self.dist_sign_layer2 = nn.Linear(16,8)

        self.fc_final_mean = nn.Linear(8, 3)
        self.fc_final_log_std = nn.Linear(8, 3)



    def forward(self, state, hidden_layers):
        self.hidden_layers = [layer for layer in hidden_layers]
        # rgb, depth, grid, heuristic, position, orientation, ang_vel, lin_acc
        rgb_out, depth_out, grid_out = self.extractor(state)
        # grid_out = self.extractor(state)
        # x = torch.cat((grid_out, state["heuristic"], state["orientation"], state["wall"]), dim=1)
        x = torch.cat((rgb_out, depth_out, grid_out, state["heuristic"], state["lin_acc"], state["ang_vel"], state["orientation"]), dim=1)
        # x = torch.cat((rgb_out, depth_out, grid_out), dim=1)
        x = self.fc1(x)
        
        for i, layer in enumerate(self.hidden_layers):
            x, self.hidden_layers[i] = self.lstm(x, layer)

        x = self.fusion_net(x)
        

        p = self.dist_sign_layer1(x)
        p = self.dist_sign_layer2(p)

        mean = self.fc_final_mean(p)
        # mean[:, [0]] = torch.clamp(mean[:, [0]], min=-MAX_X_VEL, max=MAX_X_VEL)
        # mean[:, [1]] = torch.clamp(mean[:, [0]], min=-MAX_Y_VEL, max=MAX_Y_VEL)
        # mean[:, [2]] = torch.clamp(mean[:, [2]], min=-MAX_Z_VEL, max=MAX_Z_VEL)

        log_std = self.fc_final_log_std(p)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()

        return mean, std
    

import rospy
class ValueNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(ValueNetwork, self).__init__()
        # self.extractor = StateExtractionNet()
        self.fc1 = nn.Linear(7, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dense_net = DenseNetwork(64)
        self.value_sign_layer1 = nn.Linear(32,16)
        self.value_sign_layer2 = nn.Linear(16,8)
        self.fc_final_value = nn.Linear(8, 2)  # Output a single value for state value
        
    def forward(self, state, action, hidden_layers):
        self.hidden_layers = [layer for layer in hidden_layers]
        # rgb_out, depth_out, grid_out = self.extractor(state)
        # x = torch.cat((rgb_out, depth_out, grid_out, state["heuristic"], state["orientation"], action), dim=1)
        x = torch.concatenate((state["heuristic"], state["orientation"], action), dim=1)
        x = self.fc1(x)
        for i, layer in enumerate(self.hidden_layers):
            x, self.hidden_layers[i] = self.lstm(x, layer)
        x = self.dense_net(x)
        x = self.value_sign_layer1(x)
        x = self.value_sign_layer2(x)
        # rospy.loginfo(f"input: {x}")
        x = self.fc_final_value(x)
        # rospy.loginfo(f"output: {x}")

        x = torch.softmax(x, dim=1)  # Apply softmax activation for classification
        return x
