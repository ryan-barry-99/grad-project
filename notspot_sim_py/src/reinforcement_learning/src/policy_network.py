#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import rospy


class CNNBranch(nn.Module):
    def __init__(self, in_channels, scaler, output_size):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(int(scaler*64), output_size)  # Adjust output_size based on feature requirement

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Initialize CNN branches with appropriate input channels
        out_sz = 64
        self.rgb_branch = CNNBranch(in_channels=3, scaler=4800, output_size=out_sz)  # RGB image
        self.depth_branch = CNNBranch(in_channels=1, scaler=14400, output_size=out_sz)  # Depth image
        self.grid_branch = CNNBranch(in_channels=1, scaler=2.25, output_size=out_sz)  # Grid image
        
        # Dense network
        num_goals = 5
        self.fc1 = nn.Linear(out_sz * 3 + num_goals * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output layer with 3 outputs


    def forward(self, rgb, depth, grid, heuristic):
        # Assuming inputs are single instances, ensure they're correctly batched
        # No need to unsqueeze if inputs already have a batch dimension
        rgb_out = self.rgb_branch(rgb)
        depth_out = self.depth_branch(depth.unsqueeze(0).unsqueeze(0))
        grid_out = self.grid_branch(grid.unsqueeze(0))
        
        # Ensure 'heuristic' is correctly shaped: [1, num_goals * 3]
        # Adjust as necessary to match your specific input shape requirements
        heuristic_flattened = heuristic.view(1, -1)
        
        # Concatenate outputs: Ensure dimensions align for concatenation
        combined = torch.cat((rgb_out, depth_out, grid_out, heuristic_flattened), dim=1)
        
        # Dense network processing
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        rospy.loginfo(f"{x.size()}")
        
        return x
    # def forward(self, rgb, depth, grid, heuristic):
    #     # Process inputs through their respective branches
    #     depth_out = self.depth_branch(depth.unsqueeze(0))
    #     rgb_out = self.rgb_branch(rgb).expand_as(depth_out)
    #     grid_out = self.grid_branch(grid.unsqueeze(0))
    #     heuristic_flattened = heuristic.view(-1).unsqueeze(0).repeat(64, 1)
        
    #     # Concatenate outputs
    #     combined = torch.cat((rgb_out, depth_out, grid_out, heuristic_flattened), dim=1)
        
    #     # Dense network
    #     x = F.relu(self.fc1(combined))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)

    #     return x