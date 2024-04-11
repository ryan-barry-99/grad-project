import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBranch(nn.Module):
    def __init__(self, in_channels, output_size):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64, output_size)  # Adjust output_size based on feature requirement

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Initialize CNN branches with appropriate input channels
        self.rgb_branch = CNNBranch(in_channels=3, output_size=64)  # RGB image
        self.depth_branch = CNNBranch(in_channels=1, output_size=64)  # Depth image
        self.grid_branch = CNNBranch(in_channels=1, output_size=64)  # Grid image
        
        # Dense network
        self.fc1 = nn.Linear(64 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # Output layer with 3 outputs

    def forward(self, rgb, depth, grid):
        # Process inputs through their respective branches
        rgb_out = self.rgb_branch(rgb)
        depth_out = self.depth_branch(depth)
        grid_out = self.grid_branch(grid)
        
        # Concatenate outputs
        combined = torch.cat((rgb_out, depth_out, grid_out), dim=1)
        
        # Dense network
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x