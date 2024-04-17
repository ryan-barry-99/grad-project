import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Branch(nn.Module):
    def __init__(self, in_channels, dim1, dim2):
        super(CNN_Branch, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * (dim1 // 8) * (dim2 // 8), 16)  # Assuming 3 maxpool layers with kernel_size=2 and stride=2

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(1, 64 * (self.dim1 // 8) * (self.dim2 // 8))  # Flatten before fully connected layer
        x = self.fc(x)
        return x
    
class DenseNetwork(nn.Module):
    def __init__(self, input_size):
        super(DenseNetwork, self).__init__()
        # Define the layers of the dense network
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)

    def forward(self, rgb_out, depth_out, grid_out, heuristic, position, orientation, ang_vel, lin_acc):
        # Concatenate the output tensors from the CNN branches with other tensors
        x = torch.cat((rgb_out, depth_out, grid_out, heuristic, position, orientation, ang_vel, lin_acc), dim=1)
        # Pass through dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # Initialize CNN branches with appropriate input channels
        self.rgb_branch = CNN_Branch(in_channels=3, dim1=480, dim2=640)  # RGB image
        self.depth_branch = CNN_Branch(in_channels=1, dim1=720, dim2=1280)  # Depth image
        self.grid_branch = CNN_Branch(in_channels=1, dim1=100, dim2=100)  # Grid image
        self.fusion_net = DenseNetwork(63)
        


    def forward(self, rgb, depth, grid, heuristic, position, orientation, ang_vel, lin_acc):
        rgb_out = self.rgb_branch(rgb)
        depth_out = self.depth_branch(depth)
        grid_out = self.grid_branch(grid)

        x = self.fusion_net(rgb_out, depth_out, grid_out, heuristic, position, orientation, ang_vel, lin_acc)
        return x