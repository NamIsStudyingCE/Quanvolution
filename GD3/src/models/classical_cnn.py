import torch
import torch.nn as nn

class SymmetricalMinimumCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SymmetricalMinimumCNN, self).__init__()
        
        # 1 Conv2D layer to strictly mimic the 2x2 quantum kernel with stride 2
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        # Output spatial dim: 14x14. Channels: 4. Total flattened: 4 * 14 * 14 = 784
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
