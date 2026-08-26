import torch
import torch.nn as nn

class QuanvolutionClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(QuanvolutionClassifier, self).__init__()
        
        # Precomputed quantum features have shape (N, 4, 14, 14)
        # Apply the exact same post-processing as the classical baseline
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        # Output spatial dim: 14x14. Channels: 4. Total flattened: 4 * 14 * 14 = 784
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
