import torch
import torch.nn as nn
import torch.nn.functional as F

class IPPGCNN(nn.Module):
    def __init__(self):
        super(IPPGCNN, self).__init__()
        
        # Camada convolucional
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 38 * 131, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # flatten automático
        x = self.fc1(x)
        x = self.relu(x)
        x = self.out(x)
        return x