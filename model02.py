import torch
import torch.nn as nn

class IPPGCNN(nn.Module):
    def __init__(self):
        super(IPPGCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(74, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # entrada esperada: [batch, 1, 74, 180]
        x = x.squeeze(1)          # vira [batch, 74, 180]
        x = self.conv(x)
        x = x.squeeze(-1)         # vira [batch, 256]
        x = self.fc(x)
        return x
