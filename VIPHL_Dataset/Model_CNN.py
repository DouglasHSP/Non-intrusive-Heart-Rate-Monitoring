import torch
import torch.nn as nn


class IPPGCNN(nn.Module):

    def __init__(self):

        super(IPPGCNN, self).__init__()

        self.features = nn.Sequential(

            # =========================
            # BLOCO 1
            # =========================
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # =========================
            # BLOCO 2
            # =========================
            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # =========================
            # BLOCO 3
            # =========================
            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # =========================
            # GLOBAL POOLING
            # =========================
            nn.AdaptiveAvgPool1d(1)
        )

        # =========================
        # REGRESSÃO BPM
        # =========================
        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.regressor(x)

        return x