import torch
import torch.nn as nn

class IPPGCNN02(nn.Module):
    def __init__(self, input_shape=(1, 74, 261)):
        super(IPPGCNN02, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.dropout = nn.Dropout(0.5)

        # Calcula automaticamente o tamanho do flatten
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy_out = self.pool(self.relu(self.conv1(dummy)))
            self.flatten_dim = dummy_out.numel()
            print(f"[DEBUG] Tamanho flatten calculado: {self.flatten_dim}")

        # Camadas totalmente conectadas
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x
