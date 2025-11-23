import torch
from torch.utils.data import DataLoader
from IPPG_Dataset import IPPGDataset
from Model_CNN import IPPGCNN
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt

def train(train_root, val_root, save_model_path="model_cnn_ippg.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Carrega datasets
    train_dataset = IPPGDataset(root_dir=train_root)
    val_dataset = IPPGDataset(root_dir=val_root)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = IPPGCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # learning rate

    epochs = 50
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validação
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")

    # Salvar modelo treinado
    torch.save(model.state_dict(), save_model_path)
    print(f"Modelo salvo em: {save_model_path}")

    # Plotar gráfico de loss
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Evolução do Loss por Época")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_val_loss.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    train(
        train_root="C:/Users/Douglas/OneDrive - cefet-rj.br/Documentos/Material _ faculdade/Projeto IC/TREINAMENTO_COHFACE",
        val_root="C:/Users/Douglas/OneDrive - cefet-rj.br/Documentos/Material _ faculdade/Projeto IC/VALIDACAO_COHFACE",
        save_model_path="model_cnn_ippg.pth"
    )
