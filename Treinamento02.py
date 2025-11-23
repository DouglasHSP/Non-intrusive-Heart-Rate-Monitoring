import torch
from torch.utils.data import DataLoader
from dataset02 import IPPGDataset02
from model02 import IPPGCNN02
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

def train02(train_root, val_root, save_model_path="model_cnn_ippg.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Dataset de treino
    train_dataset = IPPGDataset02(root_dir=train_root)
    val_dataset = IPPGDataset02(root_dir=val_root,
                              mean_bpm=train_dataset.mean_bpm,
                              std_bpm=train_dataset.std_bpm)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = IPPGCNN02().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 50
    train_losses, val_losses = [], []

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

        print(f"Epoch {epoch+1}/{epochs} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

    torch.save(model.state_dict(), save_model_path)
    print(f"[INFO] Modelo salvo em {save_model_path}")

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Treino", marker='o')
    plt.plot(val_losses, label="Validação", marker='s')
    plt.xlabel("Época")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Curva de Treinamento e Validação")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_val_loss.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    train02(
        train_root=r"C:\Users\Douglas\OneDrive - cefet-rj.br\Documentos\Material _ faculdade\Projeto IC\TREINAMENTO_COHFACE",
        val_root=r"C:\Users\Douglas\OneDrive - cefet-rj.br\Documentos\Material _ faculdade\Projeto IC\VALIDACAO_COHFACE",
        save_model_path="model_cnn_ippg.pth"
    )
