import torch
from torch.utils.data import DataLoader
from Dataset_VIPHL import get_dataset
from Model_CNN import IPPGCNN
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np

def train(train_root, val_root, save_model_path, scenario, signal):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 🔥 DATASETS FLEXÍVEIS
    train_dataset, val_dataset, _ = get_dataset(
    train_root=train_root,
    val_root=val_root,
    scenario=scenario,
    ippg_type=signal
)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # =========================
    # 🔥 MODELO
    # =========================
    model = IPPGCNN().to(device)

    # =========================
    # 🔥 LOSS (opcional ajustado)
    # =========================
    criterion = nn.L1Loss()  # MAE (melhor interpretável em BPM)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-2)

    epochs = 50
    train_losses = []
    val_losses = []

    # =========================
    # 🔥 TREINAMENTO
    # =========================
    for epoch in range(epochs):

        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
 
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # =========================
        # 🔥 VALIDAÇÃO
        # =========================
        model.eval()
        running_val_loss = 0.0

        # 🔥 NOVO: erro real em BPM
        running_val_loss_bpm = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                # loss normalizada
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

                # 🔥 converter para BPM real
                outputs_bpm = outputs * train_dataset.std + train_dataset.mean
                targets_bpm = targets * train_dataset.std + train_dataset.mean

                loss_bpm = torch.mean(torch.abs(outputs_bpm - targets_bpm))
                running_val_loss_bpm += loss_bpm.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_loss_bpm = running_val_loss_bpm / len(val_loader.dataset)

        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss (norm): {epoch_train_loss:.4f}")
        print(f"Val Loss (norm): {epoch_val_loss:.4f}")
        print(f"Val Loss (BPM): {epoch_val_loss_bpm:.2f} BPM")
        print("-" * 40)

    # =========================
    # 🔥 SALVAR MODELO
    # =========================
    torch.save(model.state_dict(), save_model_path)
    print(f"Modelo salvo em: {save_model_path}")

    # 🔥 salvar mean/std CORRETOS
    np.save("bpm_mean_std.npy", np.array([train_dataset.mean, train_dataset.std]))

    # =========================
    # 🔥 PLOT
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE Normalizado)")
    plt.title("Evolução do Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("train_val_loss.png", dpi=300)
    plt.show()


if __name__ == "__main__":

    train(
        train_root=r"C:\Users\Douglas\Documents\VIPL_IPPG_TRAIN",

        val_root=r"C:\Users\Douglas\Documents\VIPL_IPPG_VALIDATION",

        save_model_path="model_v1_ippg3.pth",

        scenario=["v1", "v2", "v3"],

        signal="ippg1"
    )