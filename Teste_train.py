import torch
import numpy as np
import h5py
import cv2

from VIPHL_Dataset.Model_CNN import IPPGCNN

# =========================
# 📌 CAMINHOS
# =========================
npy_path = r"C:\Users\Douglas\OneDrive - cefet-rj.br\Documentos\Material _ faculdade\Projeto IC\cohface\cohface\1\0\data\data_cwt.npy"
hdf5_path = r"C:\Users\Douglas\OneDrive - cefet-rj.br\Documentos\Material _ faculdade\Projeto IC\cohface\cohface\1\0\data.hdf5"

# 🔥 estatísticas do treino (OBRIGATÓRIO se usou normalização)
mean_std_path = r"C:\Users\Douglas\OneDrive - cefet-rj.br\Documentos\Material _ faculdade\Projeto IC\Monitoramento_FA_PA\bpm_mean_std.npy"


# =========================
# 📌 PREPROCESSAMENTO (IGUAL AO TREINO)
# =========================
def preprocess(spec, target_size=(224, 224)):
    # 🔥 resize
    spec = cv2.resize(spec, target_size)

    # 🔥 normalização (igual dataset)
    spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)

    # 🔥 formato (C, H, W)
    spec = np.expand_dims(spec, axis=0)

    return torch.tensor(spec, dtype=torch.float32)


# =========================
# 📌 CARREGAR ESPECTROGRAMA
# =========================
spec = np.load(npy_path)
spec = preprocess(spec)  # (1, 224, 224)

# adicionar batch → (1, 1, 224, 224)
spec = spec.unsqueeze(0)


# =========================
# 📌 VALOR REAL (GROUND TRUTH)
# =========================
with h5py.File(hdf5_path, 'r') as f:
    pulse = f['pulse'][:]
    time = f['time'][:]

    mask = time <= 9
    pulse_9s = pulse[mask]

    pulse_mean = pulse_9s.mean() if pulse_9s.size > 0 else 0

print(f"Pulso real (média 0–9s): {pulse_mean:.2f} bpm")


# =========================
# 📌 MODELO
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = IPPGCNN().to(device)

try:
    model.load_state_dict(torch.load("ippg_cnn_model.pth", map_location=device))
    print("Modelo carregado com sucesso.")
except FileNotFoundError:
    print("Modelo não encontrado, usando pesos aleatórios.")

model.eval()


# =========================
# 📌 CARREGAR NORMALIZAÇÃO
# =========================
try:
    mean_bpm, std_bpm = np.load(mean_std_path)
    print(f"[INFO] Mean treino: {mean_bpm:.2f} | Std: {std_bpm:.2f}")
except:
    mean_bpm, std_bpm = 0, 1
    print("[WARNING] Mean/Std não encontrados — saída será normalizada")


# =========================
# 📌 INFERÊNCIA
# =========================
with torch.no_grad():
    spec = spec.to(device)

    pred_norm = model(spec)  # saída normalizada
    pred_norm = pred_norm.item()

    # 🔥 desnormalizar
    pred_bpm = pred_norm * std_bpm + mean_bpm


# =========================
# 📌 RESULTADO
# =========================
print(f"Pulso estimado (normalizado): {pred_norm:.4f}")
print(f"Pulso estimado (BPM real): {pred_bpm:.2f} bpm")