import torch
import numpy as np
import h5py
from Model_CNN import IPPGCNN

# Caminhos dos arquivos
npy_path = r"C:\Users\Douglas\OneDrive - cefet-rj.br\Documentos\Material _ faculdade\Projeto IC\cohface\cohface\1\0\data\data_cwt.npy"
hdf5_path = r"C:\Users\Douglas\OneDrive - cefet-rj.br\Documentos\Material _ faculdade\Projeto IC\cohface\cohface\1\0\data.hdf5"

# Carregar espectrograma (shape: 60x60)
spec = np.load(npy_path)
spec = np.expand_dims(spec, axis=0)  # (1, 60, 60)
spec = torch.tensor(spec, dtype=torch.float32)

# Carregar valor de pulso dos 9 primeiros segundos
with h5py.File(hdf5_path, 'r') as f:
    pulse = f['pulse'][:]
    time = f['time'][:]
    mask = time <= 9
    pulse_9s = pulse[mask]
    pulse_mean = pulse_9s.mean() if pulse_9s.size > 0 else 0

# Mostrar valor real
print(f"Pulso real (média 0–9s): {pulse_mean:.2f} bpm")

# Instanciar modelo
model = IPPGCNN()

# (Opcional) Carregar pesos treinados
try:
    model.load_state_dict(torch.load("ippg_cnn_model.pth"))
    print("Modelo carregado com sucesso.")
except FileNotFoundError:
    print("Modelo não encontrado, usando pesos aleatórios.")

model.eval()

# Fazer previsão
with torch.no_grad():
    pred = model(spec.unsqueeze(0))  # (1, 1, 60, 60)
    predicted_pulse = pred.item()

print(f"Pulso estimado pela CNN: {predicted_pulse:.2f} bpm")
