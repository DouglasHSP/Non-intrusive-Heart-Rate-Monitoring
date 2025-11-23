import os
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
import glob
from scipy.signal import find_peaks

class IPPGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []

        # Coleta todos os pares (cwt, hdf5)
        for npy_path in glob.glob(os.path.join(root_dir, "*", "*", "data", "data_cwt.npy")):
            hdf5_path = os.path.join(os.path.dirname(os.path.dirname(npy_path)), "data.hdf5")
            if os.path.exists(hdf5_path):
                self.samples.append((npy_path, hdf5_path))
            else:
                print(f"[Aviso] HDF5 não encontrado para {npy_path}")

        # === Cálculo da média e desvio padrão global ===
        self.bpm_list = []
        print("\n[DEBUG] Testando compatibilidade CWT/HDF5:")

        for npy_path, hdf5_path in self.samples[:10]:  # primeiros 10
            print("\nCWT encontrado:")
            print("   ", npy_path)

            print("HDF5 esperado:")
            print("   ", hdf5_path)

            print("HDF5 existe?", os.path.exists(hdf5_path))
            
        for _, hdf5_path in self.samples:
            with h5py.File(hdf5_path, 'r') as f:
                pulse = f['/pulse'][:]
                time = f['/time'][:]

                mask = (time >= 0) & (time <= 9)
                pulse_9s = pulse[mask]
                time_9s = time[mask]

                bpm = self.calcular_bpm(pulse_9s, time_9s)
                if not np.isnan(bpm):
                    self.bpm_list.append(bpm)

        self.mean_bpm = np.mean(self.bpm_list)
        self.std_bpm = np.std(self.bpm_list)
        print(f"[INFO] Média BPM global: {self.mean_bpm:.2f}, Desvio padrão: {self.std_bpm:.2f}")

    def __len__(self):
        return len(self.samples)

    def calcular_bpm(self, pulse, time):
        try:
            fs = 1 / (time[1] - time[0])
            min_distance_sec = 0.6
            distance_samples = int(min_distance_sec * fs)
            peaks, _ = find_peaks(pulse, distance=distance_samples, prominence=0.5)

            if len(peaks) >= 2:
                rr_intervals = np.diff(time[peaks])
                bpm = 60 / np.mean(rr_intervals)
            else:
                bpm = np.nan
            return bpm
        except Exception as e:
            print(f"[Erro cálculo BPM]: {e}")
            return np.nan

    def __getitem__(self, idx):
        npy_path, hdf5_path = self.samples[idx]

        cwt_img = np.load(npy_path)
            
        cwt_img = (cwt_img - np.mean(cwt_img)) / (np.std(cwt_img) + 1e-8)
        cwt_img = np.expand_dims(cwt_img, axis=0).astype(np.float32)

        with h5py.File(hdf5_path, 'r') as f:
            pulse = f['/pulse'][:]
            time = f['/time'][:]

            mask = (time >= 0) & (time <= 9)
            pulse_9s = pulse[mask]
            time_9s = time[mask]

            bpm = self.calcular_bpm(pulse_9s, time_9s)
            

        # === Normalização (z-score) ===
        bpm_norm = (bpm - self.mean_bpm) / self.std_bpm
        target = np.array([bpm_norm], dtype=np.float32)

        cwt_tensor = torch.tensor(cwt_img)
        target_tensor = torch.tensor(target)

        if self.transform and callable(self.transform):
            cwt_tensor = self.transform(cwt_tensor)

        return cwt_tensor, target_tensor