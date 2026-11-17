from logging import root
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class IPPGDatasetBase(Dataset):

    def __init__(
        self,
        root_dir,
        scenario,
        ippg_type="ippg3",
        mean=None,
        std=None,
        start_sec=1,
        end_sec=11,
        fps=25
    ):

        self.samples = []
        self.mean = mean
        self.std = std

        self.ippg_type = ippg_type.lower()

        self.start_sec = start_sec
        self.end_sec = end_sec
        self.fps = fps

        # ==========================================
        # CONVERTER SEGUNDOS -> ÍNDICES
        # ==========================================
        self.start_idx = int(start_sec * fps)
        self.end_idx = int(end_sec * fps)

        # tamanho fixo esperado
        self.target_length = self.end_idx - self.start_idx

        # ==========================================
        # VARREDURA DAS AMOSTRAS
        # ==========================================
        for root, _, files in os.walk(root_dir):

            # aceita string ou lista
            if isinstance(scenario, str):
                scenarios = [scenario.lower()]
            else:
                scenarios = [s.lower() for s in scenario]

            # verifica cenário no path
            if not any(s in root.lower() for s in scenarios):
                continue

            has_ippg = any(
                (
                    self.ippg_type in f.lower()
                    and f.endswith(".npy")
                )
                for f in files
            )

            has_label = "label.txt" in files

            if has_ippg and has_label:
                self.samples.append(root)

        print(f"[INFO] Total amostras ({scenario}): {len(self.samples)}")

        if len(self.samples) == 0:
            raise ValueError(
                f"Nenhuma amostra encontrada para cenário {scenario}"
            )

        # ==========================================
        # CALCULAR MEAN / STD
        # ==========================================
        if self.mean is None or self.std is None:
            self._compute_stats()

    # ==================================================
    # CALCULAR ESTATÍSTICAS DO LABEL
    # ==================================================
    def _compute_stats(self):

        labels = []

        for path in self.samples:

            try:
                with open(os.path.join(path, "label.txt")) as f:
                    labels.append(float(f.read()))

            except:
                continue

        self.mean = np.mean(labels)
        self.std = np.std(labels)

        print(f"[INFO] Mean BPM: {self.mean:.2f}")
        print(f"[INFO] Std BPM : {self.std:.2f}")

    # ==================================================
    def __len__(self):
        return len(self.samples)

    # ==================================================
    # CARREGAR IPPG
    # ==================================================
    def _load_ippg(self, path):

        files = os.listdir(path)

        for f in files:

            if (
                self.ippg_type in f.lower()
                and f.endswith(".npy")
            ):

                signal = np.load(os.path.join(path, f))

                return signal

        raise FileNotFoundError(
            f"{self.ippg_type} não encontrado em {path}"
        )

    # ==================================================
    # RECORTAR SEGMENTO
    # ==================================================
    def _crop_signal(self, signal):

        return signal[self.start_idx:self.end_idx]

    # ==================================================
    # GARANTIR TAMANHO FIXO
    # ==================================================
    def _pad_or_truncate(self, signal):

        if len(signal) > self.target_length:

            return signal[:self.target_length]

        elif len(signal) < self.target_length:

            pad_size = self.target_length - len(signal)

            return np.pad(signal, (0, pad_size))

        return signal

    # ==================================================
    # GET ITEM
    # ==================================================
    def __getitem__(self, idx):

        path = self.samples[idx]

        # =========================
        # carregar ippg
        # =========================
        signal = self._load_ippg(path)

        # =========================
        # recortar segmento
        # =========================
        signal = self._crop_signal(signal)

        # =========================
        # garantir tamanho fixo
        # =========================
        signal = self._pad_or_truncate(signal)

        # =========================
        # normalização
        # =========================
        signal = (
            signal - np.mean(signal)
        ) / (
            np.std(signal) + 1e-8
        )

        # =========================
        # formato tensor
        # =========================
        signal = np.expand_dims(signal, axis=0)

        signal = signal.astype(np.float32)

        # =========================
        # label
        # =========================
        with open(os.path.join(path, "label.txt")) as f:

            y = float(f.read())

        # normalizar label
        y = (y - self.mean) / self.std

        return (
            torch.tensor(signal),
            torch.tensor([y], dtype=torch.float32)
        )


# ======================================================
# FUNÇÃO PRINCIPAL
# ======================================================
def get_dataset(
    train_root,
    val_root,
    scenario,
    ippg_type="ippg3",
    test_root=None,
    start_sec=1,
    end_sec=26,
    fps=25
):

    # =========================
    # TREINO
    # =========================
    train_dataset = IPPGDatasetBase(
        train_root,
        scenario=scenario,
        ippg_type=ippg_type,
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps
    )

    # =========================
    # VALIDAÇÃO
    # =========================
    val_dataset = IPPGDatasetBase(
        val_root,
        scenario=scenario,
        ippg_type=ippg_type,
        mean=train_dataset.mean,
        std=train_dataset.std,
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps
    )

    # =========================
    # TESTE
    # =========================
    test_dataset = None

    if test_root is not None:

        test_dataset = IPPGDatasetBase(
            test_root,
            scenario=scenario,
            ippg_type=ippg_type,
            mean=train_dataset.mean,
            std=train_dataset.std,
            start_sec=start_sec,
            end_sec=end_sec,
            fps=fps
        )

    return train_dataset, val_dataset, test_dataset