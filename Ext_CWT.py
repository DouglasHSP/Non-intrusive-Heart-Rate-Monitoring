import cv2
import mediapipe as mp
import os
import matplotlib.pyplot as plt 
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter
import pywt

def process_video(input_video_path, output_base_dir):
    # Função para criar o filtro passa-banda
    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    # Aplicação do filtro passa-banda
    def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        filtered_data = filtfilt(b, a, data)
        return filtered_data

    def moving_average(signal, window_size):
        return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

    # Remoção de baseline com ajuste de spline
    def remove_baseline(signal, poly_order=2):
        x = np.arange(len(signal))
        coeffs = np.polyfit(x, signal, poly_order)  
        baseline = np.polyval(coeffs, x)     
        return signal - baseline  # Remove a baseline

    def normalized_signal(signal):
        sig= np.array(signal) 
        signal_min = np.min(sig)
        signal_max = np.max(sig)
        normalized_signal = (sig - signal_min) / (signal_max - signal_min)
        return normalized_signal

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {input_video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Parâmetros de ROI e redimensionamento
    amplification_factor_x = 0.7 
    amplification_factor_y = 0.7 
    resize_dim = (115, 50)       # Dimensão fixa para as imagens recortadas

    # Índices dos pontos de referência e conexões
    pontos_referencia = [103, 67, 104, 109, 10, 338, 297, 332, 333, 299, 337, 151, 108, 69, 104, 105, 66, 107, 9, 336, 296, 334]
    conexoes = [
        (103, 67), (67, 109), (109, 10), (10, 338),
        (338, 297), (297, 332), (103, 105),
        (105, 66), (66, 107), (107, 9), (9, 336), (336, 296), (296, 334), (334, 332)
    ]

    # Lista para armazenar o sinal verde
    green_signal = []
    # Metodo POS
    R_channel = []
    G_channel = []
    B_channel = []
    
    # Processamento dos frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converte o frame para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(frame_rgb)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Converter landmarks para coordenadas de pixel
                pontos = []
                for idx in pontos_referencia:
                    landmark = face_landmarks.landmark[idx] 
                    if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        pontos.append((x, y))

                if pontos:
                    # Determinar os limites da ROI com base nos pontos de referência
                    xs = [p[0] for p in pontos]
                    ys = [p[1] for p in pontos]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    x_min = max(0, x_min)
                    x_max = min(frame_width, x_max)
                    y_min = max(0, y_min)
                    y_max = min(frame_height, y_max)

                    roi = frame[y_min:y_max, x_min:x_max]
                    if roi.size > 0:
                        resized_roi = cv2.resize(roi, resize_dim)

                        R_channel.append(np.mean(resized_roi[:, :, 2]))
                        G_channel.append(np.mean(resized_roi[:, :, 1]))
                        B_channel.append(np.mean(resized_roi[:, :, 0]))
                        # Extração da componente verde
                        green_channel = resized_roi[:, :, 1]  # Componente verde (canal 1 em BGR)
                        mean_green = np.mean(green_channel)  # Média da intensidade verde
                        green_signal.append(mean_green)

        frame_count += 1

    # Aplicar o filtro passa-banda no sinal verde
    lowcut = 0.7  
    highcut = 4.0  
    window_size = 10  
    poly_order = 2    # Ordem do polinômio para a baseline

    # Conversão para numpy arrays
    R_channel = np.array(R_channel)
    G_channel = np.array(G_channel)
    B_channel = np.array(B_channel)

    # Cálculo dos sinais POS
    X_POS = G_channel - 0.5 * (R_channel + B_channel)
    Y_POS = 3 * G_channel - 2 * R_channel - B_channel

    X_POS = (X_POS - np.mean(X_POS)) / np.std(X_POS)
    Y_POS = (Y_POS - np.mean(Y_POS)) / np.std(Y_POS)

    # Projeção do sinal POS
    S_POS = X_POS + Y_POS
    filtered_signal_2 = apply_bandpass_filter(S_POS, lowcut, highcut, fps)

    smoothed_signal_2 = moving_average(filtered_signal_2, window_size)
    baseline_removed_signal_2 = remove_baseline(smoothed_signal_2, poly_order)

    IPPG2 = baseline_removed_signal_2
    IPPG2_normalized = normalized_signal(IPPG2)

    filtered_signal_1 = apply_bandpass_filter(green_signal, lowcut, highcut, fps)
    smoothed_signal_1 = moving_average(filtered_signal_1, window_size)
    baseline_removed_signal_1 = remove_baseline(smoothed_signal_1, poly_order)

    IPPG1 = baseline_removed_signal_1   
    IPPG1_normalized = normalized_signal(IPPG1)

    # Encontrando IPPG3
    IPPG3 = (IPPG1 + IPPG2) / 2
    IPPG3_normalized = normalized_signal(IPPG3)

    # Recortar o sinal IPP3 
    total_frames = 1600   
    duracao_total = 55      
    fps = total_frames / duracao_total

    frames = np.arange(0, len(IPPG3_normalized))  
    tempo = frames / fps

    inicio_frame = int(1 * fps)
    fim_frame = int(10 * fps)
    sinal_recortado = IPPG3_normalized[inicio_frame:fim_frame + 1]
    tempo_recortado = tempo[inicio_frame:fim_frame + 1]

    # Gerar o espectrograma
    def plot_wavelet(time, data, wavelet, title, ax):
        widths = np.geomspace(1, 1024, num=75)
        cwtmatr, freqs = pywt.cwt(
            data, widths, wavelet, sampling_period=np.diff(time).mean()
        )
        cwtmatr = np.abs(cwtmatr[:-1, :-1])
        
        pcm = ax.pcolormesh(time, freqs, cwtmatr, cmap= 'viridis', shading='auto')
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        return ax

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_wavelet(tempo_recortado, sinal_recortado, wavelet='cgau1', 
                title="Análise Wavelet da (Wavelet CGAU1)", ax=ax)
    

   
    
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{video_name}_espectrograma.png")
    fig.savefig(output_path)
    plt.close(fig)
    
    return True

