import os
import numpy as np
import librosa
from scipy.signal import butter, filtfilt

def butterworth_filter(wave, sr, order, band):
    b, a = butter(N=order, Wn=band, btype='bandpass', fs=sr)
    return filtfilt(b, a, wave)

def compute_waves(df, sr=22050, filter=butterworth_filter, directory="data/audios_development"):
    waves = {}
    for filename in os.listdir(directory):
        filename_split = filename.split(".")
        id = int(filename_split[0])
        file_path = os.path.join(directory, filename)  # Full file path
        wave, _ = librosa.load(file_path, sr=sr)
        waves[id-1] = wave
    
    data = df.copy()
    data['wave'] = data.index.map(waves)
    
    if filter:
        data['wave'] = data['wave'].apply(lambda x: filter(x, sr=sr, order=4, band=[300, 3400]))
    
    data['duration'] = data['wave'].apply(lambda wave: len(wave) / sr)
    data['intensity'] = data['wave'].apply(lambda wave: np.sqrt(np.mean(wave**2)))
    
    return data

def compute_spectrograms(df, sr=22050, n_fft=2048, hop_length=512):
    
    data = df.copy()
    
    data['spectrogram'] = data['wave'].apply(lambda y: np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=512)))
    data['melspectrogram'] = data['spectrogram'].apply(lambda S: librosa.feature.melspectrogram(S=S, sr=sr, n_fft=n_fft, hop_length=hop_length))
    data['log_melspectrogram'] = data['melspectrogram'].apply(lambda S: librosa.power_to_db(np.abs(S)**2))

    return data