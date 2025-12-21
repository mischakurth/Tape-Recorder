import torch
from torch.utils.data import Dataset
import numpy as np
import random
# Angenommen audio_processor.py liegt im selben Ordner oder PYTHONPATH ist gesetzt
from src.audio_processor import AudioProcessor
import glob
import os


class AudioDataset(Dataset):
    def __init__(self, directory: str, processor: AudioProcessor, crop_width: int = 256):
        self.files = sorted(glob.glob(os.path.join(directory, "*.wav")))
        self.processor = processor
        self.crop_width = crop_width
        # Ziel-Höhe für das U-Net (muss durch 16 teilbar sein)
        self.target_height = 1040

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        # 1. Volle Datei laden & STFT berechnen (verhindert Klicks an Rändern)
        y = self.processor.load_audio(file_path)
        stft = self.processor.audio_to_stft(y)

        # Shape ist jetzt (Channels, 1025, Time_Frames)
        cnn_input = self.processor.stft_to_cnn_input(stft)

        # 2. PADDING (Lossless): Von 1025 auf 1040 auffüllen
        # Pad-Config: ((0,0), (0, 15), (0,0)) -> Nur 2. Dimension (Freq) hinten auffüllen
        c, h, w = cnn_input.shape
        pad_height = self.target_height - h
        if pad_height > 0:
            cnn_input = np.pad(cnn_input, ((0, 0), (0, pad_height), (0, 0)), mode='constant')

        # 3. SLICING (Zeit): Hier schneiden wir das STFT-Bild
        # Das ist mathematisch sicher, da STFT-Fenster sich bereits überlappt haben.
        _, _, time_frames = cnn_input.shape

        if time_frames > self.crop_width:
            start = random.randint(0, time_frames - self.crop_width)
            crop = cnn_input[:, :, start: start + self.crop_width]
        else:
            # Falls Audio kürzer als der Crop ist -> Padding in Zeitrichtung
            pad_time = self.crop_width - time_frames
            crop = np.pad(cnn_input, ((0, 0), (0, 0), (0, pad_time)), mode='constant')

        return torch.from_numpy(crop).float()