import torch
from torch.utils.data import Dataset
import numpy as np
import soundfile as sf
# Hinweis: Wir nutzen den Processor hier nur für Config-Werte (hop_length etc.),
# nicht für Berechnungen.
from src.audio_processor_torch import TorchAudioProcessor


class StreamingAudioDataset(Dataset):
    def __init__(self, file_pairs, processor: TorchAudioProcessor, crop_width: int = 256, overlap: float = 0.5):
        """
        Liest RAW AUDIO direkt von der Festplatte.
        Keine STFT-Berechnung hier! Das passiert später auf der GPU.
        """
        self.file_pairs = file_pairs
        self.processor = processor
        self.crop_width = crop_width

        # Wir holen uns die Werte aus dem Processor
        self.hop_length = processor.hop_length
        self.n_fft = processor.n_fft

        # Berechne Samples pro Crop
        # Damit das U-Net ein Spektrogramm der Breite 256 bekommt, brauchen wir so viele Samples:
        # Formel: (Frames - 1) * Hop + n_fft
        self.samples_per_crop = (crop_width - 1) * self.hop_length + self.n_fft

        # Schrittweite in Samples für Sliding Window
        stride_frames = int(crop_width * (1 - overlap))
        self.stride_samples = stride_frames * self.hop_length
        if self.stride_samples < 1: self.stride_samples = 1

        # Index aufbauen
        self.samples_index = []
        self._build_index()

    def _build_index(self):
        print(f"Indiziere {len(self.file_pairs)} Dateien für Streaming (Raw Audio)...")

        for file_idx, (in_path, tgt_path) in enumerate(self.file_pairs):
            # Nur Header lesen
            info = sf.info(str(in_path))
            total_samples = info.frames

            # Startpunkte berechnen
            max_start = total_samples - self.samples_per_crop
            if max_start > 0:
                starts = range(0, max_start, self.stride_samples)
                for s in starts:
                    self.samples_index.append((file_idx, s))
            else:
                # Fallback: Datei ist kürzer als ein Crop -> Einmalig hinzufügen (wird gepaddet)
                self.samples_index.append((file_idx, 0))

        print(f"Index fertig: {len(self.samples_index)} Audio-Chunks bereit.")

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        file_idx, start_sample = self.samples_index[idx]
        in_path, tgt_path = self.file_pairs[file_idx]

        # 1. Audio laden
        # Wir laden genau die benötigte Menge an Samples
        stop_sample = start_sample + self.samples_per_crop

        # sf.read ist robust: Wenn stop_sample > Dateiende, liefert es einfach weniger zurück
        y_in, _ = sf.read(str(in_path), start=start_sample, stop=stop_sample, always_2d=True)
        y_tgt, _ = sf.read(str(tgt_path), start=start_sample, stop=stop_sample, always_2d=True)

        # Transponieren: (Time, Channels) -> (Channels, Time)
        y_in = y_in.T
        y_tgt = y_tgt.T

        # Mono -> Stereo Check
        if y_in.shape[0] == 1:
            y_in = np.concatenate([y_in, y_in], axis=0)
        if y_tgt.shape[0] == 1:
            y_tgt = np.concatenate([y_tgt, y_tgt], axis=0)

        # 2. AUDIO PADDING (WICHTIG!)
        # Falls wir am Ende der Datei sind und weniger Samples bekommen haben als nötig
        # pad_width Format für numpy: ((channel_before, channel_after), (time_before, time_after))

        # Check Input
        current_samples_in = y_in.shape[1]
        if current_samples_in < self.samples_per_crop:
            pad_len = self.samples_per_crop - current_samples_in
            y_in = np.pad(y_in, ((0, 0), (0, pad_len)), mode='constant')

        # Check Target (unabhängig padden, falls Datei kürzer)
        current_samples_tgt = y_tgt.shape[1]
        if current_samples_tgt < self.samples_per_crop:
            pad_len = self.samples_per_crop - current_samples_tgt
            y_tgt = np.pad(y_tgt, ((0, 0), (0, pad_len)), mode='constant')

        # 3. Return als Tensor
        # Wir geben KEIN Spektrogramm zurück, sondern das Audio!
        # Die Umwandlung passiert auf der GPU im Training-Loop.
        return torch.from_numpy(y_in).float(), torch.from_numpy(y_tgt).float()