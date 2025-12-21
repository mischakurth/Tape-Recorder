import torch
from torch.utils.data import Dataset
import numpy as np
from src.audio_processor import AudioProcessor
from collections import OrderedDict
import soundfile as sf
import sys


class SlidingWindowDataset(Dataset):
    def __init__(self, file_pairs, processor: AudioProcessor, crop_width: int = 256, overlap: float = 0.5):
        self.file_pairs = file_pairs
        self.processor = processor
        self.crop_width = crop_width
        self.target_height = 1040
        self.stride = int(crop_width * (1 - overlap))
        if self.stride < 1: self.stride = 1

        # --- RAM Management ---
        # Wir schätzen den Speicherbedarf.
        # 1 Minute Stereo Audio als CNN-Tensor (float32) braucht ca. 350 MB.
        # Bei 36 GB RAM können wir locker 10-15 GB nutzen.
        self.ram_limit_gb = 12.0

        # Cache Init
        self.cache = OrderedDict()
        self.preloaded_data = {}  # Speichert ALLES, wenn es passt
        self.use_preload = False

        # Index erstellen
        self.samples = []
        self._prepare_index_and_check_memory()

    def _prepare_index_and_check_memory(self):
        print("Analysiere Dataset & Speicherbedarf...")
        total_frames_estimated = 0

        for file_idx, (in_path, _) in enumerate(self.file_pairs):
            # Schnell-Check der Länge
            info = sf.info(str(in_path))
            # STFT Frames Schätzung
            num_frames = 1 + int(info.frames / self.processor.hop_length)
            total_frames_estimated += num_frames

            # Sliding Window Index berechnen
            max_start = num_frames - self.crop_width
            if max_start > 0:
                starts = range(0, max_start, self.stride)
                for s in starts:
                    self.samples.append((file_idx, s))

        # Speicher-Schätzung:
        # (4 Channels * 1040 Freq * Frames * 4 Bytes * 2 (Input+Target)) / 1024^3
        estimated_gb = (total_frames_estimated * 4 * 1040 * 4 * 2) / (1024 ** 3)
        print(f"Index fertig: {len(self.samples)} Chunks.")
        print(f"Geschätzter RAM-Bedarf für vollständiges Laden: {estimated_gb:.2f} GB")

        if estimated_gb < self.ram_limit_gb:
            print(
                "🚀 Dataset passt in den RAM! Starte Preloading (das dauert kurz, beschleunigt aber das Training enorm)...")
            self.use_preload = True
            self._preload_all()
        else:
            print(f"⚠️ Dataset zu groß für RAM-Limit ({self.ram_limit_gb} GB). Nutze Smart-Caching.")
            # Cache Größe erhöhen: Mindestens 4 Songs sollten reinpassen, um Thrashing zu reduzieren
            self.cache_limit = 4

    def _preload_all(self):
        for i in range(len(self.file_pairs)):
            self.preloaded_data[i] = self._load_and_process_file(i)
            # Kleiner Progress-Indikator
            print(f"   Lade Datei {i + 1}/{len(self.file_pairs)} in RAM...", end='\r')
        print("\nPreloading abgeschlossen.")

    def _load_and_process_file(self, file_idx):
        in_path, tgt_path = self.file_pairs[file_idx]

        # 1. Input
        y_in = self.processor.load_audio(str(in_path))
        stft_in = self.processor.audio_to_stft(y_in)
        cnn_in = self.processor.stft_to_cnn_input(stft_in)

        # 2. Target
        y_tgt = self.processor.load_audio(str(tgt_path))
        stft_tgt = self.processor.audio_to_stft(y_tgt)
        cnn_tgt = self.processor.stft_to_cnn_input(stft_tgt)

        # 3. Padding (Freq)
        c, h, w = cnn_in.shape
        pad_height = self.target_height - h
        if pad_height > 0:
            cnn_in = np.pad(cnn_in, ((0, 0), (0, pad_height), (0, 0)), mode='constant')
            cnn_tgt = np.pad(cnn_tgt, ((0, 0), (0, pad_height), (0, 0)), mode='constant')

        # Länge angleichen
        min_len = min(cnn_in.shape[2], cnn_tgt.shape[2])
        return (cnn_in[:, :, :min_len], cnn_tgt[:, :, :min_len])

    def _get_data(self, file_idx):
        # Modus A: Alles im RAM
        if self.use_preload:
            return self.preloaded_data[file_idx]

        # Modus B: Cache (für riesige Datasets)
        if file_idx in self.cache:
            self.cache.move_to_end(file_idx)
            return self.cache[file_idx]

        # Load & Cache
        data = self._load_and_process_file(file_idx)
        self.cache[file_idx] = data
        if len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, start_frame = self.samples[idx]

        # Blitzschneller Zugriff (da im RAM oder Cache)
        full_input, full_target = self._get_data(file_idx)

        end_frame = start_frame + self.crop_width

        # Safety Check falls File kürzer als erwartet (Rundungsfehler)
        if end_frame > full_input.shape[2]:
            end_frame = full_input.shape[2]
            start_frame = end_frame - self.crop_width

        crop_in = full_input[:, :, start_frame:end_frame]
        crop_target = full_target[:, :, start_frame:end_frame]

        return torch.from_numpy(crop_in).float(), torch.from_numpy(crop_target).float()