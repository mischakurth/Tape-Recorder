import torch
from torch.utils.data import Dataset
import numpy as np
import soundfile as sf
from src.audio_processor import AudioProcessor


class StreamingAudioDataset(Dataset):
    def __init__(self, file_pairs, processor: AudioProcessor, crop_width: int = 256, overlap: float = 0.5):
        """
        Liest Audio direkt von der Festplatte (Chunk für Chunk).
        Minimale RAM-Last, aber höhere CPU-Last durch ständiges STFT-Berechnen.
        """
        self.file_pairs = file_pairs
        self.processor = processor
        self.crop_width = crop_width
        self.target_height = 1040

        # Berechne Samples pro Crop
        # Formel: (Frames - 1) * Hop + n_fft
        # Wir addieren etwas Sicherheitspuffer (+ n_fft)
        self.samples_per_crop = (crop_width - 1) * processor.hop_length + processor.n_fft

        # Schrittweite in Samples
        stride_frames = int(crop_width * (1 - overlap))
        self.stride_samples = stride_frames * processor.hop_length
        if self.stride_samples < 1: self.stride_samples = 1

        # Index aufbauen (Nur Pfade und Start-Positionen speichern)
        self.samples_index = []
        self._build_index()

    def _build_index(self):
        print(f"Indiziere {len(self.file_pairs)} Dateien für Streaming...")

        for file_idx, (in_path, tgt_path) in enumerate(self.file_pairs):
            # Nur Header lesen für Länge (sehr schnell)
            info = sf.info(str(in_path))
            total_samples = info.frames

            # Berechne Startpunkte
            max_start = total_samples - self.samples_per_crop
            if max_start > 0:
                # Range(start, stop, step)
                starts = range(0, max_start, self.stride_samples)
                for s in starts:
                    self.samples_index.append((file_idx, s))

            # WICHTIG: Auch den letzten Rest mitnehmen (der kürzer sein kann)
            # Das ist besonders für die Validierung wichtig, damit wir den Song bis zum Ende hören
            # Bei overlap=0.0 bleibt oft ein Rest übrig. Wir fügen ihn hinzu und padden ihn später.
            # (Optionaler Schritt, falls du wirklich ALLES hören willst. Der einfache Loop oben reicht meist.)

        print(f"Index fertig: {len(self.samples_index)} Chunks bereit zum Streamen.")

    def __len__(self):
        return len(self.samples_index)

    def __getitem__(self, idx):
        file_idx, start_sample = self.samples_index[idx]
        in_path, tgt_path = self.file_pairs[file_idx]

        # 1. Nur den kleinen Ausschnitt laden (IO-Operation)
        stop_sample = start_sample + self.samples_per_crop

        # sf.read lädt exakt diesen Bereich
        # always_2d=True sorgt für (samples, channels) Format
        y_in, _ = sf.read(str(in_path), start=start_sample, stop=stop_sample, always_2d=True)
        y_tgt, _ = sf.read(str(tgt_path), start=start_sample, stop=stop_sample, always_2d=True)

        # Transponieren zu (Channels, Samples)
        y_in = y_in.T
        y_tgt = y_tgt.T

        # Falls Mono, zu Pseudo-Stereo machen
        if y_in.shape[0] == 1:
            y_in = np.concatenate([y_in, y_in], axis=0)
        if y_tgt.shape[0] == 1:
            y_tgt = np.concatenate([y_tgt, y_tgt], axis=0)

        # 2. STFT Berechnung On-The-Fly
        stft_in = self.processor.audio_to_stft(y_in)
        stft_tgt = self.processor.audio_to_stft(y_tgt)

        # 3. CNN Input Konvertierung
        cnn_in = self.processor.stft_to_cnn_input(stft_in)
        cnn_tgt = self.processor.stft_to_cnn_input(stft_tgt)

        # 4. Frequency Padding (Höhe anpassen: x -> 1040)
        # Input
        c, h, w = cnn_in.shape
        pad_h = self.target_height - h
        if pad_h > 0:
            cnn_in = np.pad(cnn_in, ((0, 0), (0, pad_h), (0, 0)), mode='constant')

        # Target (separat prüfen!)
        c_t, h_t, w_t = cnn_tgt.shape
        pad_h_t = self.target_height - h_t
        if pad_h_t > 0:
            cnn_tgt = np.pad(cnn_tgt, ((0, 0), (0, pad_h_t), (0, 0)), mode='constant')

        # 5. Time Padding (Breite anpassen: x -> 256) [DER FIX]
        # Wir müssen BEIDE separat prüfen, falls das Target-File kürzer ist als der Input

        # Check Input
        current_width_in = cnn_in.shape[2]
        if current_width_in < self.crop_width:
            pad_w = self.crop_width - current_width_in
            cnn_in = np.pad(cnn_in, ((0, 0), (0, 0), (0, pad_w)), mode='constant')

        # Check Target (Das hat gefehlt!)
        current_width_tgt = cnn_tgt.shape[2]
        if current_width_tgt < self.crop_width:
            pad_w = self.crop_width - current_width_tgt
            cnn_tgt = np.pad(cnn_tgt, ((0, 0), (0, 0), (0, pad_w)), mode='constant')

        # 6. Slicing (Falls durch Rundung 1 Frame zu viel da ist)
        cnn_in = cnn_in[:, :, :self.crop_width]
        cnn_tgt = cnn_tgt[:, :, :self.crop_width]

        return torch.from_numpy(cnn_in).float(), torch.from_numpy(cnn_tgt).float()