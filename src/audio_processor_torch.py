import torch
import torchaudio
import numpy as np


class TorchAudioProcessor:
    def __init__(self, sample_rate=96000, n_fft=2048, hop_length=512, device='cpu'):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft).to(device)
        self.device = device

    def load_audio(self, file_path):
        """Lädt Audio direkt als Tensor auf die GPU/CPU"""
        # Torchaudio lädt (Channels, Time)
        y, sr = torchaudio.load(file_path)

        # Resampling falls nötig
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(y.device)
            y = resampler(y)

        # Mono zu Stereo (falls nötig)
        if y.shape[0] == 1:
            y = y.repeat(2, 1)

        return y.to(self.device)

    def audio_to_cnn(self, audio_tensor):
        """
        Input: (Batch, Channels, Time) oder (Channels, Time)
        Output: (Batch, 4, Freq, Frames) - Real/Imag Interleaved
        """
        # Sicherstellen, dass wir 3 Dimensionen haben (Batch, Channel, Time)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        # STFT auf GPU
        # Output shape: (Batch, Channel, Freq, Time) complex64
        stft = torch.stft(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True  # Wichtig für perfektes Padding
        )

        # Wir zerlegen Complex in Real und Imaginärteil für das CNN
        # stft.real -> (Batch, 2, 1025, Time)
        # Wir wollen (Batch, 4, 1025, Time)
        real = stft.real
        imag = stft.imag

        # Stacken: Channel 0 Real, Channel 0 Imag, Channel 1 Real, ...
        # Das ist effizienter für das Netz
        cnn_input = torch.cat([real, imag], dim=1)

        return cnn_input

    def cnn_to_audio(self, cnn_tensor, length=None):
        """
        Kehrt den Prozess um: (Batch, 4, Freq, Frames) -> Audio
        Differentiable! Das ist der Schlüssel für besseren Sound.
        """
        # Zurück splitten in Real/Imag
        # Annahme: Input ist (Batch, 4, Freq, Time)
        # Channel 0+1 sind Stereo-Links (Real/Imag), 2+3 sind Stereo-Rechts (Real/Imag)
        # ODER: 0+1 sind Real (L/R), 2+3 sind Imag (L/R).
        # Hängt vom `cat` oben ab. Oben haben wir [real, imag] gemacht (Dim 1).
        # real war (B, 2, F, T), imag war (B, 2, F, T).
        # Also sind Kanal 0,1 = Real(L), Real(R) und 2,3 = Imag(L), Imag(R).

        channels = cnn_tensor.shape[1] // 2
        real = cnn_tensor[:, :channels, :, :]
        imag = cnn_tensor[:, channels:, :, :]

        complex_stft = torch.complex(real, imag)

        # ISTFT
        audio = torch.istft(
            complex_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=length  # Optional: Exakte Länge erzwingen
        )

        return audio