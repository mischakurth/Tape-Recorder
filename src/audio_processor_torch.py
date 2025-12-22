import torch
import torchaudio
import torch.nn.functional as F


class TorchAudioProcessor:
    def __init__(self, sample_rate=96000, n_fft=2048, hop_length=512, device='cpu'):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft).to(device)
        self.device = device

        # FIX: Auf 1056 erhöht.
        # 1056 / 16 = 66 (Gerade Zahl -> Sauberes Pooling)
        # 1040 / 16 = 65 (Ungerade Zahl -> Risiko für Mismatches)
        self.target_freq = 1056

    def load_audio(self, file_path):
        """Lädt Audio direkt als Tensor auf die GPU/CPU"""
        # Torchaudio lädt (Channels, Time)
        y, sr = torchaudio.load(file_path)

        # Resampling
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(y.device)
            y = resampler(y)

        # Mono -> Stereo
        if y.shape[0] == 1:
            y = y.repeat(2, 1)

        return y.to(self.device)

    def audio_to_cnn(self, audio_tensor):
        """
        Wandelt Audio in CNN-Input um (mit Padding auf 1056px Höhe).
        Input: (Batch, Channels, Time)
        Output: (Batch, 4, 1056, Time)
        """
        # 1. Sicherstellen, dass wir 3 Dimensionen haben
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        # 2. Flattening für torch.stft (mag keine Batch+Channels Dim)
        batch_size, channels, time_steps = audio_tensor.shape
        flat_audio = audio_tensor.reshape(batch_size * channels, time_steps)

        # 3. STFT Berechnung
        stft = torch.stft(
            flat_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
            normalized=False,
            onesided=True
        )
        # Output: (B*C, Freq=1025, Time)

        # 4. Reshape zurück zu (Batch, Channels, Freq, Time)
        freqs = stft.shape[1]
        frames = stft.shape[2]
        stft = stft.view(batch_size, channels, freqs, frames)

        # 5. Padding (Freq: 1025 -> 1056)
        pad_h = self.target_freq - freqs
        if pad_h > 0:
            # (TimeLeft, TimeRight, FreqTop, FreqBottom)
            stft = F.pad(stft, (0, 0, 0, pad_h))

        # 6. Real/Imag Splitting
        real = stft.real
        imag = stft.imag

        # Stacken zu 4 Channels
        cnn_input = torch.cat([real, imag], dim=1)

        return cnn_input

    def cnn_to_audio(self, cnn_tensor, length=None):
        """
        Wandelt CNN-Input zurück in Audio (schneidet Padding ab).
        Input: (Batch, 4, Freq, Time)
        """
        # 1. Padding entfernen (1056 -> 1025)
        original_n_bins = self.n_fft // 2 + 1

        if cnn_tensor.shape[2] > original_n_bins:
            cnn_tensor = cnn_tensor[:, :, :original_n_bins, :]

        # 2. Split in Real/Imag
        channels = cnn_tensor.shape[1] // 2
        real = cnn_tensor[:, :channels, :, :]
        imag = cnn_tensor[:, channels:, :, :]

        complex_stft = torch.complex(real, imag)

        # 3. Flattening für ISTFT
        batch_size, ch, freqs, frames = complex_stft.shape
        flat_stft = complex_stft.reshape(batch_size * ch, freqs, frames)

        # 4. ISTFT
        audio = torch.istft(
            flat_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            length=length
        )

        # 5. Reshape zurück
        new_time = audio.shape[1]
        audio = audio.view(batch_size, channels, new_time)

        return audio