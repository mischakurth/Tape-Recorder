import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from typing import Tuple
import os


class AudioProcessor:
    def __init__(self, sample_rate: int = 96000, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def load_audio(self, file_path: str) -> np.ndarray:
        # mono=False lädt Stereo als (2, samples)
        # Wenn die Datei Mono ist, kommt trotzdem (samples,) zurück, daher der Reshape-Check
        y, _ = librosa.load(file_path, sr=self.sample_rate, mono=False)

        # Sicherstellen, dass wir immer (Channels, Samples) haben
        if y.ndim == 1:
            y = y[np.newaxis, :]  # Erweitere zu (1, samples)
        return y

    def save_audio(self, file_path: str, waveform: np.ndarray) -> None:
        # Soundfile erwartet (Samples, Channels), Librosa liefert (Channels, Samples)
        # Wir müssen transponieren (.T)
        sf.write(file_path, waveform.T, self.sample_rate)

    def audio_to_stft(self, waveform: np.ndarray) -> np.ndarray:
        # Wir iterieren über die Kanäle (z.B. Links und Rechts)
        # Input Shape: (Channels, Samples) -> Output Shape: (Channels, Freqs, Frames)
        stft_channels = []
        for channel_idx in range(waveform.shape[0]):
            stft = librosa.stft(waveform[channel_idx], n_fft=self.n_fft, hop_length=self.hop_length)
            stft_channels.append(stft)
        return np.array(stft_channels)

    def stft_to_audio(self, stft_matrix: np.ndarray, original_length: int = None) -> np.ndarray:
        # Input Shape: (Channels, Freqs, Frames)
        audio_channels = []
        for channel_idx in range(stft_matrix.shape[0]):
            y = librosa.istft(stft_matrix[channel_idx], hop_length=self.hop_length, length=original_length)
            audio_channels.append(y)
        return np.array(audio_channels)

    # --- Feature Engineering ---
    def separate_magnitude_phase(self, stft_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.abs(stft_matrix), np.angle(stft_matrix)

    def combine_magnitude_phase(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        return magnitude * np.exp(1j * phase)

    def normalize(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Normalisiert Amplitude zu DB.
        Änderungen für Lossless-Verhalten:
        - amin=1e-10: Erlaubt extrem leise Signale ohne Clipping (Standard war 1e-5).
        - top_db=None: Deaktiviert das Abschneiden leiser Frequenzen (Standard war 80dB).
        """
        return librosa.amplitude_to_db(magnitude, ref=1.0, amin=1e-10, top_db=None)

    def denormalize(self, magnitude_db: np.ndarray) -> np.ndarray:
        """
        Kehrt die Log-Skalierung um.
        ref=1.0 muss mit normalize übereinstimmen.
        """
        return librosa.db_to_amplitude(magnitude_db, ref=1.0)

    # --- CNN Interfaces ---
    def stft_to_cnn_input(self, stft_matrix: np.ndarray) -> np.ndarray:
        """
        Wandelt Stereo-STFT in 4-Kanal-Tensor.
        Input: (2, Freq, Time) complex
        Output: (4, Freq, Time) float -> [L_Mag, L_Phase, R_Mag, R_Phase]
        """
        channels, freq, time = stft_matrix.shape
        cnn_layers = []

        for c in range(channels):
            mag, phase = self.separate_magnitude_phase(stft_matrix[c])
            mag_db = self.normalize(mag)  # Deine korrigierte normalize Methode (ref=1.0)
            cnn_layers.append(mag_db)
            cnn_layers.append(phase)

        # Stacken entlang der ersten Achse
        return np.stack(cnn_layers, axis=0)

    def cnn_input_to_stft(self, cnn_tensor: np.ndarray) -> np.ndarray:
        """
        Input: (4, Freq, Time) -> [L_Mag, L_Phase, R_Mag, R_Phase]
        Output: (2, Freq, Time) complex
        """
        total_layers = cnn_tensor.shape[0]
        # Wir nehmen an: 2 Layer pro Audio-Kanal (Mag + Phase)
        num_audio_channels = total_layers // 2

        stft_channels = []

        for c in range(num_audio_channels):
            # Layer Indizes berechnen: 0,1 für Links -> 2,3 für Rechts
            mag_idx = c * 2
            phase_idx = c * 2 + 1

            mag_db = cnn_tensor[mag_idx]
            phase = cnn_tensor[phase_idx]

            mag = self.denormalize(mag_db)  # Deine korrigierte denormalize Methode
            stft = self.combine_magnitude_phase(mag, phase)
            stft_channels.append(stft)

        return np.array(stft_channels)


# --- TEST SKRIPT ---
if __name__ == "__main__":
    # 1. Setup
    test_dir = os.path.join(os.path.expanduser("~"), "Downloads", "test files")
    filename = os.path.join(test_dir, "test sweep 30sec V2 490-8900Hz 2024-03-31 vorher.wav")
    processor = AudioProcessor(sample_rate=96000)

    print(f"1. Lade {filename}...")
    try:
        y_original = processor.load_audio(filename)
    except FileNotFoundError:
        print("Datei nicht gefunden. Erzeuge Dummy-Sinuston...")
        y_original = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, 96000 * 2))  # 2s Sinus

    # 2. Forward Pass (Audio -> Bild-Tensor)
    print("2. Encoding: Audio -> STFT -> CNN Tensor...")
    stft_complex = processor.audio_to_stft(y_original)
    cnn_input = processor.stft_to_cnn_input(stft_complex)

    print(f"   Shape des CNN-Inputs: {cnn_input.shape}")
    print(f"   (Kanäle, Frequenzen, Zeitframes)")

    # 3. Visualisierung
    print("3. Visualisiere Daten...")
    plt.figure(figsize=(12, 8))

    # Kanal 0: Magnitude (Log-Spectrogram)
    plt.subplot(2, 1, 1)
    librosa.display.specshow(cnn_input[0], sr=processor.sample_rate, hop_length=processor.hop_length, x_axis='time',
                             y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Kanal 1: Magnitude (dB) - Was wir sehen')

    # Kanal 1: Phase
    plt.subplot(2, 1, 2)
    librosa.display.specshow(cnn_input[1], sr=processor.sample_rate, hop_length=processor.hop_length, x_axis='time',
                             y_axis='hz')
    plt.colorbar()
    plt.title('Kanal 2: Phase (Radiant) - Die Strukturinformation')

    plt.tight_layout()
    plt.show()

    # 4. Backward Pass (Bild-Tensor -> Audio)
    print("4. Decoding: CNN Tensor -> STFT -> Audio...")
    stft_recon = processor.cnn_input_to_stft(cnn_input)
    y_recon = processor.stft_to_audio(stft_recon, original_length=y_original.shape[-1])

    # 5. Validierung
    # Toleranz etwas höher wegen float32 Rundungsfehlern bei log/exp
    mse = np.mean((y_original - y_recon) ** 2)
    is_close = np.allclose(y_original, y_recon, atol=1e-5)

    print("-" * 30)
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Rekonstruktion erfolgreich (Toleranz 1e-5)? {'JA' if is_close else 'NEIN'}")

    output_filename = os.path.join(test_dir, "test_reconstructed.wav")
    processor.save_audio(output_filename, y_recon)
    print(f"Rekonstruierte Datei gespeichert unter: {output_filename}")