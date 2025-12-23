import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import os

class AudioProcessor:
    def __init__(self, sample_rate: int = 96000, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.original_subtype = None  # Speichert Bit-Tiefe der Eingabedatei

    def load_audio(self, file_path: str) -> np.ndarray:
        """Lädt Audio als Stereo (2, samples). Konvertiert Mono zu Stereo."""
        # Speichere Original-Subtype für späteres Speichern
        info = sf.info(file_path)
        self.original_subtype = info.subtype

        # mono=False ist wichtig für Stereo
        y, _ = librosa.load(file_path, sr=self.sample_rate, mono=False)

        # Falls Mono geladen wurde (1D Array), zu (1, samples) erweitern
        if y.ndim == 1:
            y = y[np.newaxis, :]

        # Falls wirklich Mono (1 Channel), duplizieren zu Pseudo-Stereo für Konsistenz
        if y.shape[0] == 1:
            y = np.concatenate([y, y], axis=0)

        return y

    def save_audio(self, file_path: str, waveform: np.ndarray) -> None:
        """Speichert Audio. Erwartet (Channels, Samples), Soundfile braucht (Samples, Channels)."""
        # Verwende gleiche Bit-Tiefe wie Eingabedatei (falls verfügbar)
        subtype = self.original_subtype if self.original_subtype else 'PCM_16'
        sf.write(file_path, waveform.T, self.sample_rate, subtype=subtype)

    def audio_to_stft(self, waveform: np.ndarray) -> np.ndarray:
        """Erzeugt komplexes STFT. Output: (Channels, Freq, Time)."""
        stft_channels = []
        for channel_idx in range(waveform.shape[0]):
            stft = librosa.stft(waveform[channel_idx], n_fft=self.n_fft, hop_length=self.hop_length)
            stft_channels.append(stft)
        return np.array(stft_channels)

    def stft_to_audio(self, stft_matrix: np.ndarray, original_length: int = None) -> np.ndarray:
        """Rekonstruiert Audio via iSTFT."""
        audio_channels = []
        for channel_idx in range(stft_matrix.shape[0]):
            y = librosa.istft(stft_matrix[channel_idx], hop_length=self.hop_length, length=original_length)
            audio_channels.append(y)
        return np.array(audio_channels)

    def stft_to_cnn_input(self, stft_matrix: np.ndarray) -> np.ndarray:
        """
        Wandelt Stereo-STFT in 4-Kanal-Tensor (Real/Imag).
        Strategie: Real/Imag ist linear und MSE-freundlich (besser als Mag/Phase).
        Output: (4, Freq, Time) -> [L_Real, L_Imag, R_Real, R_Imag]
        """
        cnn_layers = []
        for c in range(stft_matrix.shape[0]):
            # Real und Imaginärteil splitten
            # Skalierung * 0.1 bringt Werte grob in Bereich -1..1 für das Netz
            re = np.real(stft_matrix[c]) * 0.1
            im = np.imag(stft_matrix[c]) * 0.1
            cnn_layers.extend([re, im])

        return np.stack(cnn_layers, axis=0)

    def cnn_input_to_stft(self, cnn_tensor: np.ndarray) -> np.ndarray:
        """
        Kehrt die Real/Imag Transformation um.
        Input: (4, Freq, Time) oder (Batch, 4, Freq, Time)
        Output: (2, Freq, Time) komplex
        """
        # Falls Batch-Dimension existiert, muss das hier handled werden (meistens aber nicht nötig im Processor)
        if cnn_tensor.ndim == 4:
            # Falls du Tensor direkt übergibst: (Batch, 4, H, W) -> nicht Standard für diese Klasse,
            # aber falls es passiert, nimm das erste Element oder iteriere.
            # Hier gehen wir von 3D (Channels, H, W) aus (einzelnes Sample).
            pass

        stft_channels = []
        # Wir erwarten 4 Kanäle -> 2 Audio Kanäle
        num_audio_channels = cnn_tensor.shape[0] // 2

        for c in range(num_audio_channels):
            # Indizes: 0,1 für Links -> 2,3 für Rechts
            re_idx = c * 2
            im_idx = c * 2 + 1

            re = cnn_tensor[re_idx]
            im = cnn_tensor[im_idx]

            # Skalierung rückgängig (* 10.0) und komplex zusammensetzen
            stft = (re + 1j * im) * 10.0
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