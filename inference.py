import torch
import numpy as np
import os
from pathlib import Path
from src.audio_processor import AudioProcessor
from src.model import AudioUNet
import soundfile as sf


def run_inference(input_file: str, model_path: str, output_file: str, duration_sec: float = 10.0):
    # 1. Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Lade Modell auf {device}...")

    # Processor & Model
    proc = AudioProcessor(sample_rate=96000)
    model = AudioUNet(n_channels=4, n_classes=4).to(device)

    # Gewichte laden
    if not os.path.exists(model_path):
        print(f"Fehler: Modell nicht gefunden unter {model_path}")
        return

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # Wichtig: Schaltet Dropout/BatchNorm in den Test-Modus

    # 2. Audio Laden
    print(f"Lade Audio: {input_file}")
    y = proc.load_audio(input_file)

    # Auf gewünschte Länge kürzen (z.B. erste 30 Sek), um RAM zu sparen
    # Falls duration_sec None ist, nimm alles
    if duration_sec:
        max_samples = int(duration_sec * 96000)
        if y.shape[1] > max_samples:
            y = y[:, :max_samples]

    # 3. Preprocessing & Padding
    stft = proc.audio_to_stft(y)
    cnn_input = proc.stft_to_cnn_input(stft)

    # Input Shape ist (4, 1025, Time)
    c, h, w = cnn_input.shape

    # A) Frequenz Padding (auf 1040)
    target_h = 1040
    pad_h = target_h - h

    # B) Time Padding (auf nächstes Vielfaches von 16 für U-Net)
    target_w = ((w - 1) // 16 + 1) * 16
    pad_w = target_w - w

    print(f"Padding Input: Freq {h}->{target_h}, Time {w}->{target_w}")

    # Pad input: ((Before, After) für Channel, Freq, Time)
    padded_input = np.pad(cnn_input, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')

    # Zu Torch Tensor & Batch Dimension hinzufügen
    tensor_in = torch.from_numpy(padded_input).float().unsqueeze(0).to(device)

    # 4. Forward Pass
    print("Modell rechnet...")
    with torch.no_grad():
        prediction = model(tensor_in)

    # 5. Postprocessing
    pred_np = prediction.squeeze(0).cpu().numpy()

    # Padding wieder abschneiden
    pred_np = pred_np[:, :1025, :w]

    # Zurückrechnen
    stft_recon = proc.cnn_input_to_stft(pred_np)
    y_recon = proc.stft_to_audio(stft_recon, original_length=y.shape[1])

    # 6. Speichern
    proc.save_audio(output_file, y_recon)
    print(f"Fertig! Gespeichert als: {output_file}")

    # --- 7. VERGLEICH (Validation) ---
    # Sucht nach -9tape.wav im selben Ordner wie der Input
    input_path_obj = Path(input_file)
    ref_filename = "-9tape.wav"
    ref_path = input_path_obj.parent / ref_filename

    if ref_path.exists():
        print(f"\n--- Starte Vergleich mit {ref_filename} ---")

        # Referenz laden (nutzt proc für konsistente Samplerate)
        y_ref = proc.load_audio(str(ref_path))

        # Längen angleichen: Wir nehmen das Minimum von beiden
        # (Falls y_recon durch duration_sec gekürzt wurde oder das Tape minimal anders ist)
        min_len = min(y_recon.shape[1], y_ref.shape[1])

        # Signale zuschneiden
        sig_gen = y_recon[:, :min_len]
        sig_ref = y_ref[:, :min_len]

        # 1. Metrik: Mean Squared Error (MSE) - Je kleiner desto besser
        mse = np.mean((sig_gen - sig_ref) ** 2)

        # 2. Metrik: Cross-Correlation (Pearson) - 1.0 ist perfekt
        # Wir 'flatten' die Arrays (Stereo -> 1 lange Reihe), um einen Gesamtwert zu bekommen
        flat_gen = sig_gen.flatten()
        flat_ref = sig_ref.flatten()

        # Korrelationskoeffizient berechnen
        correlation = np.corrcoef(flat_gen, flat_ref)[0, 1]

        print(f"Validation Duration: {min_len / 96000:.2f} sek")
        print(f"MSE Loss (Fehler):   {mse:.6f}")
        print(f"Cross-Correlation:   {correlation:.4f}  (1.0 = Identisch)")

    else:
        print(f"\nWarnung: Referenzdatei '{ref_filename}' nicht im Ordner gefunden. Kein Vergleich möglich.")


if __name__ == "__main__":
    # Pfade anpassen!
    base_path = Path(os.path.expanduser("~")) / "Downloads" / "test files"

    # Input (orig oder tape, je nachdem was du reinsteckst)
    input_wav = base_path / "-9orig.wav"

    # Pfad zum Modell
    script_dir = Path(__file__).parent
    model_file = script_dir / "models" / "unet_test_best.pth"

    output_wav = base_path / "test_inference_result.wav"

    if not input_wav.exists():
        print(f"Fehler: Input-Datei nicht gefunden: {input_wav}")
    elif not model_file.exists():
        print(f"Fehler: Modell-Datei nicht gefunden: {model_file}")
    else:
        # Duration: None für ganze Datei, oder float (z.B. 10.0) für kurze Tests
        duration_sec = 10
        run_inference(str(input_wav), str(model_file), str(output_wav), duration_sec=duration_sec)