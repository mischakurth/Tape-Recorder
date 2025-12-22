import os
import random
import gc
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Eigene Module
from src.model import AudioUNet
from src.dataset import StreamingAudioDataset
from src.audio_processor_torch import TorchAudioProcessor


# --- Hilfsfunktion zum Finden der Dateien ---
def find_paired_files(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    print(f"--- Scanne Ordner (ignoriere Unterordner-Struktur)... ---")
    output_map = {f.name: f for f in output_dir.rglob('*.wav') if not f.name.startswith('.')}

    for input_file in sorted(input_dir.rglob('*.wav')):
        if input_file.name.startswith('.'): continue
        stem = input_file.stem
        suffix = input_file.suffix
        candidates = [
            stem + "-taped" + suffix, stem + "_taped" + suffix,
            stem + " taped" + suffix, input_file.name,
            input_file.name.replace('vorher', 'taped'),
            input_file.name.replace('orig', 'tape')
        ]
        for candidate in candidates:
            if candidate in output_map:
                pairs.append((input_file, output_map[candidate]))
                break
    return pairs


def get_all_file_pairs(data_root: Path, target_dataset: str | None = None) -> list[tuple[Path, Path]]:
    all_pairs = []
    datasets = [data_root / target_dataset] if target_dataset else [d for d in data_root.iterdir() if
                                                                    d.is_dir() and d.name.startswith("dataset-")]

    for dataset_path in datasets:
        if not dataset_path.exists(): continue
        input_dir = dataset_path / "tape-input"
        output_dir = dataset_path / "tape-output-recordings"
        if input_dir.exists() and output_dir.exists():
            pairs = find_paired_files(input_dir, output_dir)
            all_pairs.extend(pairs)
    return all_pairs


# --- PERCEPTUAL LOSS ERSATZ (Stabil auf MPS) ---
def log_spectral_loss(pred_cnn, target_cnn):
    """
    Berechnet den Fehler im Log-Bereich (dB).
    Das entspricht der menschlichen Wahrnehmung, benötigt aber keine
    fehleranfällige ISTFT-Umwandlung.
    """
    # 1. Wir müssen Real/Imag in Magnitude (Lautstärke) umwandeln
    # Input Shape: (Batch, 4, Freq, Time) -> Channels 0/1 sind Real/Imag (L), 2/3 sind (R)

    # Links
    real_L = pred_cnn[:, 0, :, :]
    imag_L = pred_cnn[:, 1, :, :]
    mag_pred_L = torch.sqrt(real_L ** 2 + imag_L ** 2 + 1e-7)

    real_tgt_L = target_cnn[:, 0, :, :]
    imag_tgt_L = target_cnn[:, 1, :, :]
    mag_tgt_L = torch.sqrt(real_tgt_L ** 2 + imag_tgt_L ** 2 + 1e-7)

    # Rechts
    real_R = pred_cnn[:, 2, :, :]
    imag_R = pred_cnn[:, 3, :, :]
    mag_pred_R = torch.sqrt(real_R ** 2 + imag_R ** 2 + 1e-7)

    real_tgt_R = target_cnn[:, 2, :, :]
    imag_tgt_R = target_cnn[:, 3, :, :]
    mag_tgt_R = torch.sqrt(real_tgt_R ** 2 + imag_tgt_R ** 2 + 1e-7)

    # 2. Logarithmus (Dezibel-Ebene)
    # Das ist der "Perceptual" Teil: Wir vergleichen Lautheit, nicht Energie.
    log_pred_L = torch.log(mag_pred_L)
    log_tgt_L = torch.log(mag_tgt_L)

    log_pred_R = torch.log(mag_pred_R)
    log_tgt_R = torch.log(mag_tgt_R)

    # 3. L1 Loss auf den Log-Werten
    loss_L = F.l1_loss(log_pred_L, log_tgt_L)
    loss_R = F.l1_loss(log_pred_R, log_tgt_R)

    return (loss_L + loss_R) / 2


def main():
    # --- Config ---
    # target_dataset = "dataset-alfred"
    target_dataset = None
    data_root = Path("data/audio/datasets")

    models_dir = Path(__file__).parent.resolve() / "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = models_dir / "unet_log_spectral_all_best.pth"

    batch_size = 24
    lr = 1e-4
    epochs = 20

    # --- Setup ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training auf: {device}")

    proc = TorchAudioProcessor(device=device)

    # Daten
    all_pairs = get_all_file_pairs(data_root, target_dataset)
    if not all_pairs: return

    random.seed(42)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.9)
    train_pairs, val_pairs = all_pairs[:split_idx], all_pairs[split_idx:]

    print(f"Training: {len(train_pairs)} | Validation: {len(val_pairs)}")

    train_loader = DataLoader(StreamingAudioDataset(train_pairs, proc), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(StreamingAudioDataset(val_pairs, proc, overlap=0.0), batch_size=batch_size, shuffle=False)

    model = AudioUNet(n_channels=4, n_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        print(f"Starte Epoch {epoch + 1}/{epochs} um {datetime.now().strftime('%H:%M:%S')}...")

        for batch_idx, (x_audio, y_audio) in enumerate(train_loader):
            x_audio, y_audio = x_audio.to(device), y_audio.to(device)

            # 1. Audio -> Spec
            x_spec = proc.audio_to_cnn(x_audio)
            y_spec = proc.audio_to_cnn(y_audio)

            # 2. Modell
            pred_spec = model(x_spec)

            # 3. Log-Spectral Loss (Perceptual aber stabil)
            # Wir verzichten auf Audio-Rückumwandlung im Loop!
            loss = log_spectral_loss(pred_spec, y_spec)

            if torch.isnan(loss):
                print("⚠️ NaN Loss detected. Skipping.")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()

            # Leichtes Clipping zur Sicherheit
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)} | Loss (dB): {loss.item():.4f}")

        avg_train = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # --- Validation ---
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x_audio, y_audio in val_loader:
                x_audio, y_audio = x_audio.to(device), y_audio.to(device)

                x_spec = proc.audio_to_cnn(x_audio)
                y_spec = proc.audio_to_cnn(y_audio)

                pred_spec = model(x_spec)

                # Wir nehmen den gleichen Loss für Val
                val_total += log_spectral_loss(pred_spec, y_spec).item()

        avg_val = val_total / len(val_loader) if len(val_loader) > 0 else 0

        print(f"==> Epoch {epoch + 1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), model_path)
            print("Neue Bestleistung, Modell gespeichert.")

        if device.type == 'mps': torch.mps.empty_cache()


if __name__ == "__main__":
    main()