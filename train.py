from datetime import datetime
import random
import gc
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.model import AudioUNet
from src.dataset import StreamingAudioDataset
from src.audio_processor import AudioProcessor
import os

# --- Hilfsfunktion zum Finden der Dateien ---
def find_paired_files(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """
    Findet Paare unabhängig von der Ordnerstruktur (ignoriert Unterordner).
    Erstellt eine 'Map' aller Output-Dateien für extrem schnelles Finden.
    """
    pairs = []
    print(f"--- Scanne Ordner (ignoriere Unterordner-Struktur)... ---")

    # 1. Wir bauen ein "Telefonbuch" aller Output-Dateien
    # Key: Dateiname (z.B. "song-taped.wav"), Value: Der echte Pfad
    output_map = {}

    # rglob('*') sucht rekursiv in ALLEN Unterordnern
    for f in output_dir.rglob('*.wav'):
        if not f.name.startswith('.'):  # Versteckte Dateien ignorieren
            output_map[f.name] = f

    print(f"   -> {len(output_map)} Output-Dateien im Index gefunden.")

    # 2. Input Dateien durchgehen und im "Telefonbuch" nachschlagen
    input_cnt = 0
    for input_file in sorted(input_dir.rglob('*.wav')):
        if input_file.name.startswith('.'): continue
        input_cnt += 1

        # Generiere mögliche Namen für den Output
        stem = input_file.stem
        suffix = input_file.suffix

        # Liste aller möglichen Schreibweisen, die wir akzeptieren
        candidates = [
            stem + "-taped" + suffix,  # Standard
            stem + "_taped" + suffix,  # Unterstrich
            stem + " taped" + suffix,  # Leerzeichen
            input_file.name,  # Identischer Name
            input_file.name.replace('vorher', 'taped'),
            input_file.name.replace('orig', 'tape')
        ]

        # Prüfen, ob EINER der Kandidaten in unserer Map existiert
        match_found = False
        for candidate in candidates:
            if candidate in output_map:
                # Treffer! Wir nehmen den Pfad aus der Map
                output_path = output_map[candidate]
                pairs.append((input_file, output_path))
                match_found = True
                break  # Suche für dieses File beenden

        if not match_found:
            # Debugging: Nur einschalten wenn du wissen willst, was fehlt
            # print(f"Miss: {input_file.name}")
            pass

    print(f"   -> {len(pairs)} Paare erfolgreich gematched (von {input_cnt} Inputs).")
    return pairs


def get_all_file_pairs(data_root: Path, target_dataset: str | None = None) -> list[tuple[Path, Path]]:
    all_pairs = []

    if target_dataset:
        datasets = [data_root / target_dataset]
    else:
        datasets = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("dataset-")]

    for dataset_path in datasets:
        if not dataset_path.exists():
            print(f"Warnung: Dataset {dataset_path} nicht gefunden.")
            continue

        input_dir = dataset_path / "tape-input"
        output_dir = dataset_path / "tape-output-recordings"

        if input_dir.exists() and output_dir.exists():
            pairs = find_paired_files(input_dir, output_dir)
            all_pairs.extend(pairs)
            print(f"Dataset {dataset_path.name}: {len(pairs)} Paare gefunden.")
        else:
            print(f"Warnung: Ordnerstruktur unvollständig in {dataset_path.name}")

    return all_pairs


def main():
    # 1. Config
    target_dataset = "dataset-test"
    # target_dataset = None  # Nimm None für ALLES (empfohlen für das finale Training)

    data_root = Path("data/audio/datasets")

    # Pfad zum Modell
    project_root = Path(__file__).parent.resolve()
    models_dir = project_root / "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = models_dir / "unet_test_best.pth"  # Nennen wir es 'best', das ist üblicher

    # --- Training Config ---
    batch_size = 24  # Perfekt für M3 Pro
    lr = 1e-4
    epochs = 5  # Da SlidingWindow riesig ist, reichen oft weniger Epochen!
    load_pretrained = True  # Setze auf True, um Training fortzusetzen!
    pretrained_model_path = models_dir / "unet_v1_alfred_best.pth"  # Pfad zum gewünschten Modell"

    # Daten laden
    all_pairs = get_all_file_pairs(data_root, target_dataset)
    print(f"Gesamt: {len(all_pairs)} Datei-Paare gefunden.")

    if not all_pairs:
        print("Keine Trainingsdaten gefunden.")
        return

    # Split
    random.seed(42)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.80)  # 80% Training ist Standard
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"Daten Split: {len(train_pairs)} Files Training, {len(val_pairs)} Files Validierung")

    # Check Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training auf: {device}")

    # 2. Objekte erstellen
    proc = AudioProcessor(sample_rate=96000)

    # num_workers austesten

    # --- Training Dataset ---
    print("Initialisiere Training Dataset...")
    train_dataset = StreamingAudioDataset(train_pairs, proc, crop_width=256, overlap=0.5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # prefetch_factor=2,  # <--- Wichtig für Streaming
    )

    # --- Validation Dataset ---
    print("Initialisiere Validation Dataset...")
    val_dataset = StreamingAudioDataset(val_pairs, proc, crop_width=256, overlap=0.0)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        # prefetch_factor=2,     # <--- Wichtig für Streaming
    )

    print(f"Total Training Chunks: {len(train_dataset)}")
    print(f"Total Validation Chunks: {len(val_dataset)}")

    # Modell setup
    model = AudioUNet(n_channels=4, n_classes=4).to(device)

    # --- RESUME LOGIC ---
    if load_pretrained and pretrained_model_path.exists():
        print(f"Lade existierendes Modell von {pretrained_model_path}...")
        state_dict = torch.load(pretrained_model_path, map_location=device)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"❌ Unerwarteter Fehler beim Laden: {e}")
    else:
        print("Starte neues Training.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- WICHTIGE ÄNDERUNG: L1 Loss ---
    loss_fn = torch.nn.L1Loss()  # Besser für Audio als MSE

    # 3. Training Loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        train_loss = 0.0

        print(f"Starte Epoch {epoch + 1}/{epochs}...")
        print("um Uhrzeit", datetime.now())
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Kleines Feedback alle 50 Batches
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDIERUNG ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"==> Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # --- WICHTIG: MPS SPEICHER BEREINIGUNG ---
        # Das verhindert, dass der RAM/VRAM volläuft und das Training langsam wird.
        if device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()  # Python Garbage Collector forcieren

        # --- Speichern (Nur wenn es das beste bisher ist) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"   Neuer Bestwert! Modell gespeichert unter: {model_path}")

        # Optional: Immer den letzten Stand auch speichern, falls Absturz
        torch.save(model.state_dict(), models_dir / "unet_test_last.pth")




if __name__ == "__main__":
    main()