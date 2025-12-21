import random
import gc
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.model import AudioUNet
from src.dataset import SlidingWindowDataset
from src.audio_processor import AudioProcessor
import os


# --- Hilfsfunktion zum Finden der Dateien ---
def find_paired_files(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """
    Findet Paare von Eingabe- und Ausgabe-Dateien basierend auf der Namenskonvention.
    """
    pairs = []
    # Rekursive Suche nach wav Dateien
    for input_file in sorted(input_dir.rglob('*.wav')):
        if 'vorher' in input_file.name:
            out_name = input_file.name.replace('vorher', 'taped')
        elif 'orig' in input_file.name:
            out_name = input_file.name.replace('orig', 'tape')
        else:
            out_name = input_file.name

        possible_output = output_dir / out_name

        if possible_output.exists():
            pairs.append((input_file, possible_output))

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
    model_path = models_dir / "unet_v1_best.pth"  # Nennen wir es 'best', das ist üblicher

    # --- Training Config ---
    batch_size = 16  # Perfekt für M3 Pro
    lr = 1e-4
    epochs = 10  # Da SlidingWindow riesig ist, reichen oft weniger Epochen!
    load_pretrained = False  # Setze auf True, um Training fortzusetzen!

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

    # --- Training Dataset ---
    print("Initialisiere Training Dataset...")
    train_dataset = SlidingWindowDataset(train_pairs, proc, crop_width=256, overlap=0.5)

    # Automatische Worker-Entscheidung
    if train_dataset.use_preload:
        print("Dataset im RAM: Setze num_workers=0 (schnellster Modus)")
        train_workers = 0
    else:
        print("Dataset auf Disk: Setze num_workers=4 (Hintergrund-Laden)")
        train_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=False
    )

    # --- Validation Dataset ---
    print("Initialisiere Validation Dataset...")
    val_dataset = SlidingWindowDataset(val_pairs, proc, crop_width=256, overlap=0.0)

    # Auch hier automatisch prüfen
    val_workers = 0 if val_dataset.use_preload else 2

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=False
    )

    print(f"Total Training Chunks: {len(train_dataset)}")
    print(f"Total Validation Chunks: {len(val_dataset)}")

    # Modell setup
    model = AudioUNet(n_channels=4, n_classes=4).to(device)

    # --- RESUME LOGIC ---
    if load_pretrained and model_path.exists():
        print(f"Lade existierendes Modell von {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
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
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Kleines Feedback alle 100 Batches
            if batch_idx % 100 == 0:
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
        torch.save(model.state_dict(), models_dir / "unet_last.pth")


if __name__ == "__main__":
    main()