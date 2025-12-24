from datetime import datetime
import random
import gc
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.model import AudioUNet, ResidualBlock, DoubleConv
from src.dataset import StreamingAudioDataset
from src.audio_processor import AudioProcessor
from src.experiment_manager import ExperimentManager
import os
import argparse

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


def parse_args():
    parser = argparse.ArgumentParser(description="Audio Training Script")

    # Hier definierst du die DEFAULTS für PyCharm (Wenn du nichts angibst)
    parser.add_argument("--variant", type=str, default="standard", choices=["standard", "resnet"],
                        help="Modell Architektur")
    parser.add_argument("--dataset", type=str, default="All", help="Name des Datasets (oder 'All' für alle)")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch Size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=10, help="Anzahl Epochen")
    # Flags (Schalter): Wenn sie da sind = True, sonst = False
    parser.add_argument("--resume", action="store_true", help="Versuche letztes Training fortzusetzen")

    return parser.parse_args()


def main():
    # Argumente lesen (CLI oder Defaults)
    args = parse_args()

    # --- SETUP ---
    # Basis-Pfade relativ zur Datei bestimmen
    project_root = Path(__file__).parent.resolve()
    models_dir = project_root / "models"
    os.makedirs(models_dir, exist_ok=True)

    # 1. Experiment Manager initialisieren
    manager = ExperimentManager(models_dir)

    # --- CONFIGURATION ---
    # Wähle hier die Modell-Architektur: "standard" oder "resnet"
    variant_name = args.variant
    resume_mode = args.resume  # Wird True wenn --resume gesetzt ist

    # Config Dictionary (Werte aus args übernehmen)
    config = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "loss": "L1Loss",
        "dataset_overlap_train": 0.5,
        "dataset_overlap_val": 0.0,
        "scheduler_patience": 2,  # Warte 2 Epochen ohne Verbesserung
        "scheduler_factor": 0.1  # Dann reduziere LR auf 10% (z.B. 1e-4 -> 1e-5)
    }

    # Architektur-Mapping
    variants_map = {
        "standard": DoubleConv,
        "resnet": ResidualBlock
    }

    if variant_name not in variants_map:
        raise ValueError(f"Unbekannte Variante '{variant_name}'. Verfügbar: {list(variants_map.keys())}")

    # --- DATENPFADE ---
    # Wenn "All" übergeben wird, setzen wir target_dataset auf None.
    # Das signalisiert der Funktion get_all_file_pairs, dass ALLE Ordner durchsucht werden sollen.
    if args.dataset == "All":
        target_dataset = None
    else:
        target_dataset = args.dataset
    data_root = project_root / "data" / "audio" / "datasets"

    # --- AUTOMATISCHES RESUME HANDLING ---
    pretrained_path = None
    # Manueller Pfad für alte Modelle (setze auf None, wenn nicht benötigt)
    # Beispiel: project_root / "models" / "altes_backup" / "best_model.pth"
    manual_path = None

    if manual_path and Path(manual_path).exists():
        pretrained_path = Path(manual_path)
        print(f"--> ⚠️ Lade manuell definiertes Modell (Legacy): {pretrained_path.name}")

    elif resume_mode:
        # Frage den Manager nach dem letzten Stand dieser Variante
        pretrained_path = manager.get_latest_checkpoint(variant_name)
        if pretrained_path:
            print(f"--> Resume aktiv. Setze fort von Run: {pretrained_path.parent.name}")
        else:
            print("--> Kein vorheriger Run gefunden. Starte neu.")

    # --- NEUEN RUN STARTEN ---
    # Bestimme schönen Namen für den Ordner (z.B. "All" oder "Berta")
    if target_dataset is None:
        dataset_label = "All"
    else:
        # Entfernt "dataset-" und macht den ersten Buchstaben groß (dataset-berta -> Berta)
        dataset_label = target_dataset.replace("dataset-", "").capitalize()

    source_model_str = str(pretrained_path) if pretrained_path else None

    # Wir erstellen IMMER einen neuen Ordner, damit die Historie sauber bleibt.
    run_dir = manager.start_new_run(
        variant=variant_name,
        dataset_name=dataset_label,
        config=config,
        resumed_from=source_model_str
    )

    best_model_path = run_dir / "best.pth"
    last_model_path = run_dir / "last.pth"

    print(f"--> Experiment Output Ordner: {run_dir}")

    # --- DATASET LADEN ---
    all_pairs = get_all_file_pairs(data_root, target_dataset)
    print(f"Gesamt: {len(all_pairs)} Datei-Paare gefunden.")

    if not all_pairs:
        print("❌ FEHLER: Keine Trainingsdaten gefunden. Pfad prüfen!")
        return

    # Split (80/20)
    random.seed(42)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.80)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"Daten Split: {len(train_pairs)} Training | {len(val_pairs)} Validierung")

    # --- DEVICE SETUP ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training auf Device: {device}")

    # --- DATA LOADER ---
    proc = AudioProcessor(sample_rate=96000)

    print("Initialisiere Datasets...")
    # Training mit Overlap für mehr Daten
    train_dataset = StreamingAudioDataset(train_pairs, proc, crop_width=256, overlap=config["dataset_overlap_train"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    # Validierung ohne Overlap (schneller & realistischer)
    val_dataset = StreamingAudioDataset(val_pairs, proc, crop_width=256, overlap=config["dataset_overlap_val"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    print(f"Chunks pro Epoche: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # --- MODELL SETUP ---
    print(f"Initialisiere Modell-Variante: {variant_name}")
    chosen_block = variants_map[variant_name]

    model = AudioUNet(n_channels=4, n_classes=4, block_class=chosen_block).to(device)

    # Gewichte laden (falls Resume)
    if pretrained_path:
        print(f"Lade Gewichte von {pretrained_path}...")
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Gewichte erfolgreich geladen.")
        except RuntimeError as e:
            print(f"❌ Fehler beim Laden der Gewichte (Architektur passt nicht?): {e}")

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )
    loss_fn = torch.nn.L1Loss()

    # --- TRAINING LOOP ---
    # Wir starten 'best_val_loss' bei Unendlich oder dem Wert aus dem alten Run (falls wir das auslesen würden).
    # Hier starten wir sauber neu für den aktuellen Run-Ordner.
    best_val_loss = float('inf')

    print(f"Starte Training über {config['epochs']} Epochen...")

    for epoch in range(config["epochs"]):
        start_time = datetime.now()

        # 1. Training
        model.train()
        train_loss = 0.0

        num_batches = len(train_loader)
        width = len(str(num_batches))

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Log alle 50 Batches
            if batch_idx % 50 == 0:
                print(f"   [Epoch {epoch + 1}] Batch {batch_idx:>{width}}/{num_batches} | Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)

        # 2. Validierung
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        # Aktuelle LR anzeigen
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   -> Learning Rate: {current_lr:.2e}")

        # Zeitmessung
        duration = datetime.now() - start_time
        print(
            f"==> End Epoch {epoch + 1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Time: {duration}")

        # 3. Speichern & Manager Update

        # 'Last' speichern wir immer
        torch.save(model.state_dict(), last_model_path)

        # 'Best' prüfen
        is_best = False
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            is_best = True
            torch.save(model.state_dict(), best_model_path)
            print(f"   🏆 Neuer Bestwert! Gespeichert in: {best_model_path.name}")

        # Manager updaten (schreibt in JSON)
        manager.update_metrics(epoch, avg_val_loss, is_best)

        # 4. Cleanup (Wichtig für MPS/Mac)
        if device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()

    print("Training abgeschlossen.")

if __name__ == "__main__":
    main()