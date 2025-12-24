import torch
from torch.utils.data import DataLoader
from pathlib import Path
import os
import sys
import json
from datetime import datetime

# Damit Imports funktionieren, Pfad zum Root hinzufügen
sys.path.append(str(Path(__file__).parent.parent))

from src.experiment_manager import ExperimentManager
from src.model import AudioUNet, ResidualBlock, DoubleConv
from src.audio_processor import AudioProcessor
from src.dataset import StreamingAudioDataset
from train import get_all_file_pairs


def evaluate_model(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            steps += 1

    return total_loss / steps if steps > 0 else float('inf')


def main():
    # --- SETUP ---
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    benchmark_file = project_root / "benchmark.json"

    data_root = project_root / "data" / "audio" / "datasets"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"--- BENCHMARK START ---")
    print(f"Device: {device}")

    # 1. Den 'Gold Standard' Datensatz laden (Validierungs-Teil von ALL)
    print("Lade Benchmark-Datensatz (Validierungs-Set von 'All')...")

    all_pairs = get_all_file_pairs(data_root, target_dataset=None)

    if not all_pairs:
        print("FEHLER: Keine Daten gefunden.")
        return

    # Gleicher Seed wie in train.py für fairen Vergleich
    import random
    random.seed(42)
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * 0.80)
    val_pairs = all_pairs[split_idx:]  # Nur die ungesehenen 20%

    proc = AudioProcessor(sample_rate=96000)
    val_dataset = StreamingAudioDataset(val_pairs, proc, crop_width=256, overlap=0.0)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=0)

    print(f"Benchmark-Größe: {len(val_dataset)} Samples.")

    # 2. Modelle vorbereiten (Pre-Scan für Tabellenbreite)
    manager = ExperimentManager(models_dir)
    raw_runs = manager.registry

    valid_runs = []

    # Wir filtern erst und bereiten Strings vor, damit wir die Breite kennen
    for run in raw_runs:
        model_path = Path(run["best_model_path"])
        if not model_path.exists():
            continue

        # Resume-String vorbereiten
        resumed_path_raw = run.get("resumed_from")
        if resumed_path_raw and resumed_path_raw != "None":
            resumed_display = Path(resumed_path_raw).parent.name
            # Falls extrem lang, mitte kürzen
            if len(resumed_display) > 40:
                resumed_display = resumed_display[:15] + "..." + resumed_display[-20:]
        else:
            resumed_display = "-"

        dataset_name = run.get("dataset", "Unknown")
        if dataset_name is None: dataset_name = "All"

        # Daten anreichern für spätere Anzeige
        run["_display_resume"] = resumed_display
        run["_display_dataset"] = dataset_name
        run["_epochs"] = run.get("epochs_completed", 0)

        valid_runs.append(run)

    if not valid_runs:
        print("Keine trainierten Modelle gefunden.")
        return

    # 3. Dynamische Spaltenbreite berechnen
    w_id = max([len(r["id"]) for r in valid_runs] + [len("ID")]) + 2
    w_var = 10
    w_data = max([len(r["_display_dataset"]) for r in valid_runs] + [len("Data")]) + 2
    w_eps = max([len(str(r["_epochs"])) for r in valid_runs] + [len("Eps")]) + 2
    w_res = max([len(r["_display_resume"]) for r in valid_runs] + [len("Resumed From")]) + 2
    w_loss = 12

    total_width = w_id + w_var + w_data + w_eps + w_res + w_loss + 18

    # 4. Header Ausgabe
    print(f"\nTeste {len(valid_runs)} Modelle gegen den Benchmark...")
    print("=" * total_width)
    header = (
        f"{'ID':<{w_id}} | "
        f"{'Var':<{w_var}} | "
        f"{'Data':<{w_data}} | "
        f"{'Eps':<{w_eps}} | "
        f"{'Resumed From':<{w_res}} | "
        f"{'LOSS (L1)':<{w_loss}}"
    )
    print(header)
    print("=" * total_width)

    # 5. Loop & Evaluation
    results = []
    loss_fn = torch.nn.L1Loss()
    variants_map = {"standard": DoubleConv, "resnet": ResidualBlock}

    for run in valid_runs:
        try:
            # Modell laden
            block_class = variants_map[run["variant"]]
            model = AudioUNet(n_channels=4, n_classes=4, block_class=block_class).to(device)

            state_dict = torch.load(run["best_model_path"], map_location=device)
            model.load_state_dict(state_dict)

            # Rechnen
            loss = evaluate_model(model, val_loader, device, loss_fn)

            # Zeile Ausgeben
            row = (
                f"{run['id']:<{w_id}} | "
                f"{run['variant']:<{w_var}} | "
                f"{run['_display_dataset']:<{w_data}} | "
                f"{run['_epochs']:<{w_eps}} | "
                f"{run['_display_resume']:<{w_res}} | "
                f"{loss:.5f}"
            )
            print(row)

            results.append({
                "id": run["id"],
                "variant": run["variant"],
                "dataset": run["_display_dataset"],
                "epochs": run["_epochs"],
                "resumed_short": run["_display_resume"],
                "resumed_full_path": run.get("resumed_from"),  # Für JSON behalten wir den vollen Pfad
                "loss": loss
            })

        except Exception as e:
            print(f"{run['id']:<{w_id}} | FEHLER: {e}")

    # 6. Ranking Sortieren
    results.sort(key=lambda x: x["loss"])

    # Ranking Ausgabe
    print("=" * total_width)
    print("\nRANKING (Globaler Benchmark auf 'All' Val-Set):")
    for i, res in enumerate(results):
        rank_str = f"{i + 1}."
        resume_info = f"(via {res['resumed_short']})" if res['resumed_short'] != "-" else ""
        print(
            f"{rank_str:<4} {res['loss']:.5f} | {res['variant']:<{w_var}} | {res['dataset']:<{w_data}} ({res['epochs']:>{w_eps - 2}} Ep.) {resume_info} -> {res['id']}")

    # 7. JSON Export
    json_output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_dataset": "All (Validation Split 20%)",
        "device": str(device),
        "total_models": len(results),
        "ranking": []
    }

    # Wir fügen den Rank explizit hinzu
    for i, res in enumerate(results):
        entry = res.copy()
        entry["rank"] = i + 1
        # Helper-Keys für die Anzeige entfernen wir für sauberes JSON
        entry.pop("_display_resume", None)
        entry.pop("_display_dataset", None)
        entry.pop("_epochs", None)
        entry.pop("resumed_short", None)  # Wir haben ja resumed_full_path
        json_output["ranking"].append(entry)

    with open(benchmark_file, "w") as f:
        json.dump(json_output, f, indent=4)

    print(f"\n✅ Benchmark gespeichert in: {benchmark_file}")


if __name__ == "__main__":
    main()