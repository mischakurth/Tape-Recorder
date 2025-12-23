import json
import os
from pathlib import Path
from datetime import datetime


class ExperimentManager:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.registry_path = models_dir / "experiments.json"
        self.current_run = None
        self.registry = self._load_registry()

    def _load_registry(self):
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []
        return []

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=4, default=str)

    def start_new_run(self, variant: str, dataset_name: str, config: dict):
        """
        Erstellt Ordner und bereitet Daten vor.
        Format: timestamp_variant_dataset (z.B. 2024-04-01_10-00_resnet_Berta)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Bereinige den Namen sicherheitshalber von Pfad-Zeichen
        safe_dataset_name = dataset_name.replace(os.sep, "-")

        run_id = f"{timestamp}_{variant}_{safe_dataset_name}"
        run_dir = self.models_dir / run_id
        os.makedirs(run_dir, exist_ok=True)

        # Wir speichern das Objekt nur im RAM (self.current_run)
        self.current_run = {
            "id": run_id,
            "variant": variant,
            "dataset": dataset_name,  # Neu: Speichern wir auch im JSON
            "timestamp": timestamp,
            "status": "running",
            "best_val_loss": float('inf'),
            "epochs_completed": 0,
            "dir": str(run_dir),
            "config": config,
            "best_model_path": str(run_dir / "best.pth"),
            "last_model_path": str(run_dir / "last.pth")
        }

        print(f"Manager: Run vorbereitet (wird erst nach Epoche 1 gespeichert): {run_id}")
        return Path(run_dir)

    def update_metrics(self, epoch, val_loss, is_best=False):
        """
        Aktualisiert Metriken. Fügt den Run erst beim ERSTEN Aufruf zur JSON hinzu.
        """
        if not self.current_run: return

        # Lazy Adding: Wenn der Run noch nicht in der Liste ist, jetzt hinzufügen
        if self.current_run not in self.registry:
            self.registry.append(self.current_run)

        self.current_run["epochs_completed"] = epoch + 1
        self.current_run["status"] = "active"

        if is_best:
            self.current_run["best_val_loss"] = val_loss

        # Erst jetzt wird physisch in die Datei geschrieben
        self._save_registry()

    def get_latest_checkpoint(self, variant: str):
        """Findet den NEUESTEN Run dieser Variante und gibt dessen BESTES Modell zurück."""
        candidates = [r for r in self.registry if r["variant"] == variant]
        if not candidates:
            return None

        # Sortieren nach Timestamp (neueste zuerst)
        candidates.sort(key=lambda x: x["timestamp"], reverse=True)
        latest = candidates[0]

        path = Path(latest["best_model_path"])
        if path.exists():
            print(f"Manager: Bestes Modell des letzten Runs gefunden: {latest['id']} (Loss: {latest['best_val_loss']})")
            return path
        return None

    def get_global_best_model(self):
        """Findet das ABSOLUT BESTE Modell aller Zeiten."""
        if not self.registry: return None

        valid_runs = [r for r in self.registry if r["best_val_loss"] != float('inf')]
        if not valid_runs: return None

        valid_runs.sort(key=lambda x: x["best_val_loss"])
        best = valid_runs[0]

        return Path(best["best_model_path"]), best