from pathlib import Path
from src.experiment_manager import ExperimentManager


def main():
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"

    mgr = ExperimentManager(models_dir)
    result = mgr.get_global_best_model()

    if result is None:
        print("❌ Noch kein registriertes Modell gefunden.")
        print("Grund: Entweder wurde noch kein Training gestartet oder noch keine Epoche abgeschlossen.")
        return

    path, info = result

    print(f"Der Gewinner ist: {info['variant']}")
    print(f"Run ID: {info['id']}")
    print(f"Loss: {info['best_val_loss']}")
    print(f"Pfad: {path}")


if __name__ == "__main__":
    main()