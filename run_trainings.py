import subprocess
import sys
import time
from datetime import datetime

experiments = [
        # Experiment 1
        {
            "variant": "standard",
            "dataset": "All",
            "epochs": 10,
            "batch_size": 24,
            "resume": False
        },
        # Experiment 2
        {
            "variant": "resnet",
            "dataset": "All",
            "epochs":7,
            "batch_size": 24,
            "resume": True  # Setze True, falls du fortsetzen willst
        },
        # Experiment 3
        {
            "variant": "standard",
            "dataset": "All",
            "epochs": 10,
            "batch_size": 20,  # Kleinerer Batch da "All" evtl. mehr Varianz hat
            "resume": False
        },
    ]

def run_experiments():
    # --- KONFIGURATION DER EXPERIMENTE ---

    print(f"🚀 Starte Queue mit {len(experiments)} Experimenten...\n")
    total_start = datetime.now()

    for i, exp in enumerate(experiments):
        print("=" * 60)
        print(f"EXPERIMENT {i + 1}/{len(experiments)}: {exp['variant']} @ {exp['dataset']}")
        print("=" * 60)

        # Kommando zusammenbauen (nutzt 'uv run' wie in deiner Umgebung)
        cmd = [
            "uv", "run", "train.py",
            "--variant", exp["variant"],
            "--dataset", exp["dataset"],
            "--epochs", str(exp["epochs"]),
            "--batch_size", str(exp["batch_size"]),
            # Weitere Parameter hier ergänzen falls nötig (z.B. --lr)
        ]

        if exp["resume"]:
            cmd.append("--resume")

        try:
            # check=True wirft Fehler, wenn das Training crasht
            subprocess.run(cmd, check=True)
            print(f"\n✅ Experiment {i + 1} erfolgreich abgeschlossen.\n")

            # Kurze Pause zum Abkühlen/Speicherbereinigen
            time.sleep(5)

        except subprocess.CalledProcessError as e:
            print(f"\n❌ FEHLER in Experiment {i + 1}. Abbruchcode: {e.returncode}")
            # Entscheidung: Weitermachen oder alles stoppen?
            # Hier: Wir machen weiter mit dem nächsten Experiment.
            print("Setze fort mit nächstem Experiment...\n")
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n🛑 Abbruch durch Nutzer.")
            sys.exit(0)

    total_duration = datetime.now() - total_start
    print("=" * 60)
    print(f"🏁 Alle Experimente beendet. Gesamtdauer: {total_duration}")


if __name__ == "__main__":
    run_experiments()