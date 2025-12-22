import os
import shutil
from pathlib import Path

# --- KONFIGURATION ---
DIST_DIR = Path("dist/tape-simulator")
MODEL_SOURCE = Path("models/berta_best.pth")
FILES_TO_COPY = ["app.py", "requirements.txt"]
SRC_FILES = ["src/__init__.py", "src/model.py", "src/audio_processor.py"]


def create_start_scripts(dist_path):
    # Windows (.bat) - bleibt gleich
    with open(dist_path / "start_windows.bat", "w") as f:
        f.write('@echo off\n')
        f.write('echo Installiere fehlende Bibliotheken...\n')
        f.write('pip install -r requirements.txt\n')
        f.write('echo Starte Tape Simulator...\n')
        f.write('streamlit run app.py\n')
        f.write('pause\n')

    # Mac/Linux (.command) - WICHTIGE ÄNDERUNG
    # .command erzwingt das Öffnen im Terminal bei MacOS
    sh_path = dist_path / "start_mac.command"
    with open(sh_path, "w") as f:
        f.write('#!/bin/bash\n')
        # Ins Verzeichnis des Skripts wechseln (Wichtig bei .command!)
        f.write('cd "$(dirname "$0")"\n')
        f.write('echo "Installiere Bibliotheken..."\n')
        f.write('pip install -r requirements.txt\n')
        f.write('echo "Starte Tape Simulator..."\n')
        f.write('streamlit run app.py\n')

    # Ausführbar machen (rwxr-xr-x)
    os.chmod(sh_path, 0o755)


def main():
    print(f"Erstelle Release in {DIST_DIR}...")

    # 1. Clean Build
    if DIST_DIR.exists(): shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)
    (DIST_DIR / "src").mkdir()
    (DIST_DIR / "models").mkdir()

    # 2. Dateien kopieren
    for f in FILES_TO_COPY:
        if Path(f).exists(): shutil.copy(f, DIST_DIR / f)

    for f in SRC_FILES:
        if Path(f).exists(): shutil.copy(f, DIST_DIR / "src" / Path(f).name)

    # 3. Modell kopieren
    if MODEL_SOURCE.exists():
        shutil.copy(MODEL_SOURCE, DIST_DIR / "models" / MODEL_SOURCE.name)
        print("Modell kopiert.")
    else:
        print(f"ACHTUNG: Modell nicht gefunden unter {MODEL_SOURCE}!")

    # 4. Start-Skripte
    create_start_scripts(DIST_DIR)

    print("Fertig. Der Ordner 'dist/tape-simulator' ist bereit.")


if __name__ == "__main__":
    main()