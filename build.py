import os
import shutil
from pathlib import Path

# --- KONFIGURATION ---
DIST_DIR = Path("dist/tape-simulator")

# Dein Modell (Pfad anpassen, falls sich der Name ändert!)
MODEL_SOURCE = Path("models/charlie_best.pth")

# Ordner, die komplett kopiert werden sollen
SRC_DIRS = ["src"]

# Einzeldateien: "Quelle im Projekt" -> "Name im Zielordner"
FILE_MAPPING = {
    "app.py": "app.py",
    "requirements.txt": "requirements.txt",

    # Hier wird deine USER_README zur README.txt für den Nutzer
    "USER_README.md": "README.txt"
}


def create_start_scripts(dist_path):
    """
    Erstellt nur die Start-Skripte (die einzigen Dateien, die wir generieren).
    """
    # Windows (.bat)
    with open(dist_path / "start_windows.bat", "w", encoding="utf-8") as f:
        f.write('@echo off\n')
        f.write('echo Installiere fehlende Bibliotheken...\n')
        f.write('pip install -r requirements.txt\n')
        f.write('echo Starte Tape Simulator...\n')
        f.write('streamlit run app.py\n')
        f.write('pause\n')

    # Mac/Linux (.command)
    sh_path = dist_path / "start_mac.command"
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write('#!/bin/bash\n')
        f.write('cd "$(dirname "$0")"\n')
        f.write('echo "Installiere Bibliotheken..."\n')
        f.write('pip install -r requirements.txt\n')
        f.write('echo "Starte Tape Simulator..."\n')
        f.write('streamlit run app.py\n')

    # Ausführbar machen
    os.chmod(sh_path, 0o755)


def main():
    print(f"📦 Erstelle Release in: {DIST_DIR} ...")

    # 1. Clean Build (Alten Ordner löschen)
    if DIST_DIR.exists(): shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)

    # Unterordner erstellen
    (DIST_DIR / "models").mkdir()

    # 2. Ordner kopieren (src/)
    for folder in SRC_DIRS:
        src = Path(folder)
        dst = DIST_DIR / folder
        if src.exists():
            shutil.copytree(src, dst)
            print(f"✅ Ordner kopiert: {folder}")
        else:
            print(f"⚠️ Warnung: Ordner '{folder}' nicht gefunden!")

    # 3. Einzeldateien kopieren (App, Readme, Requirements)
    for src_name, dst_name in FILE_MAPPING.items():
        src = Path(src_name)
        dst = DIST_DIR / dst_name

        if src.exists():
            shutil.copy(src, dst)
            print(f"✅ Datei kopiert: {src_name} -> {dst_name}")
        else:
            print(f"❌ FEHLER: Datei '{src_name}' fehlt im Hauptverzeichnis!")

    # 4. Modell kopieren
    if MODEL_SOURCE.exists():
        shutil.copy(MODEL_SOURCE, DIST_DIR / "models" / MODEL_SOURCE.name)
        print(f"✅ Modell kopiert: {MODEL_SOURCE.name}")
    else:
        print(f"❌ FEHLER: Modell nicht gefunden unter {MODEL_SOURCE}")

    # 5. Start-Skripte generieren
    create_start_scripts(DIST_DIR)

    print(f"\n🎉 Fertig! Der Ordner '{DIST_DIR}' ist bereit.")


if __name__ == "__main__":
    main()