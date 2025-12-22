import os
import shutil
from pathlib import Path

# --- KONFIGURATION ---
# Wie soll der Ordner heißen, den wir zippen?
DIST_DIR = Path("dist/band-salat-demo")

# Welche Dateien sind für die App zwingend nötig?
FILES_TO_COPY = [
    "app.py",
    # "README.md", # Falls du eine Anleitung schreibst
]

# Welche Code-Dateien aus src/ werden gebraucht? (KEIN train.py, KEIN dataset.py!)
SRC_FILES = [
    "src/__init__.py",
    "src/model.py",
    "src/audio_processor_torch.py",
]

# Welches Modell soll eingepackt werden?
MODEL_SOURCE = Path("models/unet_v1_l1.pth")


def main():
    print(f"📦 Erstelle Release-Paket in: {DIST_DIR} ...")

    # 1. Alten Dist-Ordner löschen (Clean Build)
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)

    # Ordnerstruktur neu anlegen
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    (DIST_DIR / "src").mkdir(exist_ok=True)
    (DIST_DIR / "models").mkdir(exist_ok=True)

    # 2. Hauptdateien kopieren
    for file_name in FILES_TO_COPY:
        src = Path(file_name)
        dst = DIST_DIR / file_name
        if src.exists():
            shutil.copy(src, dst)
            print(f"  ✅ Kopiert: {file_name}")
        else:
            print(f"  ⚠️ WARNUNG: {file_name} nicht gefunden!")

    # 3. Source-Code kopieren
    for file_name in SRC_FILES:
        src = Path(file_name)
        dst = DIST_DIR / "src" / src.name
        if src.exists():
            shutil.copy(src, dst)
            print(f"  ✅ Kopiert: {src.name}")
        else:
            print(f"  ⚠️ WARNUNG: {file_name} nicht gefunden!")

    # 4. Modell kopieren
    if MODEL_SOURCE.exists():
        shutil.copy(MODEL_SOURCE, DIST_DIR / "models" / MODEL_SOURCE.name)
        print(f"  ✅ Modell kopiert: {MODEL_SOURCE.name}")
    else:
        print(f"  ❌ FEHLER: Modell nicht gefunden unter {MODEL_SOURCE}")

    # 5. requirements.txt erstellen (Minimalversion)
    # Wir erstellen eine saubere, nur für die Demo nötige Liste
    req_content = """streamlit
torch
torchaudio
numpy
"""
    with open(DIST_DIR / "requirements.txt", "w") as f:
        f.write(req_content)
    print("  ✅ requirements.txt erstellt.")

    # 6. Starter-Skripte erstellen (Luxus für den Nutzer)

    # Windows (.bat)
    with open(DIST_DIR / "start_windows.bat", "w") as f:
        f.write('@echo off\n')
        f.write('echo Installiere fehlende Pakete...\n')
        f.write('pip install -r requirements.txt\n')
        f.write('echo Starte App...\n')
        f.write('streamlit run app.py\n')
        f.write('pause\n')

    # Mac/Linux (.sh)
    with open(DIST_DIR / "start_mac.sh", "w") as f:
        f.write('#!/bin/bash\n')
        f.write('echo "Installiere Pakete..."\n')
        f.write('pip install -r requirements.txt\n')
        f.write('echo "Starte App..."\n')
        f.write('streamlit run app.py\n')

    # Executable machen
    os.chmod(DIST_DIR / "start_mac.sh", 0o755)

    print("  ✅ Starter-Skripte erstellt.")

    # 7. Zippen (Optional)
    shutil.make_archive("band-salat-release", 'zip', DIST_DIR)
    print(f"\n🎉 Fertig! Die Datei 'band-salat-release.zip' ist bereit zum Versenden.")


if __name__ == "__main__":
    main()