import logging
from pydub import AudioSegment
import librosa
import librosa.feature
import librosa.sequence
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal
import traceback


class AudioPreprocessor:
    def __init__(self, data_root: str, sample_rate: int = 96000, chunk_length_sec: float = 2.0,
                     target_dataset: str = None):
        self.data_root = Path(data_root)
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length_sec
        self.target_dataset = target_dataset

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _has_files(self, path: Path, pattern="*.wav") -> bool:
        """Hilfsfunktion: Prüft, ob Ordner existiert und Dateien enthält."""
        return path.exists() and any(path.rglob(pattern))

    def _validate_structure(self) -> list:
        """
        Entscheidet pro Dataset: Überspringen oder volle Pipeline.
        """
        tasks = []
        datasets_dir = self.data_root / "audio" / "datasets"

        if not datasets_dir.exists():
            self.logger.error(f"Datasets Ordner nicht gefunden: {datasets_dir}")
            return []

        for dataset_path in datasets_dir.glob("dataset-*"):
            # Option Datasets zu überspringen, zB nur test-datatset wird angezeigt
            if self.target_dataset and dataset_path.name != self.target_dataset:
                continue
            # Definition aller relevanten Ordner für dieses Dataset
            raw_in = dataset_path / "tape-input"
            raw_out = dataset_path / "tape-output-recordings"

            # Ordner für die finalen Chunks
            chunk_in = dataset_path / "tape-input-chunks"
            chunk_out = dataset_path / "tape-output-recordings-chunks"

            # FALL 1: Dataset ist komplett fertig
            # (Chunks existieren für Input und Output)
            if self._has_files(chunk_in) and self._has_files(chunk_out):
                self.logger.info(f"Überspringe {dataset_path.name}: Chunks bereits vorhanden.")
                continue

            # FALL 2: Rohdaten vorhanden -> Volle Pipeline (Chunking)
            elif self._has_files(raw_in) and self._has_files(raw_out):
                self.logger.info(f"Plane Chunking für {dataset_path.name} (Rohdaten).")

                pairs = self.find_paired_files(raw_in, raw_out)
                if not pairs:
                    self.logger.warning(f"Keine Zuordnung möglich in {dataset_path.name} (Raw In -> Raw Out). Überspringe.")
                    continue

                for inp, tgt in pairs:
                    tasks.append({
                        "mode": "full_pipeline",
                        "dataset": dataset_path.name,
                        "id": inp.stem,
                        "input_path": inp,              # Quelle: Raw Input
                        "target_path": tgt,             # Quelle: Raw Output
                        "dest_chunk_in": chunk_in,      # Ziel: Input Chunks
                        "dest_chunk_out": chunk_out     # Ziel: Output Chunks
                    })

            else:
                self.logger.warning(f"Ignoriere {dataset_path.name}: Ordnerstruktur unklar oder leer.")

        self.logger.info(f"Aufgaben geplant: {len(tasks)}")
        return tasks

    def calculate_chunk_boundaries(self, duration_ms: int, chunk_duration_ms: int, chunk_spacing_ms: int) -> list[
        tuple[int, int]]:
        """
        Berechnet Start- und Endzeiten für Chunks mit Überlappung.
        Quelle: demo-audio-chunking.ipynb
        """
        chunks = []
        for start in range(0, duration_ms, chunk_spacing_ms):
            end = start + chunk_duration_ms
            if end > duration_ms:
                # Optionale Logik für den letzten Chunk (verwerfen oder kürzen)
                break
            chunks.append((start, end))
        return chunks


    def slice_and_save_chunks(self, audio: AudioSegment, output_dir: Path, file_stem: str, chunk_duration_ms: int,
                              chunk_spacing_ms: int, silence_threshold_dbfs: float = -50.0):
        """
        Zerteilt das Audio und speichert die Segmente. Überspringt Chunks, die zu leise sind.
        """
        boundaries = self.calculate_chunk_boundaries(len(audio), chunk_duration_ms, chunk_spacing_ms)

        for start, end in boundaries:
            chunk = audio[start:end]
            
            # Prüfen, ob der Chunk Stille ist
            if chunk.dBFS < silence_threshold_dbfs:
                # self.logger.debug(f"Überspringe stillen Chunk bei {start}ms ({chunk.dBFS:.2f} dBFS)")
                continue

            chunk_filename = f"{file_stem}_{start:08d}.wav"
            out_path = output_dir / chunk_filename
            chunk.export(str(out_path), format="wav")

    # TODO muss Paare anhand der Nummern finden, nicht anhand Namensersetzungen o. Ä.
    def find_paired_files(self, input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
        """
        Findet Paare von Eingabe- und Ausgabe-Dateien basierend auf der Namenskonvention.
        Quelle: demo-play-before-after-audio.ipynb
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

            # Pfad-Rekonstruktion (unter der Annahme gleicher Unterordner-Struktur)
            # relative_path = input_file.parent.relative_to(input_dir)
            # possible_output = output_dir / relative_path / out_name
            
            # Vereinfachte Suche: Wir suchen einfach im output_dir nach der Datei
            possible_output = output_dir / out_name

            if possible_output.exists():
                pairs.append((input_file, possible_output))

        return pairs

    def _save_audio(self, path: Path, data, sr):
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, data, sr)

    def process_pair(self, task: dict):
        """
        Führt die entsprechende Verarbeitung basierend auf 'mode' aus.
        """
        mode = task['mode']

        try:
            if mode == "full_pipeline":
                self.logger.info(f"Verarbeite Paar: {task['input_path'].name} -> {task['target_path'].name}")
                
                # 1. Laden
                inp_audio = AudioSegment.from_file(task['input_path'])
                tgt_audio = AudioSegment.from_file(task['target_path'])

                # 2. Chunken & Speichern (ohne Normalisierung, mit Silence Skipping)
                chunk_ms = int(self.chunk_length * 1000)
                # Wir nehmen 50% Overlap an, wie im Demo-Notebook
                chunk_spacing = chunk_ms // 2 

                self.logger.info(f"Erstelle Chunks für {task['id']}...")
                
                # Wir nutzen die dBFS des Input-Audio als Referenz für Stille.
                # Wenn der Input still ist, brauchen wir auch keinen Output-Chunk.
                # Da slice_and_save_chunks unabhängig aufgerufen wird, müssen wir sicherstellen,
                # dass wir synchrone Chunks haben.
                # Lösung: Wir berechnen die Boundaries einmal und filtern dann.
                
                boundaries = self.calculate_chunk_boundaries(len(inp_audio), chunk_ms, chunk_spacing)
                
                saved_chunks = 0
                for start, end in boundaries:
                    inp_chunk = inp_audio[start:end]
                    
                    # Silence Check auf dem Input Chunk
                    if inp_chunk.dBFS < -50.0: # Schwellenwert anpassbar
                        continue
                        
                    # Wenn Input laut genug ist, speichern wir beide Chunks
                    tgt_chunk = tgt_audio[start:end]
                    
                    chunk_filename = f"{task['id']}_{start:08d}.wav"
                    
                    inp_out_path = task['dest_chunk_in'] / chunk_filename
                    inp_chunk.export(str(inp_out_path), format="wav")
                    
                    tgt_out_path = task['dest_chunk_out'] / chunk_filename
                    tgt_chunk.export(str(tgt_out_path), format="wav")
                    
                    saved_chunks += 1
                
                self.logger.info(f"  -> {saved_chunks} Chunks erstellt (Stille übersprungen).")

        except Exception as e:
            self.logger.error(f"Fehler bei {task['id']} ({mode}): {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        self.logger.info("Starte Datenvorverarbeitung...")
        tasks = self._validate_structure()

        if not tasks:
            self.logger.info("Nichts zu tun.")
            return

        # Sicherstellen, dass Zielordner existieren (einmalig pro Run wäre effizienter, aber so ist es sicher)
        for task in tasks:
            if "dest_chunk_in" in task:
                task["dest_chunk_in"].mkdir(parents=True, exist_ok=True)
            if "dest_chunk_out" in task:
                task["dest_chunk_out"].mkdir(parents=True, exist_ok=True)

            self.process_pair(task)

        self.logger.info("Verarbeitung abgeschlossen.")


if __name__ == "__main__":
    target_dataset = "dataset-test"
    # target_dataset = None # Alle verarbeiten
    
    # Pfad zum data Ordner relativ zum Skript
    data_root = Path(__file__).parent.parent / "data"
    
    processor = AudioPreprocessor(data_root=str(data_root), target_dataset=target_dataset)
    processor.run()