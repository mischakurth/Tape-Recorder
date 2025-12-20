from pathlib import Path
import logging
from pydub import AudioSegment
import os


class AudioPreprocessor:
    def __init__(self, data_root: str, sample_rate: int = 44100, chunk_length_sec: float = 2.0,
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
        Entscheidet pro Dataset: Überspringen, nur Warping oder volle Pipeline.
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

            chunk_in = dataset_path / "tape-input-chunks"
            chunk_out = dataset_path / "tape-output-recordings-chunks"

            warped_in = dataset_path / "tape-input-warped"

            # FALL 1: Dataset ist komplett fertig
            # (Warped Inputs existieren und Output Chunks existieren)
            if self._has_files(warped_in) and self._has_files(chunk_out):

                pairs = self.find_paired_files(warped_in, chunk_out)
                if pairs:
                    self.logger.info(f"Überspringe {dataset_path.name}: Bereits vollständig verarbeitet.")
                else:
                    self.logger.warning(f"Keine Zuordnung möglich in {dataset_path.name}. Überspringe.")
                continue

            # FALL 2: Chunks existieren, aber Warping fehlt (z.B. Berta, falls Warping fehlt)
            elif self._has_files(chunk_in) and self._has_files(chunk_out):
                self.logger.info(f"Plane Warping für {dataset_path.name} (Chunks vorhanden).")

                pairs = self.find_paired_files(chunk_in, chunk_out)
                if not pairs:
                    self.logger.warning(f"Keine Zuordnung möglich in {dataset_path.name}. Überspringe.")
                    continue

                for inp, tgt in pairs:
                    tasks.append({
                        "mode": "warp_only",
                        "dataset": dataset_path.name,
                        "id": inp.stem,
                        "input_path": inp,  # Quelle: Chunk
                        "target_path": tgt,  # Referenz für Warping: Output Chunk
                        "dest_path": warped_in  # Ziel: tape-input-warped
                    })

            # FALL 3: Keine Chunks -> Volle Pipeline (Alfred, Charlie)
            elif self._has_files(raw_in) and self._has_files(raw_out):
                self.logger.info(f"Plane volle Pipeline für {dataset_path.name} (Rohdaten).")

                pairs = self.find_paired_files(warped_in, chunk_out)
                if not pairs:
                    self.logger.warning(f"Keine Zuordnung möglich in {dataset_path.name}. Überspringe.")
                    continue

                for inp, tgt in pairs:
                    tasks.append({
                        "mode": "full_pipeline",
                        "dataset": dataset_path.name,
                        "id": inp.stem,
                        "input_path": inp,  # Quelle: Raw Input
                        "target_path": tgt,  # Quelle: Raw Output
                        "dest_chunk_in": chunk_in,  # Zwischenspeicher
                        "dest_chunk_out": chunk_out,  # Zwischenspeicher
                        "dest_warped": warped_in  # Endziel
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
                              chunk_spacing_ms: int):
        """
        Zerteilt das Audio und speichert die Segmente.
        Quelle: demo-audio-chunking.ipynb
        """
        boundaries = self.calculate_chunk_boundaries(len(audio), chunk_duration_ms, chunk_spacing_ms)

        for start, end in boundaries:
            chunk = audio[start:end]
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
            relative_path = input_file.parent.relative_to(input_dir)
            possible_output = output_dir / relative_path / out_name

            if possible_output.exists():
                pairs.append((input_file, possible_output))

        return pairs


    def _align_and_warp(self, input_path: Path, target_path: Path, dest_dir: Path):
        # Wird in beiden Modi benötigt
        print(f'aligning {input_path} with {target_path}')
        pass


    def normalize_audio(self, audio_segment: AudioSegment, target_dbfs: float = -0.1) -> AudioSegment:
        """
        Normalisiert ein AudioSegment auf einen Ziel-dBFS-Wert.
        Quelle: demo-loudness-normalisation.ipynb
        """
        change_in_dbfs = target_dbfs - audio_segment.max_dBFS
        return audio_segment.apply_gain(change_in_dbfs)



    def process_pair(self, task: dict):
        """
        Führt die entsprechende Verarbeitung basierend auf 'mode' aus.
        """
        mode = task['mode']

        try:
            if mode == "full_pipeline":
                # 1. Laden & Normalisieren
                inp_audio = AudioSegment.from_file(task['input_path'])
                inp_audio = self.normalize_audio(inp_audio)

                tgt_audio = AudioSegment.from_file(task['target_path'])
                tgt_audio = self.normalize_audio(tgt_audio)

                # 2. Chunken & Speichern (Pathlib kompatibel machen)
                chunk_ms = int(self.chunk_length * 1000)

                self.slice_and_save_chunks(inp_audio, task['dest_chunk_in'], task['id'], chunk_ms, chunk_ms)
                self.slice_and_save_chunks(tgt_audio, task['dest_chunk_out'], task['id'], chunk_ms, chunk_ms)

                self._align_and_warp(task['input_path'], task['target_path'], task['dest_warped'])

            elif mode == "warp_only":
                self._align_and_warp(task['input_path'], task['target_path'], task['dest_path'])

        except Exception as e:
            self.logger.error(f"Fehler bei {task['id']} ({mode}): {e}")

    def run(self):
        self.logger.info("Starte Datenvorverarbeitung...")
        tasks = self._validate_structure()

        if not tasks:
            self.logger.info("Nichts zu tun.")
            return

        # Sicherstellen, dass Zielordner existieren (einmalig pro Run wäre effizienter, aber so ist es sicher)
        for task in tasks:
            if "dest_path" in task:  # warp_only
                task["dest_path"].mkdir(parents=True, exist_ok=True)
            if "dest_warped" in task:  # full_pipeline
                task["dest_chunk_in"].mkdir(parents=True, exist_ok=True)
                task["dest_chunk_out"].mkdir(parents=True, exist_ok=True)
                task["dest_warped"].mkdir(parents=True, exist_ok=True)

            self.process_pair(task)

        self.logger.info("Verarbeitung abgeschlossen.")


if __name__ == "__main__":
    target_dataset = "dataset-test" # oder None, bzw weglassen wenn alle Daten bearbeitet werden sollen.
    processor = AudioPreprocessor(data_root="../data", target_dataset=target_dataset)
    processor.run()