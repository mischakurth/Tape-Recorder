from pathlib import Path
import logging


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
                input_files = sorted(list(warped_in.rglob("*.wav")))
                target_files = sorted(list(chunk_out.rglob("*.wav")))
                if len(input_files) == len(target_files):
                    self.logger.info(f"Überspringe {dataset_path.name}: Bereits vollständig verarbeitet.")
                else:
                    self.logger.warning(f"Mismatch in Chunks bei {dataset_path.name}. Überspringe.")
                continue

            # FALL 2: Chunks existieren, aber Warping fehlt (z.B. Berta, falls Warping fehlt)
            elif self._has_files(chunk_in) and self._has_files(chunk_out):
                self.logger.info(f"Plane Warping für {dataset_path.name} (Chunks vorhanden).")

                # Wir holen die Paare direkt aus den Chunk-Ordnern
                input_files = sorted(list(chunk_in.rglob("*.wav")))
                target_files = sorted(list(chunk_out.rglob("*.wav")))

                if len(input_files) != len(target_files):
                    self.logger.warning(f"Mismatch in Chunks bei {dataset_path.name}. Überspringe.")
                    continue

                for inp, tgt in zip(input_files, target_files):
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

                input_files = sorted(list(raw_in.rglob("*.wav")))
                target_files = sorted(list(raw_out.rglob("*.wav")))

                if len(input_files) != len(target_files):
                    self.logger.warning(f"Mismatch in Rohdaten bei {dataset_path.name}. Überspringe.")
                    continue

                for inp, tgt in zip(input_files, target_files):
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

    def _clean_audio(self, audio_data):
        pass

    def _align_and_warp(self, input_audio, target_audio):
        # Wird in beiden Modi benötigt
        pass

    def _create_chunks(self, audio_data):
        # Wird nur in full_pipeline benötigt
        pass

    def process_pair(self, task: dict):
        """
        Führt die entsprechende Verarbeitung basierend auf 'mode' aus.
        """
        mode = task['mode']

        try:
            if mode == "full_pipeline":
                # input_path ist ein langes Raw-File
                # 1. Laden & Cleanen
                # 2. Chunken -> Speichern in dest_chunk_in / dest_chunk_out
                # 3. Warpen der Input-Chunks gegen Output-Chunks
                # 4. Speichern in dest_warped
                # self.logger.info(f"Full Pipeline: {task['id']}")
                pass

            elif mode == "warp_only":
                # input_path ist hier bereits ein Chunk
                # 1. Laden
                # 2. _align_and_warp(input_chunk, target_chunk)
                # 3. Speichern in task['dest_path']
                # self.logger.info(f"Warping Chunk: {task['id']}")
                pass

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


    # TODO
    # Next steps: Gemini notebooks geben, in AudioProcessor diese Methoden erstellen lassen.
    # AudioProcessor umbenennen
