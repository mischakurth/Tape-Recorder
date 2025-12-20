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

            # Neuer Ordner für die zeitlich angepassten (lag-corrected) Inputs
            lag_corrected_in = dataset_path / "tape-input-lag-corrected"

            # Ordner für die finalen Chunks
            chunk_in = dataset_path / "tape-input-chunks"
            chunk_out = dataset_path / "tape-output-recordings-chunks"

            # FALL 1: Dataset ist komplett fertig
            # (Chunks existieren für Input und Output)
            if self._has_files(chunk_in) and self._has_files(chunk_out):
                self.logger.info(f"Überspringe {dataset_path.name}: Chunks bereits vorhanden.")
                continue

            # FALL 2: Lag-korrigierte Inputs existieren bereits, aber Chunks fehlen
            elif self._has_files(lag_corrected_in) and self._has_files(raw_out):
                self.logger.info(f"Plane Chunking für {dataset_path.name} (Lag-korrigierte Inputs vorhanden).")
                
                # Wir paaren die lag-korrigierten Inputs mit den Raw Outputs
                # (da die Outputs nicht verändert werden, nur normalisiert beim Chunken)
                pairs = self.find_paired_files(lag_corrected_in, raw_out)
                
                if not pairs:
                    self.logger.warning(f"Keine Zuordnung möglich in {dataset_path.name} (Lag-Corrected -> Raw Out). Überspringe.")
                    continue

                for inp, tgt in pairs:
                    tasks.append({
                        "mode": "chunk_only",
                        "dataset": dataset_path.name,
                        "id": inp.stem,
                        "input_path": inp,          # Quelle: Lag-Corrected Input
                        "target_path": tgt,         # Quelle: Raw Output
                        "dest_chunk_in": chunk_in,  # Ziel: Input Chunks
                        "dest_chunk_out": chunk_out # Ziel: Output Chunks
                    })

            # FALL 3: Rohdaten vorhanden -> Volle Pipeline (Lag-Korrektur + Chunking)
            elif self._has_files(raw_in) and self._has_files(raw_out):
                self.logger.info(f"Plane volle Pipeline für {dataset_path.name} (Rohdaten).")

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
                        "dest_lag_corrected": lag_corrected_in, # Ziel: Lag-Corrected Input
                        "dest_chunk_in": chunk_in,      # Ziel: Input Chunks
                        "dest_chunk_out": chunk_out     # Ziel: Output Chunks
                    })

            else:
                self.logger.warning(f"Ignoriere {dataset_path.name}: Ordnerstruktur unklar oder leer.")

        self.logger.info(f"Aufgaben geplant: {len(tasks)}")
        return tasks

    def calculate_lag(self, path1: Path, path2: Path)-> float:
        try:
            # Verwende librosa statt scipy.io.wavfile, um Warnungen bei Metadaten zu vermeiden
            # sr=None behält die originale Sample Rate bei
            audio1, fs1 = librosa.load(path1, sr=None, mono=True)
            audio2, fs2 = librosa.load(path2, sr=None, mono=True)
            
            if fs1 != fs2:
                self.logger.warning(f"Sample Rates unterscheiden sich: {fs1} vs {fs2}. Resample auf {fs1}.")
                audio2 = librosa.resample(audio2, orig_sr=fs2, target_sr=fs1)

        except Exception as e:
            self.logger.error(f"Lag Berechnung Fehler bei {path1.name}, {path2.name}: {e}")
            return -1
        
        # Nur einen Teil analysieren, um Zeit zu sparen (z.B. 20 Sekunden aus der Mitte)
        seconds_to_analyze = min(20, len(audio1) // fs1)
        center_sample = len(audio1) // 2
        slice_start = max(0, center_sample - (seconds_to_analyze * fs1) // 2)
        slice_end = min(len(audio1), center_sample + (seconds_to_analyze * fs1) // 2)
        
        sig1_slice = audio1[slice_start:slice_end]
        
        # Für das zweite Signal nehmen wir einen etwas größeren Bereich, um den Lag zu finden
        # Wir gehen davon aus, dass der Lag nicht größer als 5 Sekunden ist
        max_lag_seconds = 5
        max_lag_samples = int(max_lag_seconds * fs1)
        
        slice2_start = max(0, slice_start - max_lag_samples)
        slice2_end = min(len(audio2), slice_end + max_lag_samples)
        sig2_slice = audio2[slice2_start:slice2_end]

        # Berechnung am Ausschnitt
        correlation = signal.correlate(sig2_slice, sig1_slice, mode='valid', method='fft')
        
        # Der Index des Maximums in der Korrelation entspricht dem Versatz
        # Da wir 'valid' verwenden, müssen wir den Offset berücksichtigen
        lag_samples = np.argmax(correlation) - (slice_start - slice2_start)
        
        # Invertieren, da wir wissen wollen, wie viel wir sig1 verschieben müssen
        # Wenn lag_samples positiv ist, ist sig2 später als sig1
        
        time_shift = lag_samples / fs1
        self.logger.info(f"Versatz berechnet: {time_shift:.6f} Sekunden")
        return time_shift

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
            elif 'lagged' in input_file.name:
                # Wenn wir bereits lagged files haben, suchen wir nach dem originalen tape file
                # Annahme: input ist z.B. "-1orig_lagged.wav", output ist "-1tape.wav"
                out_name = input_file.name.replace('_lagged', '').replace('orig', 'tape')
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

    def align_audio(self, input_path: Path, target_path: Path) -> AudioSegment:
        """
        Berechnet den Lag zwischen Input und Target und gibt das verschobene Input-Audio zurück.
        """
        # 1. Versatz berechnen (input_path ist Referenz)
        # Wir wollen wissen, wie viel wir input verschieben müssen, damit es zu target passt.
        # calculate_lag gibt zurück, wie viel später target im Vergleich zu input ist.
        lag_seconds = self.calculate_lag(input_path, target_path)

        if lag_seconds == -1:
            self.logger.error("Abbruch: Lag konnte nicht berechnet werden.")
            return AudioSegment.from_file(input_path)

        # 2. Input-Datei als AudioSegment laden
        audio_input = AudioSegment.from_file(input_path)
        
        # 3. Versatz in Millisekunden umrechnen
        lag_ms = lag_seconds * 1000

        self.logger.info(f"Korrigiere Versatz von {lag_ms} ms.")

        # 4. Audio verschieben
        # Wenn lag_ms > 0, ist das Target später als der Input. 
        # Das heißt, der Input muss verzögert werden (Stille am Anfang einfügen), um zum Target zu passen.
        if lag_ms > 0:
            silence = AudioSegment.silent(duration=lag_ms, frame_rate=audio_input.frame_rate)
            audio_aligned = silence + audio_input
        
        # Wenn lag_ms < 0, ist das Target früher als der Input.
        # Das heißt, der Input ist zu spät und muss "vorgezogen" werden (Anfang abschneiden).
        elif lag_ms < 0:
            audio_aligned = audio_input[abs(lag_ms):]
            
        else:
            audio_aligned = audio_input
            
        return audio_aligned


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
                self.logger.info(f"Verarbeite Paar: {task['input_path'].name} -> {task['target_path'].name}")
                
                # 1. Lag berechnen und Input anpassen (Align)
                # Wir laden das Original-Audio, berechnen den Lag zum Target und erhalten das verschobene Input-Audio
                aligned_input_audio = self.align_audio(task['input_path'], task['target_path'])
                
                # Target Audio laden
                target_audio = AudioSegment.from_file(task['target_path'])

                # 2. Normalisieren
                aligned_input_audio = self.normalize_audio(aligned_input_audio)
                target_audio = self.normalize_audio(target_audio)
                
                # 3. Lag-Corrected Input speichern
                # Wir fügen "_lagged" zum Dateinamen hinzu
                lagged_filename = f"{task['input_path'].stem}_lagged{task['input_path'].suffix}"
                lag_corrected_path = task['dest_lag_corrected'] / lagged_filename
                lag_corrected_path.parent.mkdir(parents=True, exist_ok=True)
                aligned_input_audio.export(str(lag_corrected_path), format="wav")

                # 4. Chunken & Speichern
                chunk_ms = int(self.chunk_length * 1000)
                # Wir nehmen 50% Overlap an, wie im Demo-Notebook
                chunk_spacing = chunk_ms // 2 

                self.logger.info(f"Erstelle Chunks für {task['id']}...")
                self.slice_and_save_chunks(aligned_input_audio, task['dest_chunk_in'], task['id'], chunk_ms, chunk_spacing)
                self.slice_and_save_chunks(target_audio, task['dest_chunk_out'], task['id'], chunk_ms, chunk_spacing)

            elif mode == "chunk_only":
                self.logger.info(f"Erstelle Chunks für {task['id']} (aus existierendem Lag-Corrected Input)...")
                
                # 1. Laden & Normalisieren
                # Input ist hier bereits lag-corrected
                inp_audio = AudioSegment.from_file(task['input_path'])
                inp_audio = self.normalize_audio(inp_audio)

                tgt_audio = AudioSegment.from_file(task['target_path'])
                tgt_audio = self.normalize_audio(tgt_audio)
                
                # 2. Chunken & Speichern
                chunk_ms = int(self.chunk_length * 1000)
                chunk_spacing = chunk_ms // 2 

                # ID bereinigen, falls "_lagged" im Namen ist, damit die Chunks sauber benannt sind
                clean_id = task['id'].replace('_lagged', '')
                
                self.slice_and_save_chunks(inp_audio, task['dest_chunk_in'], clean_id, chunk_ms, chunk_spacing)
                self.slice_and_save_chunks(tgt_audio, task['dest_chunk_out'], clean_id, chunk_ms, chunk_spacing)

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
            if "dest_lag_corrected" in task:
                task["dest_lag_corrected"].mkdir(parents=True, exist_ok=True)
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