from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.dataset import AudioDataset
from src.model import AudioUNet
from src.audio_processor import AudioProcessor
import os
import numpy as np

# --- Hilfsfunktion zum Finden der Dateien (aus AudioProcessorUnusedOldSystem.py) ---
def find_paired_files(input_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    """
    Findet Paare von Eingabe- und Ausgabe-Dateien basierend auf der Namenskonvention.
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

        # Vereinfachte Suche: Wir suchen einfach im output_dir nach der Datei
        possible_output = output_dir / out_name

        if possible_output.exists():
            pairs.append((input_file, possible_output))

    return pairs

class PairedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, file_pairs, processor, crop_width=256):
        self.file_pairs = file_pairs
        self.processor = processor
        self.crop_width = crop_width
        self.target_height = 1040

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        input_path, target_path = self.file_pairs[idx]

        # Input laden & verarbeiten
        y_in = self.processor.load_audio(str(input_path))
        stft_in = self.processor.audio_to_stft(y_in)
        cnn_in = self.processor.stft_to_cnn_input(stft_in)
        
        # Target laden & verarbeiten
        y_tgt = self.processor.load_audio(str(target_path))
        stft_tgt = self.processor.audio_to_stft(y_tgt)
        cnn_tgt = self.processor.stft_to_cnn_input(stft_tgt)

        # Padding (Freq)
        c, h, w = cnn_in.shape
        pad_height = self.target_height - h
        if pad_height > 0:
            cnn_in = np.pad(cnn_in, ((0, 0), (0, pad_height), (0, 0)), mode='constant')
            cnn_tgt = np.pad(cnn_tgt, ((0, 0), (0, pad_height), (0, 0)), mode='constant')

        # Slicing (Zeit) - Random Crop
        # Wir müssen sicherstellen, dass wir an der gleichen Stelle schneiden!
        _, _, time_frames = cnn_in.shape
        
        # Achtung: Input und Target können leicht unterschiedlich lang sein.
        # Wir nehmen das Minimum.
        min_frames = min(cnn_in.shape[2], cnn_tgt.shape[2])
        
        if min_frames > self.crop_width:
            start = torch.randint(0, min_frames - self.crop_width, (1,)).item()
            crop_in = cnn_in[:, :, start: start + self.crop_width]
            crop_tgt = cnn_tgt[:, :, start: start + self.crop_width]
        else:
            # Padding in Zeitrichtung
            pad_time = self.crop_width - min_frames
            crop_in = np.pad(cnn_in[:, :, :min_frames], ((0, 0), (0, 0), (0, pad_time)), mode='constant')
            crop_tgt = np.pad(cnn_tgt[:, :, :min_frames], ((0, 0), (0, 0), (0, pad_time)), mode='constant')

        return torch.from_numpy(crop_in).float(), torch.from_numpy(crop_tgt).float()

def get_all_file_pairs(data_root: Path, target_dataset: str | None = None) -> list[tuple[Path, Path]]:
    """
    Sammelt alle Dateipaare aus den angegebenen Datasets.
    """
    all_pairs = []

    if target_dataset:
        datasets = [data_root / target_dataset]
    else:
        datasets = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("dataset-")]

    for dataset_path in datasets:
        if not dataset_path.exists():
            print(f"Warnung: Dataset {dataset_path} nicht gefunden.")
            continue
            
        input_dir = dataset_path / "tape-input"
        output_dir = dataset_path / "tape-output-recordings"
        
        if input_dir.exists() and output_dir.exists():
            pairs = find_paired_files(input_dir, output_dir)
            all_pairs.extend(pairs)
            print(f"Dataset {dataset_path.name}: {len(pairs)} Paare gefunden.")
        else:
            print(f"Warnung: Ordnerstruktur unvollständig in {dataset_path.name}")
            
    return all_pairs

def main():
    # 1. Config
    # Option: "dataset-test" für schnellen Test, None für alle Datasets
    target_dataset = "dataset-test" 
    # target_dataset = None 

    data_root = Path("data/audio/datasets")
    
    all_pairs = get_all_file_pairs(data_root, target_dataset)

    print(f"Gesamt: {len(all_pairs)} Trainings-Paare gefunden.")
    
    if not all_pairs:
        print("Keine Trainingsdaten gefunden.")
        return

    batch_size = 4
    lr = 1e-4
    epochs = 10

    # Check Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training auf: {device}")

    # 2. Objekte erstellen
    # Wichtig: complex128/float64 für maximale Präzision aktivieren wenn gewünscht
    proc = AudioProcessor(sample_rate=96000)

    dataset = PairedAudioDataset(all_pairs, proc, crop_width=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Modell: 4 Channels Input (Real/Imag für Stereo oder ähnliches, je nach Processor)
    # AudioProcessor liefert 4 Channels (2 für Magnitude, 2 für Phase oder Real/Imag bei Stereo?)
    # Laut AudioDataset Code: cnn_input = self.processor.stft_to_cnn_input(stft)
    # Ich nehme an 4 ist korrekt.
    model = AudioUNet(n_channels=4, n_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(loader):
            # x = clean, y = tape
            x = x.to(device) 
            y = y.to(device) 

            # Forward: Das Modell soll aus dem sauberen Signal (x) den Tape-Klang (y) erzeugen.
            # Input: Clean (x)
            # Target: Tape (y)
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.6f}")

        print(f"==> Epoch {epoch} beendet. Avg Loss: {total_loss / len(loader):.6f}")

    # 4. Modell speichern
    project_root = Path(__file__).parent.resolve()
    models_dir = project_root / "models"
    
    os.makedirs(models_dir, exist_ok=True)
    model_path = models_dir / "unet_v1.pth"
    
    torch.save(model.state_dict(), model_path)
    print(f"Modell gespeichert unter: {model_path}")


if __name__ == "__main__":
    main()