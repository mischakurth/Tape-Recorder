import torch
from torch.utils.data import DataLoader
from src.audio_processor import AudioProcessor
from src.dataset import AudioDataset
from src.model import AudioUNet
import os


def main():
    # 1. Config
    data_dir = os.path.join("data", "raw")  # Pfad anpassen
    batch_size = 4
    lr = 1e-4
    epochs = 10

    # Check Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training auf: {device}")

    # 2. Objekte erstellen
    # Wichtig: complex128/float64 für maximale Präzision aktivieren wenn gewünscht
    proc = AudioProcessor(sample_rate=96000)

    dataset = AudioDataset(data_dir, proc, crop_width=256)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioUNet(n_channels=4, n_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, x in enumerate(loader):
            x = x.to(device)  # x ist (Batch, 4, 1040, 256)

            # --- Hier passiert normalerweise die "Magie" (Input verrauschen etc.) ---
            # Für Autoencoder-Test: Input = Target
            target = x.clone()

            # Forward
            pred = model(x)
            loss = loss_fn(pred, target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.6f}")

        print(f"==> Epoch {epoch} beendet. Avg Loss: {total_loss / len(loader):.6f}")

    # 4. Modell speichern
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/unet_v1.pth")
    print("Modell gespeichert.")


if __name__ == "__main__":
    main()