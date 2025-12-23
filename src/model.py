import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResidualBlock(nn.Module):
    """(Convolution => [BN] => ReLU => Convolution => [BN]) + Residual Connection => ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Hauptpfad: Wichtig ist, dass das letzte ReLU fehlt, um erst zu addieren
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut: Passt Dimensionen an, falls Input != Output Channels
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        else:
            self.shortcut = nn.Identity()

        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)

        # Element-wise Addition (Skip Connection)
        out += residual

        return self.final_relu(out)


class AudioUNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, base_channels=32, block_class=DoubleConv):
        """
        Args:
            n_channels: Anzahl Input Kanäle (z.B. 4 für Stereo Real/Imag)
            n_classes: Anzahl Output Kanäle (z.B. 4 für Stereo Real/Imag)
            base_channels: Basis-Filtergröße (skaliert das Netz)
            block_class: Die Klasse für die Convolutions (DoubleConv oder ResidualBlock)
        """
        super(AudioUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Helper Funktion, um den gewählten Block zu initialisieren
        def block(in_c, out_c):
            return block_class(in_c, out_c)

        bc = base_channels

        # Encoder (Downsampling)
        self.inc = block(n_channels, bc)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), block(bc, bc * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), block(bc * 2, bc * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), block(bc * 4, bc * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), block(bc * 8, bc * 16))

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(bc * 16, bc * 8, kernel_size=2, stride=2)
        self.conv1 = block(bc * 16, bc * 8)

        self.up2 = nn.ConvTranspose2d(bc * 8, bc * 4, kernel_size=2, stride=2)
        self.conv2 = block(bc * 8, bc * 4)

        self.up3 = nn.ConvTranspose2d(bc * 4, bc * 2, kernel_size=2, stride=2)
        self.conv3 = block(bc * 4, bc * 2)

        self.up4 = nn.ConvTranspose2d(bc * 2, bc, kernel_size=2, stride=2)
        self.conv4 = block(bc * 2, bc)

        # Output Layer
        self.outc = nn.Conv2d(bc, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)

        # Padding Logik für ungerade Dimensionen (Input Size Unabhängigkeit)
        diffY = x4.size()[2] - x.size()[2]
        diffX = x4.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        return self.outc(x)