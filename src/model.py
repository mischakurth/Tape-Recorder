import torch.nn as nn
import torch
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


class AudioUNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=4, base_channels=32):
        super(AudioUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        bc = base_channels

        # Encoder
        self.inc = DoubleConv(n_channels, bc)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc, bc * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc * 2, bc * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc * 4, bc * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(bc * 8, bc * 16))

        # Decoder
        self.up1 = nn.ConvTranspose2d(bc * 16, bc * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(bc * 16, bc * 8)

        self.up2 = nn.ConvTranspose2d(bc * 8, bc * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(bc * 8, bc * 4)

        self.up3 = nn.ConvTranspose2d(bc * 4, bc * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(bc * 4, bc * 2)

        self.up4 = nn.ConvTranspose2d(bc * 2, bc, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(bc * 2, bc)

        # Output Layer
        self.outc = nn.Conv2d(bc, n_classes, kernel_size=1)

    def _pad_and_cat(self, x_up, x_skip):
        """
        Hilfsfunktion: Passt x_up an die Größe von x_skip an (durch Padding)
        und verkettet sie dann. Verhindert Crashes bei ungeraden Dimensionen.
        """
        diffY = x_skip.size()[2] - x_up.size()[2]
        diffX = x_skip.size()[3] - x_up.size()[3]

        # Padding: (Left, Right, Top, Bottom)
        x_up = F.pad(x_up, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        return torch.cat([x_skip, x_up], dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Upsampling Path mit sicherer Concatenation
        x = self.up1(x5)
        x = self._pad_and_cat(x, x4)  # <--- Safe Cat
        x = self.conv1(x)

        x = self.up2(x)
        x = self._pad_and_cat(x, x3)  # <--- Safe Cat
        x = self.conv2(x)

        x = self.up3(x)
        x = self._pad_and_cat(x, x2)  # <--- Safe Cat
        x = self.conv3(x)

        x = self.up4(x)
        x = self._pad_and_cat(x, x1)  # <--- Safe Cat
        x = self.conv4(x)

        return self.outc(x)