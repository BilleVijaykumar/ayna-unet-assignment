import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes, color_embed_dim=16):
        super().__init__()
        self.color_embed = nn.Embedding(20, color_embed_dim)
        self.encoder1 = DoubleConv(in_channels + color_embed_dim, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, image, color_id):
        batch_size, _, h, w = image.size()
        color_emb = self.color_embed(color_id).view(batch_size, -1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([image, color_emb], dim=1)
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d1 = self.decoder1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.decoder2(torch.cat([self.up2(d1), e1], dim=1))
        return self.final(d2)