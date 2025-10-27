import torch
import torch.nn as nn


LATENT_DIM = 128
BASE_CH = 32


class ConvBlock(nn.Module):
    """Purely a feature extractor, does not downsize the image"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.conv(x) + self.residual(x)
    

class ResUNetEncoder(nn.Module):
    """
    Encoder based on a ResUNet architecture.
    Assumes grayscale images.
    """
    def __init__(self, in_ch=1, base_ch=BASE_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.enc1 = ResBlock(in_ch, base_ch)
        self.enc2 = ResBlock(base_ch, base_ch*2)
        self.enc3 = ResBlock(base_ch*2, base_ch*4)
        self.enc4 = ResBlock(base_ch*4, base_ch*8)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_ch*8, latent_dim)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        latent = self.global_pool(s4).flatten(1)
        latent = self.fc(latent)
        
        return latent, (s1, s2, s3, s4)
    

class ResUNetDecoder(nn.Module):
    def __init__(self, out_ch=1, base_ch=BASE_CH, latent_dim=LATENT_DIM*2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_ch * 8)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec4 = ResBlock(base_ch*8 + base_ch*8, base_ch*4)
        self.dec3 = ResBlock(base_ch*4 + base_ch*4, base_ch*2)
        self.dec2 = ResBlock(base_ch*2 + base_ch*2, base_ch)
        self.dec1 = ResBlock(base_ch+base_ch, base_ch)

        self.final_conv = nn.Conv2d(base_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, skips):
        s1, s2, s3, s4 = skips
        x = self.fc(z).unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1)
        x = x.expand(-1, -1, s4.shape[2], s4.shape[3]) # broadcasting

        x = self.dec4(torch.cat([x, s4], dim=1))
        x = self.up(x)
        x = self.dec3(torch.cat([x, s3], dim=1))
        x = self.up(x)
        x = self.dec2(torch.cat([x, s2],dim=1))
        x = self.up(x)
        x = self.dec1(torch.cat([x, s1], dim=1))

        return self.sigmoid(self.final_conv(x))
