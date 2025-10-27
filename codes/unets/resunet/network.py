import torch
import torch.nn as nn

from blocks import ResUNetEncoder, ResUNetDecoder

LATENT_DIM = 128
BASE_CH = 32

class FingerprintReconstructor(nn.Module):
    """
    Dual Encoder ResUNet-based architecture
    Goal is fingerprint reconstruction based on high and low pass filters
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.cartoon_encoder = ResUNetEncoder(1, base_ch=BASE_CH, latent_dim=LATENT_DIM)
        self.texture_encoder = ResUNetEncoder(1, base_ch=BASE_CH, latent_dim=LATENT_DIM)

        self.decoder = ResUNetDecoder(1, base_ch=32, latent_dim=LATENT_DIM*2)

    def forward(self, cartoon, texture):
        c_latent, c_skips = self.cartoon_encoder(cartoon)
        t_latent, t_skips = self.texture_encoder(texture)

        z = torch.cat((c_latent, t_latent), dim=1)
        
        skips = tuple((c+t)/2 for c, t in zip(c_skips, t_skips))

        recon = self.decoder(z, skips)
        return recon
