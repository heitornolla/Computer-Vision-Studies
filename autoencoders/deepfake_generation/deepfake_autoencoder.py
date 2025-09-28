import torch.nn as nn

from utils.cae import (
    SharedEncoder,
    Decoder
)

class DeepFakeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.decoder_b = Decoder()
        self.decoder_a = Decoder()

    def forward(self, x, domain='b'):
        z = self.encoder(x)

        if domain == 'b':
            return self.decoder_b(z)
        elif domain == 'a':
            return self.decoder_a(z)
        else:
            raise ValueError('Domain not configured')
