import torch
import torch.nn as nn

from gan.discriminator import Discriminator
from gan.generator import Generator

def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100

    x = torch.randn((N, in_channels, H, W))
    z = torch.randn((N, z_dim, 1, 1))

    disc = Discriminator(in_channels, 8)
    gen = Generator(z_dim, in_channels, 8)
    
    initialize_weights(disc)
    initialize_weights(gen)

    assert disc(x).shape == (N, 1, 1, 1)
    assert gen(z).shape == (N, in_channels, H, W)


if __name__ == "__main__":
    test()
