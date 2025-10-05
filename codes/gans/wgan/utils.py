import torch
import torch.nn as nn

from gan.critic import Critic
from gan.generator import Generator

def initialize_weights(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(critic, real, fake, device='cpu'):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100

    x = torch.randn((N, in_channels, H, W))
    z = torch.randn((N, z_dim, 1, 1))

    disc = Critic(in_channels, 8)
    gen = Generator(z_dim, in_channels, 8)

    initialize_weights(disc)
    initialize_weights(gen)

    assert disc(x).shape == (N, 1, 1, 1)
    assert gen(z).shape == (N, in_channels, H, W)


if __name__ == "__main__":
    test()