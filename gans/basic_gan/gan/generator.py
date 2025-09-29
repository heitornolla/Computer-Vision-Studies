import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, in_features):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, in_features),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.generator(x)
